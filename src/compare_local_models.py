"""Compare a local candidate on its recorded validation split."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from config import MODELS_DIR
from validate_external import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Override the source model used as the baseline')
    args = parser.parse_args()
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    metrics = json.loads((args.run / 'metrics.json').read_text())
    names = json.loads((MODELS_DIR / 'class_names.json').read_text())
    if names != metrics['class_names']:
        raise ValueError('Candidate and active model class mappings differ')
    split = json.loads((args.run / 'split.json').read_text())
    paths = split['validation']
    if set(paths) & set(split['train']):
        raise ValueError('Training and validation paths overlap')
    labels = np.array([names.index(Path(p).parent.name) for p in paths])
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode, num_parallel_calls=2).batch(4).prefetch(1)
    output = args.run / 'comparison'
    output.mkdir(exist_ok=True)
    summary, probabilities = {}, {}
    settings_file = args.run / 'settings.json'
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    baseline = args.baseline or Path(settings.get('model', MODELS_DIR / 'best_model.keras'))
    for tag, path in [('original', baseline), ('candidate', Path(metrics['model']))]:
        print('Evaluating', tag, path, flush=True)
        model = tf.keras.models.load_model(path, compile=False)
        if model.output_shape[-1] != len(names):
            raise ValueError('Model output width mismatch')
        probs = model.predict(ds, verbose=2)
        probabilities[tag] = probs
        pred = probs.argmax(axis=1)
        report = classification_report(labels, pred, labels=range(len(names)), target_names=names, output_dict=True, zero_division=0)
        summary[tag] = {'model':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'accuracy':float(np.mean(pred == labels)), 'loss':float(log_loss(labels, probs, labels=range(len(names)))),
                        'macro_f1':report['macro avg']['f1-score'], 'weighted_f1':report['weighted avg']['f1-score'],
                        'per_class':{name:report[name] for name in names}}
        cm = confusion_matrix(labels, pred, labels=range(len(names)))
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
               xlabel='Predicted class', ylabel='Actual class', title=f'{tag.title()}: grouped validation (not independent test)')
        plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
        for row in range(len(names)):
            for col in range(len(names)):
                ax.text(col, row, str(cm[row,col]), ha='center', va='center', color='white' if cm[row,col] > cm.max()/2 else 'black')
        fig.tight_layout()
        fig.savefig(output / f'{tag}_confusion.png', dpi=160)
        plt.close(fig)
        np.savetxt(output / f'{tag}_confusion.csv', cm, delimiter=',', fmt='%d', header=','.join(names))
        del model
        tf.keras.backend.clear_session()
    np.savez_compressed(output / 'predictions.npz', labels=labels, paths=np.array(paths), **probabilities)
    with (output / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['path','source_path','actual','original_prediction','candidate_prediction','candidate_confidence','candidate_correct'])
        for i, path in enumerate(paths):
            pred = int(probabilities['candidate'][i].argmax())
            writer.writerow([path,str(Path(path).resolve()),names[labels[i]],names[int(probabilities['original'][i].argmax())],names[pred],float(probabilities['candidate'][i,pred]),pred == labels[i]])
    note = 'Same grouped validation split; reused for model selection. Prior-model training overlap is unknown. These are not independent test results.'
    (output / 'metrics.json').write_text(json.dumps({'note':note,'images':len(paths),**summary},indent=2))
    cm = confusion_matrix(labels, probabilities['candidate'].argmax(axis=1), labels=range(len(names)))
    mistakes = sorted([(int(cm[i,j]), names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j], reverse=True)
    lines = ['# Local Model Comparison','',note,'','| Metric | Original | Candidate |','|---|---:|---:|']
    for key in ['accuracy','macro_f1','weighted_f1']:
        lines.append(f"| {key} | {summary['original'][key]:.2%} | {summary['candidate'][key]:.2%} |")
    lines += ['', '## Largest Candidate Confusions',''] + [f'- {actual} predicted as {pred}: {count} images.' for count,actual,pred in mistakes[:8]]
    lines += ['', '## Decision','', 'Candidate is ready for controlled comparison on new print photos/videos. Keep the active model until external checks pass. Do not compare these scores directly with the historical 96.52% test result.']
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
