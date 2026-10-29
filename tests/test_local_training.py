import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from finetune_local import load_fixed_split, training_weights, atomic_save, TimeLimit


class TrainingChecks(unittest.TestCase):
    def test_sqrt_weights_are_less_extreme(self):
        labels = np.array([0] * 100 + [1] * 4)
        balanced = training_weights(labels, 'balanced')
        softened = training_weights(labels, 'sqrt')
        self.assertLess(softened[1] / softened[0], balanced[1] / balanced[0])
        self.assertAlmostEqual(np.mean([softened[int(i)] for i in labels]), 1)
        self.assertEqual(training_weights(labels, 'none'), {0:1., 1:1.})

    def test_alias_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'image.jpg'
            image.touch()
            alias = root / 'alias.jpg'
            alias.symlink_to(image)
            manifest = root / 'split.json'
            manifest.write_text(json.dumps({'train':[str(image)], 'validation':[str(alias)]}))
            with self.assertRaises(ValueError):
                load_fixed_split(manifest, [str(image)], [root.name])

    def test_failed_save_preserves_checkpoint(self):
        class BrokenModel:
            def save(self, path):
                path.write_text('incomplete')
                raise RuntimeError('simulated disk failure')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'latest.keras'
            path.write_text('previous checkpoint')
            with self.assertRaises(RuntimeError):
                atomic_save(BrokenModel(), path)
            self.assertEqual(path.read_text(), 'previous checkpoint')

    def test_time_budget_stops_training(self):
        callback = TimeLimit(1)
        class Model:
            stop_training = False
        model = Model()
        callback.set_model(model)
        with patch('finetune_local.time.monotonic', return_value=callback.deadline + 1):
            callback.on_train_batch_end(1)
        self.assertTrue(model.stop_training)
        self.assertTrue(callback.expired)


if __name__ == '__main__':
    unittest.main()
