# Capture and Retrain Safely

The live dashboard now has a **Capture training sample** panel. It saves a manually
verified camera frame under `data/captured/<label>/`.

## Capture rules

1. Launch the local dashboard and open the live camera.
2. Select `valid_print_scene` for a clear build plate, nozzle area and printed part.
3. Select `invalid_scene` for a face, wall, ceiling, empty unrelated view, blur or severe lighting problem.
4. Select a defect class only when the defect is visually confirmed by a person.
5. Capture different prints, camera distances, lighting conditions and print stages.
6. Never use the model's prediction as the label. That would amplify its existing mistakes.

The context labels are separate from defect labels. They should eventually train a
separate context model; they must not be added as defect classes in the EfficientNetB0
defect model.

## Recommended minimum collection

Collect at least 200-500 manually verified images per defect class, 500 or more
`No_defect` images, and a balanced set of `valid_print_scene` and `invalid_scene`
images. Capture data from multiple print jobs and sessions. Do not extract random
neighbouring frames from one video into both training and test sets.

## Retraining the defect model

Merge the captured defect folders with the existing expanded dataset, inspect the
class counts, and split by print/job rather than by adjacent frames. Then run the
existing trainer on the final class-folder root:

```bash
FDM_DATA_RAW=/path/to/final_defect_dataset \
.venv/bin/python src/train.py \
  --architecture efficientnet_b0 \
  --epochs 30 \
  --fine-tune-epochs 15 \
  --batch-size 16 \
  --augmentation strong \
  --mixed-precision
```

Compare the new model with the current model using per-class recall, macro F1,
confusion matrix, calibration, and an unseen camera-session test set. Keep the old
model until the new model improves the difficult classes without increasing false
defect predictions on valid `No_defect` images.

## Context model

The current heuristic scene gate is a temporary safety layer. A production-quality
context check should train a separate binary classifier on `valid_print_scene` and
`invalid_scene`, with invalid examples containing faces, walls, empty rooms, hands,
dark views and unrelated objects. The context output should remain separate from the
eight-class defect output.
