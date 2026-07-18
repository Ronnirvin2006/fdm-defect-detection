# Dataset Expansion Research

The current trained model uses the Kaggle FDM 3D Printing Defect Dataset and detects five classes:

- Cracking
- Layer_shifting
- Off_platform
- Stringing
- Warping

To detect more FDM defect types, the model must be retrained with labeled examples for those new classes. The current model should not be claimed to detect under-extrusion, over-extrusion, nozzle clogging, blobs/zits, healthy/no-defect, or layer separation until those labels are included in training.

## Useful Additional Dataset Sources

1. Roboflow Universe: 3D Printing Defect Classification

   Link: https://universe.roboflow.com/project-jkfnh/3d-printing-defect-classification

   Notes: Public listing reports a large classification dataset with about 22,048 images. Visible classes include No Defect, Warping, Nozzle Clog, Layer Separation, Under Extrusion, and Blobs and Zits. This is useful for expanding the project beyond the current five Kaggle classes.

2. Roboflow Universe: Defects Classification in 3D Printing

   Link: https://universe.roboflow.com/jfdsjmfsd/defects-classification-in-3d-printing-fwsu3-eyaxl

   Notes: Public listing shows classes such as Over Extrusion, Z-Banding, Bed Adhesion, Under Extrusion, Nozzle Clog, Blobs and Zits, Spaghetti, Stringing, and No Defect. This is useful if the project should detect more practical printer failure types.

3. Kaggle: 3D Printing Errors

   Link: https://www.kaggle.com/datasets/nimbus200/3d-printing-errors

   Notes: Public listing describes images recorded during printing with classes such as GOOD, STRINGING, and UNDEREXTRUSION. This can add healthy/no-defect and under-extrusion coverage.

4. Kaggle Competition: Early Detection of 3D Printing Issues

   Link: https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues

   Notes: Public listing describes close-up nozzle-camera images from seven 3D printers, where prints are either good or purposefully set to produce under-extrusion. This is useful for early-warning and under-extrusion detection.

5. Zenodo: Image Dataset for Defect Detection in Fused Filament Fabrication

   Link: https://zenodo.org/records/14712897

   Notes: Public listing describes high-resolution and cropped images captured from three camera angles, including production defects such as extrusion errors and layer shifts, plus defect-free prints.

## Recommended Expanded Class Set

If more data is added, a practical expanded classifier could use:

- No_defect
- Cracking
- Layer_shifting
- Off_platform
- Stringing
- Warping
- Under_extrusion
- Over_extrusion
- Nozzle_clog
- Blobs_and_zits
- Layer_separation
- Bed_adhesion_failure
- Z_banding

## Integration Plan

1. Download additional datasets.
2. Convert all datasets into one folder-per-class format.
3. Normalize class names consistently.
4. Remove duplicates and visually inspect confusing labels.
5. Retrain with the same pipeline.
6. Re-run evaluation, ROC-AUC, inference-time testing, Grad-CAM, and prediction verification.

## Important Limitation

Mixing datasets from different cameras, lighting conditions, printers, and label definitions can improve generalization, but it can also create label noise. The expanded dataset should be inspected carefully before claiming improved real-world performance.
