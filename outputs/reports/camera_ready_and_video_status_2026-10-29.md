# Camera-Ready and Video Feature Status

Date: 29 October 2026  
Paper ID: 54  
Project: A Real-Time Vision System for Multi-Class Defect Detection in Fused Deposition Modelling Using EfficientNetB0

## 1. Conference status

The acceptance email confirms **Accept with Minor Revision** and requests the final/camera-ready paper by **31 October 2026**. It asks for corrections to the reviewer comments, template and page-limit compliance, corrected figures/tables/references, and plagiarism below 15%.

The official IAME-2026 website states that the full-length paper is restricted to **a maximum of six pages including figures, tables and references**. It also states that submission is through Microsoft CMT and that the permitted similarity index is up to 15%. The website lists paper templates and the CMT portal, but the inspected pages do not state a separate video-upload requirement.

## 2. Reviewer requirements and current response

| Reviewer request | Current status |
|---|---|
| Explain the contribution beyond established EfficientNetB0, transfer learning and Grad-CAM | Addressed by framing the contribution as an application-level real-time inspection pipeline, evaluation and deployment workflow rather than a new neural architecture. |
| Compare against existing FDM defect-detection methods | Addressed in the revised paper with a comparison table. The values are reported as literature values or the present measured result where identified. |
| Reduce discussion and strengthen comparison | Addressed in the revised six-page manuscript. |
| Improve Fig. 2 labels and legends | Addressed with a clearer class-performance figure and vector PDF/SVG versions in this project. |
| Explain the five-frame rolling window | Addressed: the manuscript describes the five prediction vectors, the default sampling interval and the resulting temporal span. |
| Discuss calibration using ECE or temperature scaling | Addressed as a limitation and future evaluation. No calibration value is invented because the required paired prediction arrays and calibration split were not retained. |
| Use consistent English | Addressed using British spelling, including “Modelling” and “generalisation”. |

## 3. Paper files checked

The working revised manuscript is:

`outputs/paper/review_revision/IAME2026_Revised_Paper.pdf`

It is an A4, six-page PDF. The editable manuscript, response letter, figures and previews are stored beside it in `outputs/paper/review_revision/`. The earlier submitted PDF in `/home/ron/diagrams/IAME2026_FDM_Defect_Detection_Full_Paper.pdf` is also six pages, but it is the older version and should not be uploaded as the camera-ready paper.

Before CMT upload, the authors should still perform these human checks:

1. Run the final PDF through the institution/conference-approved similarity checker. The project does not contain a verified Turnitin or iThenticate result.
2. Open the final PDF at 100% and confirm every figure label, table entry, author name and affiliation.
3. Upload the revised manuscript to CMT and include the response letter if the portal requests it.
4. Confirm author metadata, registration and any camera-ready copyright or declaration form in CMT.
5. Keep the manuscript at six pages or fewer after the final export.

## 4. Video analysis feature

The Streamlit app now has an **Upload Video** tab. It accepts MP4, MOV, AVI and WEBM files. For a short clip it:

- samples up to 36 evenly spaced frames;
- classifies each sampled frame with the saved eight-class EfficientNetB0 model;
- reports the overall result using the mean class probabilities;
- lists defect classes observed in the sampled frames;
- displays a timestamped prediction table and sampled-frame contact sheet;
- displays the existing defect-specific feedback;
- deletes the temporary uploaded video after analysis.

This is a clip-level classification and review tool. It is not object detection, pixel segmentation or a guarantee that every brief transient defect will be captured. A clear, short, stable view of the print gives the most useful result. The existing Live Webcam tab remains separate and is still subject to browser/WebRTC support and Streamlit hosting limitations.

## 5. Streamlit deployment diagnosis

The deployment screenshot showed dependency installation, not model training. The large TensorFlow, OpenCV and scientific-Python downloads can take several minutes on a first build. Earlier deployment failures were caused by Python/TensorFlow compatibility and then by an incompatible NumPy pin. The repository now pins Python 3.13 through `runtime.txt`, uses the deployment-specific `src/requirements.txt`, and pins NumPy 2.1.3 and pandas 2.2.3 for TensorFlow 2.20.0 compatibility.

After the latest repository commit is visible to Streamlit Cloud, use **Reboot app** or **Clear cache and reboot** once if the old dependency build is still displayed. A successful build should move past “Your app is in the oven” and show the application page.

## 6. Evidence boundary

The reported model result remains the retained test result: 96.52% accuracy and 96.50% weighted F1 on the 661-image test split. The new video tab is an application feature and does not create a new accuracy claim. No new baseline experiment, calibration measurement or video benchmark has been run yet.
