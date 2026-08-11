# Conference Submission Checklist

The manuscript is a complete IEEE-style technical draft, but it is not tied to a
specific conference yet. Complete these items before uploading it to a submission
portal.

- [ ] Replace `[Department Name]`, `[Institution Name]`, `[City, Country]`, and the email placeholder.
- [ ] Add every genuine contributor as an author in the agreed order.
- [ ] Add the project supervisor and funding acknowledgment where applicable.
- [ ] Download the exact Word or LaTeX template from the target conference.
- [ ] Transfer the reviewed content into that template and obey its page limit.
- [ ] Check whether the conference uses double-blind review and anonymize the draft if required.
- [ ] Confirm that all reported values match `outputs/reports/metrics.json` and `classification_report.txt`.
- [ ] Run `src/validate_external.py` on genuinely unseen labeled printer images if an external dataset is available.
- [ ] Do not describe the 96.52% result as cross-printer accuracy; it is an image-level held-out result.
- [ ] Check the license and attribution requirements of every dataset used for training.
- [ ] Run the conference plagiarism/similarity and AI-disclosure checks required by its policy.
- [ ] Obtain approval from all authors before submission.
- [ ] Replace the draft note beneath the author information.
- [ ] Export the final conference-formatted PDF and visually inspect every page.

Editable draft: `FDM_Defect_Detection_Conference_Paper.docx`

Review PDF: `FDM_Defect_Detection_Conference_Paper.pdf`

Version-controlled source: `FDM_Defect_Detection_Conference_Paper.md`
