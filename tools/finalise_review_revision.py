"""Embed vector figures in the exported PDF and package the review revision."""
from pathlib import Path
import shutil
import zipfile

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper/review_revision"
paper = OUT / "IAME2026_Revised_Paper.pdf"
document = pymupdf.open(paper)
replaced = 0
for page in document:
    for info in page.get_image_info(xrefs=True):
        if info["width"] < 500:
            continue
        rect = pymupdf.Rect(info["bbox"])
        # The square confusion matrix and landscape performance plot are distinct.
        name = "figure1_confusion_matrix.pdf" if info["width"] == info["height"] else "figure2_class_performance.pdf"
        vector = pymupdf.open(OUT / name)
        page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions(images=2, graphics=0, text=1)
        page.show_pdf_page(rect, vector, 0)
        vector.close()
        replaced += 1
assert replaced == 2, f"Expected two plots, found {replaced}"
assert len(document) <= 6, f"Page limit exceeded: {len(document)}"
page_count = len(document)
temporary = OUT / "_vector_paper.pdf"
document.save(temporary, garbage=4, deflate=True)
document.close()
temporary.replace(paper)

with pymupdf.open(paper) as checked:
    text = "\n".join(page.get_text() for page in checked)
    for required in ["96.52%", "Calibration", "TABLE V", "Assistant Professor", "Class-wise score (%)"]:
        assert required in text, required
    assert "p_bar" not in text and "sum_(" not in text
    for i, page in enumerate(checked, 1):
        page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(OUT / f"preview_page_{i}.png")

readme = f"""IAME 2026 - Paper 54 - Review revision

Main paper: IAME2026_Revised_Paper.pdf ({page_count} A4 pages)
Editable paper: IAME2026_Revised_Paper.docx
Editable manuscript: IAME2026_Revised_Paper.md
Response letter: Response_to_Reviewers.pdf / .docx / .md

The main PDF embeds both plots as vectors. PDF/SVG figure files are included;
Word uses 600 dpi raster fallbacks. Figure 1 uses retained confusion counts;
Figure 2 uses the retained classification report, not online images.

The measured results are unchanged. This is a manuscript revision, not a new
training run. Calibration is discussed, not experimentally measured: the original
split manifest and paired prediction arrays are unavailable in this workspace.
No comparative baseline or physical temporal-latency test has been invented.
Review the author details and conference camera-ready instructions before upload.

Sources checked on 25 October 2026 are listed in the response letter.
The previously submitted manuscript and documents remain unchanged.
"""
(OUT / "READ_ME.txt").write_text(readme)
package = OUT / "IAME2026_Paper54_Review_Revision.zip"
with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.iterdir()):
        if path.is_file() and not path.name.startswith("preview_"):
            archive.write(path, path.name)
print(f"Verified {page_count} A4 pages and {replaced} vector figures")
print(package)
