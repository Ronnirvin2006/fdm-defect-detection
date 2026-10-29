import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "paper" / "FDM_Defect_Detection_Conference_Paper.md"
IAME_OUTPUT = ROOT / "outputs" / "paper" / "IAME2026_FDM_Defect_Detection_Full_Paper.docx"
CONFUSION = ROOT / "outputs" / "figures" / "confusion_matrix.png"
TRAINING_CURVES = ROOT / "outputs" / "figures" / "training_curves.png"
FORCE_TABLE_COLUMNS = True

AUTHORS = [
    ("Ron Nirvin J", "2023511004"),
    ("Dharunkumar V", "2023511007"),
    ("Lalith Kishore D", "2023511009"),
    ("Dr. Pavithra E", "Assistant Professor"),
    ("Dr. Arun Prakash C", "Assistant Professor"),
]
AFFILIATION = (
    "Department of Production Technology\n"
    "Madras Institute of Technology, Anna University\n"
    "Chennai, India"
)


def set_font(run, size=9, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_inline(paragraph, text, size=9.5):
    parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            set_font(paragraph.add_run(part[2:-2]), size=size, bold=True)
        elif part.startswith("*") and part.endswith("*"):
            set_font(paragraph.add_run(part[1:-1]), size=size, italic=True)
        else:
            set_font(paragraph.add_run(part), size=size)


def configure_page(section):
    # Match the official IAME-2026 A4 Word template.
    section.page_width = Inches(8.2677)
    section.page_height = Inches(11.6929)
    section.top_margin = Inches(0.375)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.62)
    section.right_margin = Inches(0.62)


def set_columns(section, count=2, space_twips="360"):
    sect_pr = section._sectPr
    columns = sect_pr.find(qn("w:cols"))
    if columns is None:
        columns = OxmlElement("w:cols")
        sect_pr.append(columns)
    columns.set(qn("w:num"), str(count))
    columns.set(qn("w:space"), space_twips)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.extend([field_begin, instruction, field_end])


def add_figure(document, image_path: Path, caption: str, width=3.2):
    if not image_path.exists():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    cap = document.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(5)
    set_font(cap.add_run(caption), size=8, italic=True)


def remove_table_borders(table):
    table_properties = table._tbl.tblPr
    borders = table_properties.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        table_properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "nil")


def fill_author_cell(cell, name, role):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(1)
    set_font(paragraph.add_run(name), size=10.5, bold=True)
    if role == "Assistant Professor":
        paragraph = cell.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(1)
        set_font(paragraph.add_run(role), size=8.5, bold=False)
    for line in AFFILIATION.splitlines():
        paragraph = cell.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        set_font(paragraph.add_run(line), size=8.5)
    if role != "Assistant Professor":
        paragraph = cell.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(2)
        set_font(paragraph.add_run(role), size=8.5)


def add_author_block(document):
    table = document.add_table(rows=2, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    remove_table_borders(table)

    first_row = [
        table.cell(0, 0).merge(table.cell(0, 1)),
        table.cell(0, 2).merge(table.cell(0, 3)),
        table.cell(0, 4).merge(table.cell(0, 5)),
    ]
    second_row = [
        table.cell(1, 0).merge(table.cell(1, 2)),
        table.cell(1, 3).merge(table.cell(1, 5)),
    ]
    for cell, author in zip(first_row + second_row, AUTHORS):
        fill_author_cell(cell, *author)

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)


def add_markdown_table(document, rows, table_number):
    headers = [cell.strip() for cell in rows[0].strip("|").split("|")]
    caption_by_headers = {
        ("Class", "Total", "Train", "Validation", "Test"): "DATASET DISTRIBUTION AFTER STRATIFIED SPLITTING",
        ("Setting", "Retained value"): "TRAINING AND MODEL CONFIGURATION",
        ("Class", "Precision / Recall / F1", "n"): "PER-CLASS PERFORMANCE ON THE HELD-OUT TEST SPLIT",
        ("Predicted class", "Visual interpretation", "First operator check"): "OPERATOR GUIDANCE IMPLEMENTED IN THE DASHBOARD",
        ("Study / model", "Task and data", "Reported metric", "Timing / scope"): "COMPARISON WITH PUBLISHED FDM SYSTEMS",
    }
    caption = caption_by_headers.get(tuple(headers), "EXPERIMENTAL RESULTS")
    label = document.add_paragraph()
    label.alignment = WD_ALIGN_PARAGRAPH.CENTER
    label.paragraph_format.space_before = Pt(3)
    label.paragraph_format.space_after = Pt(1)
    label.paragraph_format.keep_with_next = True
    roman = ("I", "II", "III", "IV", "V", "VI")[table_number - 1]
    set_font(label.add_run(f"TABLE {roman}\n{caption}"), size=8)
    body = rows[2:]
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.style = "Table Grid"
    table.autofit = False
    if len(headers) == 5:
        widths = [Inches(1.02), Inches(0.49), Inches(0.49), Inches(0.62), Inches(0.49)]
    elif len(headers) == 4:
        widths = [Inches(1.28), Inches(0.52), Inches(0.52), Inches(0.79)]
    elif tuple(headers) == ("Predicted class", "Visual interpretation", "First operator check"):
        widths = [Inches(0.82), Inches(1.02), Inches(1.42)]
    elif len(headers) == 3:
        widths = [Inches(1.12), Inches(1.72), Inches(0.42)]
    elif len(headers) == 2:
        widths = [Inches(1.48), Inches(1.78)]
    else:
        widths = [Inches(3.25 / len(headers)) for _ in headers]
    for grid_column, width in zip(table._tbl.tblGrid.gridCol_lst, widths):
        grid_column.set(qn("w:w"), str(int(width.inches * 1440)))
    for index, value in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = widths[index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(paragraph.add_run(value), size=7.5, bold=True)
    for source_row in body:
        values = [cell.strip() for cell in source_row.strip("|").split("|")]
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].width = widths[index]
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_font(paragraph.add_run(value), size=7.5)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def main():
    document = Document()
    configure_page(document.sections[0])
    document.core_properties.title = "A Real-Time Vision System for Multi-Class Defect Detection in Fused Deposition Modelling Using EfficientNetB0"
    document.core_properties.author = "; ".join(name for name, _ in AUTHORS)
    document.core_properties.subject = "IAME-2026 full-length paper"

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(2)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    if "Paper Caption" not in document.styles:
        caption_style = document.styles.add_style("Paper Caption", WD_STYLE_TYPE.PARAGRAPH)
        caption_style.font.name = "Times New Roman"
        caption_style.font.size = Pt(8)

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    two_columns_started = False
    references_mode = False
    skip_source_author_block = False
    next_paragraph_is_abstract = False
    table_number = 0
    while index < len(lines):
        raw = lines[index].rstrip()
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("| "):
            table_rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_rows.append(lines[index].strip())
                index += 1
            table_headers = tuple(cell.strip() for cell in table_rows[0].strip("|").split("|"))
            new_column_tables = {
                ("Predicted class", "Visual interpretation", "First operator check"),
                ("Class", "Precision / Recall / F1", "n"),
            }
            if FORCE_TABLE_COLUMNS and table_headers in new_column_tables:
                document.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)
            table_number += 1
            add_markdown_table(document, table_rows, table_number)
            continue

        if stripped.startswith("# "):
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(6)
            set_font(paragraph.add_run(stripped[2:]), size=20, bold=False)
            add_author_block(document)
            skip_source_author_block = True
        elif skip_source_author_block and not stripped.startswith("## Abstract"):
            index += 1
            continue
        elif stripped == "## Abstract" and not two_columns_started:
            skip_source_author_block = False
            next_paragraph_is_abstract = True
            section = document.add_section(WD_SECTION.CONTINUOUS)
            configure_page(section)
            set_columns(section, 2)
            two_columns_started = True
        elif stripped == "## I. Introduction":
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
            set_font(paragraph.add_run("I. INTRODUCTION"), size=10, bold=True)
        elif stripped.startswith("## "):
            heading = stripped[3:]
            references_mode = heading == "References"
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(3)
            paragraph.paragraph_format.keep_with_next = True
            set_font(paragraph.add_run(heading.upper()), size=10, bold=True)
            if heading == "IV. Experimental Results":
                add_figure(document, CONFUSION, "Fig. 1. Confusion matrix for the 661-image held-out test split.", width=3.05)
            elif heading == "V. Discussion":
                add_figure(document, TRAINING_CURVES, "Fig. 2. Training and validation curves for the retained model.", width=3.05)
        elif stripped.startswith("### "):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.keep_with_next = True
            set_font(paragraph.add_run(stripped[4:]), size=9, bold=True, italic=True)
        elif stripped.startswith("> "):
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(5)
            set_font(paragraph.add_run(stripped[2:]), size=8, italic=True)
        elif re.match(r"^\d+\. ", stripped):
            paragraph = document.add_paragraph(style="List Number")
            paragraph.paragraph_format.left_indent = Inches(0.18)
            paragraph.paragraph_format.first_line_indent = Inches(-0.12)
            add_inline(paragraph, re.sub(r"^\d+\. ", "", stripped), size=9.5)
        else:
            paragraph = document.add_paragraph()
            if references_mode or re.match(r"^\[\d+\]", stripped):
                paragraph.paragraph_format.left_indent = Inches(0.15)
                paragraph.paragraph_format.first_line_indent = Inches(-0.15)
                paragraph.paragraph_format.space_after = Pt(1)
                add_inline(paragraph, stripped, size=7.4)
            else:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                if next_paragraph_is_abstract:
                    set_font(paragraph.add_run("Abstract—"), size=9, bold=True, italic=True)
                    add_inline(paragraph, stripped, size=9)
                    next_paragraph_is_abstract = False
                elif stripped.startswith("**Keywords:**"):
                    keyword_text = stripped.replace("**Keywords:**", "", 1).strip()
                    set_font(paragraph.add_run("Keywords—"), size=9, bold=True, italic=True)
                    add_inline(paragraph, keyword_text, size=9)
                else:
                    add_inline(paragraph, stripped, size=10)
        index += 1

    for section in document.sections:
        section.footer.is_linked_to_previous = False
        section.footer.paragraphs[0].clear()
    footer = document.sections[-1].footer.paragraphs[0]
    add_page_number(footer)
    for run in footer.runs:
        set_font(run, size=8)

    IAME_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(IAME_OUTPUT)
    print(IAME_OUTPUT)


if __name__ == "__main__":
    main()
