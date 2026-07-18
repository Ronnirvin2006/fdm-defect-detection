from collections import Counter
from pathlib import Path

from PIL import Image

from config import DATA_RAW, REPORTS_DIR


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_root(raw_dir: Path) -> Path:
    candidates = []
    for path in [raw_dir, *raw_dir.rglob("*")]:
        if not path.is_dir():
            continue
        child_dirs = [p for p in path.iterdir() if p.is_dir()]
        image_children = [
            child for child in child_dirs
            if any(file.suffix.lower() in IMAGE_EXTS for file in child.rglob("*"))
        ]
        if len(image_children) >= 2:
            candidates.append((path, len(image_children)))
    if not candidates:
        raise FileNotFoundError(f"Could not find class folders under {raw_dir}")
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def main() -> None:
    class_root = find_class_root(DATA_RAW)
    rows = []
    corrupt = []
    sizes = Counter()

    for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        ok_count = 0
        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    sizes[img.size] += 1
                ok_count += 1
            except Exception as exc:
                corrupt.append(f"{image_path}: {exc}")
        rows.append((class_dir.name, ok_count))

    lines = [
        "# Dataset Inspection",
        "",
        f"Detected class root: `{class_root}`",
        "",
        "## Class Counts",
        "",
        "| Class | Images |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in rows)
    lines.extend([
        "",
        "## Most Common Image Sizes",
        "",
        "| Size | Count |",
        "|---|---:|",
    ])
    lines.extend(f"| {width}x{height} | {count} |" for (width, height), count in sizes.most_common(10))
    lines.extend(["", f"Corrupt images found: {len(corrupt)}"])
    if corrupt:
        lines.extend(["", "## Corrupt Files", ""])
        lines.extend(corrupt[:100])

    report_path = REPORTS_DIR / "dataset_inspection.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
