import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from inspect_dataset import IMAGE_EXTS


ALIASES = {
    "bed adhesion": "Bed_adhesion_failure",
    "bed_adhesion": "Bed_adhesion_failure",
    "bedadhesion": "Bed_adhesion_failure",
    "blob of death": "Blob_of_death",
    "blob_of_death": "Blob_of_death",
    "blobs": "Blobs_and_zits",
    "blobs and zits": "Blobs_and_zits",
    "blobs_and_zits": "Blobs_and_zits",
    "zits": "Blobs_and_zits",
    "cracking": "Cracking",
    "crack": "Cracking",
    "cracks": "Cracking",
    "good": "No_defect",
    "healthy": "No_defect",
    "no defect": "No_defect",
    "no_defect": "No_defect",
    "non-defected": "No_defect",
    "non_defected": "No_defect",
    "ok": "No_defect",
    "layer separation": "Layer_separation",
    "layer_separation": "Layer_separation",
    "layer shifting": "Layer_shifting",
    "layer_shifting": "Layer_shifting",
    "layer shift": "Layer_shifting",
    "layershift": "Layer_shifting",
    "nozzle clog": "Nozzle_clog",
    "nozzle_clog": "Nozzle_clog",
    "nozzle clogged": "Nozzle_clog",
    "off platform": "Off_platform",
    "off_platform": "Off_platform",
    "over extrusion": "Over_extrusion",
    "over_extrusion": "Over_extrusion",
    "overextrusion": "Over_extrusion",
    "spaghetti": "Spaghetti",
    "sringing": "Stringing",
    "stringging": "Stringing",
    "stringing": "Stringing",
    "under extrusion": "Under_extrusion",
    "under_extrusion": "Under_extrusion",
    "underextrusion": "Under_extrusion",
    "warping": "Warping",
    "z": "Z_banding",
    "z banding": "Z_banding",
    "z-banding": "Z_banding",
    "z_banding": "Z_banding",
}


def normalize_name(name: str) -> str:
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return ALIASES.get(cleaned, "")


def discover_images(sources: list[Path]) -> dict[str, list[Path]]:
    images_by_class = defaultdict(list)
    ignored = Counter()

    for source in sources:
        if not source.exists():
            ignored[str(source)] += 1
            continue

        for directory in [source, *source.rglob("*")]:
            if not directory.is_dir():
                continue
            class_name = normalize_name(directory.name)
            if not class_name:
                continue
            images = [
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ]
            images_by_class[class_name].extend(images)

    return {name: sorted(set(paths)) for name, paths in sorted(images_by_class.items())}


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple FDM defect datasets into folder-per-class format.")
    parser.add_argument("--sources", nargs="+", type=Path, required=True, help="Dataset roots to scan.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/expanded_dataset"))
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 keeps all discovered images.")
    parser.add_argument("--min-images", type=int, default=20, help="Skip classes with fewer images.")
    args = parser.parse_args()

    images_by_class = discover_images(args.sources)
    manifest = {
        "sources": [str(source) for source in args.sources],
        "output": str(args.output),
        "mode": args.mode,
        "classes": {},
        "skipped_classes": {},
    }

    for class_name, paths in images_by_class.items():
        if len(paths) < args.min_images:
            manifest["skipped_classes"][class_name] = len(paths)
            continue

        selected = paths[: args.max_per_class] if args.max_per_class > 0 else paths
        for index, src in enumerate(selected):
            suffix = src.suffix.lower()
            dst = args.output / class_name / f"{class_name}_{index:06d}{suffix}"
            link_or_copy(src, dst, args.mode)
        manifest["classes"][class_name] = len(selected)

    manifest_path = args.output / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Prepared expanded dataset: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
