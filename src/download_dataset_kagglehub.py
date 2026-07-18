import argparse
import os
import shutil
from pathlib import Path

import kagglehub

from config import DATA_RAW, KAGGLE_CONFIG_DIR, KAGGLE_DATASET


def copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle dataset using KaggleHub.")
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--force", action="store_true", help="Re-copy cached files into data/raw.")
    args = parser.parse_args()

    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_CONFIG_DIR))
    os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_RAW / ".kagglehub_cache"))

    access_token = KAGGLE_CONFIG_DIR / "access_token"
    if not access_token.exists() and not os.getenv("KAGGLE_API_TOKEN"):
        raise FileNotFoundError(
            f"KaggleHub access token not found at {access_token}. "
            "Save your token there or set KAGGLE_API_TOKEN."
        )

    print(f"Downloading KaggleHub dataset: {args.dataset}")
    downloaded_path = Path(kagglehub.dataset_download(args.dataset))
    print(f"KaggleHub cache path: {downloaded_path}")

    if args.force and DATA_RAW.exists():
        for child in DATA_RAW.iterdir():
            if child.name != ".gitkeep":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    if DATA_RAW in downloaded_path.parents:
        print("Dataset already lives inside data/raw cache; not duplicating files.")
    else:
        copy_tree_contents(downloaded_path, DATA_RAW)
    (DATA_RAW / ".download_complete").write_text(str(downloaded_path) + "\n", encoding="utf-8")
    print(f"Dataset copied to: {DATA_RAW}")


if __name__ == "__main__":
    main()
