import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from config import DATA_RAW, KAGGLE_CONFIG_DIR, KAGGLE_DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle FDM defect dataset.")
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    marker = DATA_RAW / ".download_complete"
    if marker.exists() and not args.force:
        print(f"Dataset already downloaded in {DATA_RAW}")
        return

    token_path = KAGGLE_CONFIG_DIR / "kaggle.json"
    if not token_path.exists():
        raise FileNotFoundError(
            f"Kaggle token not found at {token_path}. "
            "Create a Kaggle API token and place kaggle.json there."
        )

    print(f"Downloading Kaggle dataset: {args.dataset}")
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(KAGGLE_CONFIG_DIR)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(DATA_RAW),
    ]
    subprocess.run(cmd, check=True, env=env)

    for zip_path in DATA_RAW.glob("*.zip"):
        print(f"Extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(DATA_RAW)

    marker.write_text("ok\n", encoding="utf-8")
    print(f"Dataset ready at {DATA_RAW}")


if __name__ == "__main__":
    main()
