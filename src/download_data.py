"""Télécharge le Hard Hat dataset depuis Kaggle et le copie dans data/raw."""
import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET = "andrewmvd/hard-hat-detection"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print("Téléchargé dans :", cache_path)
    shutil.copytree(cache_path, RAW, dirs_exist_ok=True)
    print("Copié dans :", RAW)


if __name__ == "__main__":
    main()