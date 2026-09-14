from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

REQUIRED_MODELS = [
    "vgg16_best.keras",
    "densenet121_best.keras",
    "resnet50_best.keras",
]

MODEL_METADATA = [
    "best_models_summary.json",
    "class_names.json",
]

DATA_FILES = [
    "benchmark_results.csv",
    "preprocessing_metrics.csv",
    "evaluation_predictions.csv",
    "dataset_summary.csv",
    "roc_curves.csv",
    "pr_curves.csv",
    "auc_summary.csv",
    "confusion_matrices.csv",
    "history_vgg16.csv",
    "history_densenet121.csv",
    "history_resnet50.csv",
    "test_samples.csv",
    "misclassified_cases.csv",
]

SUPPORT_DIRS = [
    "test_samples",
    "misclassified_cases",
]


def find_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def find_dir(root: Path, dirname: str) -> Path | None:
    matches = [p for p in root.rglob(dirname) if p.is_dir()]
    return matches[0] if matches else None


def copy_support_dir(source: Path, dirname: str):
    src = find_dir(source, dirname)
    if src is None:
        return
    dst = DATA_DIR / dirname
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Installed support folder: {dirname}")


def install_from_directory(source: Path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for filename in REQUIRED_MODELS:
        src = find_file(source, filename)
        if src is None:
            missing.append(filename)
            continue
        shutil.copy2(src, MODEL_DIR / filename)
        print(f"Installed model: {filename}")

    for filename in MODEL_METADATA:
        src = find_file(source, filename)
        if src:
            shutil.copy2(src, MODEL_DIR / filename)
            print(f"Installed model metadata: {filename}")

    for filename in DATA_FILES:
        src = find_file(source, filename)
        if src:
            target_name = "latest_preprocessing_metrics.csv" if filename == "preprocessing_metrics.csv" else filename
            shutil.copy2(src, DATA_DIR / target_name)
            print(f"Installed dashboard data: {target_name}")

    for dirname in SUPPORT_DIRS:
        copy_support_dir(source, dirname)

    if missing:
        raise SystemExit("Missing required model file(s): " + ", ".join(missing))

    print("\nAll three models are installed and dashboard support data was copied where available.")


def main():
    parser = argparse.ArgumentParser(
        description="Install the Colab-exported three-model bundle into the Streamlit dashboard."
    )
    parser.add_argument("source", help="Path to three_models_for_dashboard.zip or an extracted folder")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    if source.is_dir():
        install_from_directory(source)
        return

    if source.suffix.lower() != ".zip":
        raise SystemExit("Source must be a ZIP file or extracted folder.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(tmp_dir)
        install_from_directory(tmp_dir)


if __name__ == "__main__":
    main()
