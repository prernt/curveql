from __future__ import annotations

import shutil
from pathlib import Path

from growthqa.classifier.train_from_meta import STAGE1_SELECTED_FEATURES, train_from_meta_csv
from growthqa.config import TRAIN_META_CSV as DEFAULT_TRAIN_META, MODEL_DIR as DEFAULT_MODELS_DIR

# Retraining always reuses the committed training dataset. The dataset itself
# (synthetic generation, merge with lab data, preprocessing and feature
# extraction) is produced once through the CLI and is not rebuilt here.


def train_classifier_from_meta_file(
    *,
    meta_csv_path: str | Path = DEFAULT_TRAIN_META,
    models_out_dir: str | Path = DEFAULT_MODELS_DIR,
    selected_features: list[str] | None = None,
) -> dict:
    """Train the Stage-1 classifier from an existing training_meta.csv.

    The output model directory is cleared and rewritten on every call so the
    saved models always reflect the current training file. The dataset
    generation pipeline is intentionally not invoked here; see the CLI command
    ``growthqa build-train-meta`` for the one-off generation step.
    """
    models_out_dir = Path(models_out_dir)
    if models_out_dir.exists():
        shutil.rmtree(models_out_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    return train_from_meta_csv(
        meta_csv=meta_csv_path,
        art_dir=models_out_dir,
        selected_features=STAGE1_SELECTED_FEATURES,
    )