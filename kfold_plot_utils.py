"""Helpers to plot loss/accuracy curves for K-fold CNN training."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

METRICS_DIR = Path("metrics")


def _normalize_history(history) -> pd.DataFrame:
    """Accept dict/lists/DataFrame and return a DataFrame with the expected columns."""
    if isinstance(history, pd.DataFrame):
        df = history.copy()
    else:
        df = pd.DataFrame(history)
    expected = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"History missing columns: {missing}")
    return df.sort_values("epoch")


def plot_loss_acc(history, title: str, out_path: Path | None = None, show: bool = False) -> Path | None:
    """Plot training/validation loss and accuracy curves."""
    df = _normalize_history(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = df["epoch"]

    axes[0].plot(epochs, df["train_loss"], "b", label="Training Loss")
    axes[0].plot(epochs, df["val_loss"], "r", label="Validation Loss")
    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, df["train_acc"], "b", label="Training Accuracy")
    axes[1].plot(epochs, df["val_acc"], "r", label="Validation Accuracy")
    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    saved_path = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        saved_path = out_path
    if show:
        plt.show()
    plt.close(fig)
    return saved_path


def plot_kfold_histories(
    fold_histories: Iterable[tuple[int, pd.DataFrame | dict]],
    model_label: str,
    out_dir: Path | str = METRICS_DIR,
    select: str = "best",
    metric: str = "val_loss",
    mode: str = "min",
) -> list[Path]:
    """
    Plot curves for K-fold training.

    Args:
        fold_histories: iterable of (fold_id, history) where history has columns epoch/train_loss/val_loss/train_acc/val_acc.
        model_label: e.g. "Model A".
        out_dir: where to save the plots.
        select: "all" plots every fold; "best" plots only the fold with best `metric`.
        metric: metric used to pick best fold (column name).
        mode: "min" (lower is better) or "max".
    """
    out_dir = Path(out_dir)
    rows = []
    normalized = []
    for fold_id, hist in fold_histories:
        df = _normalize_history(hist)
        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found in history columns: {df.columns}")
        best_value = df[metric].iloc[-1]
        rows.append({"fold": fold_id, "best": best_value})
        normalized.append((fold_id, df))

    if not normalized:
        raise ValueError("No fold histories provided.")

    if select.lower() == "best":
        reverse = mode.lower() == "max"
        best_row = sorted(rows, key=lambda r: r["best"], reverse=reverse)[0]
        to_plot = [(best_row["fold"], next(df for f, df in normalized if f == best_row["fold"]))]
    else:
        to_plot = normalized

    saved = []
    for fold_id, df in to_plot:
        title = f"{model_label} Fold {fold_id}"
        out_path = out_dir / f"{model_label.replace(' ', '_').lower()}_fold{fold_id}_curves.png"
        path = plot_loss_acc(df, title=title, out_path=out_path, show=False)
        saved.append(path)
    return saved


def load_histories_from_csv(pattern: str) -> list[tuple[int, pd.DataFrame]]:
    """
    Load fold histories from CSV files.

    pattern should include {fold}, e.g. \"metrics/model_a_fold{fold}_history.csv\".
    Each CSV must contain epoch/train_loss/val_loss/train_acc/val_acc.
    """
    out = []
    for fold in range(1, 100):  # stop when file missing
        path = Path(pattern.format(fold=fold))
        if not path.exists():
            if fold == 1:
                raise FileNotFoundError(f"No files match pattern {pattern}")
            break
        out.append((fold, pd.read_csv(path)))
    return out


if __name__ == "__main__":
    # Example usage for quick manual testing.
    dummy = {
        "epoch": [1, 2, 3],
        "train_loss": [0.9, 0.7, 0.5],
        "val_loss": [1.0, 0.8, 0.6],
        "train_acc": [0.6, 0.7, 0.8],
        "val_acc": [0.55, 0.65, 0.75],
    }
    plot_loss_acc(dummy, "Example Model", out_path=METRICS_DIR / "example_curves.png")
