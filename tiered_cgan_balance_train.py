"""Pipeline to balance classes with tiered CGAN augmentation and train models D/E.

Rules:
- If a class has <500 reais: gerar ate 500
- Entre 500 e 1000: gerar ate 1000
- Entre 1000 e 1500: gerar ate 1500
- >=1500: nao gera

Depois une reais+augmentadas, calcula ratios e treina Model D (11 classes anomalias)
e Model E (12 classes completas).
"""
from __future__ import annotations

import argparse
import math
import random
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR = Path("/content/InfraredSolarModules") if Path("/content").exists() else REPO_ROOT / "InfraredSolarModules"
DATA_URL = "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip"
BASE_IMAGE_DIR = DATA_DIR / "images"
TRAIN_CSV = REPO_ROOT / "full_train_data_list.csv"
FINAL_TEST_CSV = REPO_ROOT / "final_test_data_list.csv"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"
CGAN_WEIGHTS = REPO_ROOT / "cgan_generated_outputs" / "cgan_generator_minority_classes.pth"
GAN_AUG_DIR = DATA_DIR / "cgan_tiered_augmented"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
LATENT_DIM = 128
IMAGE_SIZE = (64, 64)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dataset() -> None:
    """Download InfraredSolarModules only if missing."""
    if BASE_IMAGE_DIR.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.with_suffix(".zip")
    print("Downloading InfraredSolarModules (first run only)...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR.parent)
    print(f"Dataset extracted to {DATA_DIR}")


def load_dataframes():
    ensure_dataset()
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["filename"] = train_df["path"].apply(lambda p: Path(p).name)
    train_df["path"] = train_df["filename"].apply(lambda n: BASE_IMAGE_DIR / n)
    class_pairs = train_df[["class_name", "label"]].drop_duplicates().sort_values("label")
    classes_map = {row.class_name: int(row.label) for row in class_pairs.itertuples()}
    idx_to_class = {v: k for k, v in classes_map.items()}
    anomaly_classes = sorted([c for c in classes_map if c != "No-Anomaly"])
    classes_map_B = {cls: idx for idx, cls in enumerate(anomaly_classes)}
    return train_df, classes_map, idx_to_class, anomaly_classes, classes_map_B


def load_final_test_df() -> pd.DataFrame:
    """Carrega o CSV de teste final e ajusta caminhos para o diretório local."""
    ensure_dataset()
    test_df = pd.read_csv(FINAL_TEST_CSV)
    test_df["filename"] = test_df["path"].apply(lambda p: Path(p).name)
    test_df["path"] = test_df["filename"].apply(lambda n: BASE_IMAGE_DIR / n)
    return test_df


base_transform = T.Compose(
    [
        T.Resize(IMAGE_SIZE),
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
)
train_transform = base_transform
test_transform = base_transform


class SolarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def make_loader(df, transform, batch_size=256, shuffle=False, sampler=None):
    return DataLoader(
        SolarDataset(df, transform),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


class NetworkCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        dummy_input = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            flattened_size = torch.flatten(x, 1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1 * 64 * 64),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        x = torch.cat((self.label_emb(labels), noise), dim=1)
        out = self.model(x)
        return out.view(noise.size(0), 1, 64, 64)


def load_trained_generator(num_classes: int):
    if not CGAN_WEIGHTS.exists():
        raise FileNotFoundError(f"CGAN weights not found: {CGAN_WEIGHTS}")
    generator = ConditionalGenerator(LATENT_DIM, num_classes).to(DEVICE)
    state = torch.load(CGAN_WEIGHTS, map_location=DEVICE)
    generator.load_state_dict(state)
    generator.eval()
    return generator


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    class_counts = df["label"].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = df["label"].map(class_weights).astype(float)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights.values, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def evaluate_loader_metrics(model: nn.Module, loader: DataLoader, criterion: nn.Module | None = None) -> dict:
    """Calcula loss, accuracy e f1 num loader completo."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total += labels.size(0)
    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0, "f1_weighted": 0.0}
    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / total if criterion is not None else 0.0,
        "accuracy": accuracy_score(labels_np, preds_np),
        "f1_macro": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "f1_micro": f1_score(labels_np, preds_np, average="micro", zero_division=0),
        "f1_weighted": f1_score(labels_np, preds_np, average="weighted", zero_division=0),
    }


def tier_target(count: int, thresholds=(500, 1000, 1500)) -> int:
    if count < thresholds[0]:
        return thresholds[0]
    if count < thresholds[1]:
        return thresholds[1]
    if count < thresholds[2]:
        return thresholds[2]
    return count


@dataclass
class AugSummary:
    class_name: str
    label: int
    real_count: int
    target: int
    synth_count: int

    @property
    def total(self) -> int:
        return self.real_count + self.synth_count

    @property
    def synth_ratio(self) -> float:
        return self.synth_count / self.total if self.total else 0.0

    @property
    def real_ratio(self) -> float:
        return self.real_count / self.total if self.total else 0.0


def generate_class_samples(generator, class_id: int, num_images: int, batch_size: int = 128):
    saved_paths = []
    remaining = num_images
    GAN_AUG_DIR.mkdir(parents=True, exist_ok=True)
    while remaining > 0:
        cur = min(batch_size, remaining)
        noise = torch.randn(cur, LATENT_DIM, device=DEVICE)
        labels = torch.full((cur,), class_id, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            imgs = generator(noise, labels).cpu()
        for i in range(cur):
            out_name = f"tiered_cgan_{class_id:02d}_{len(saved_paths)+1:05d}.png"
            out_path = GAN_AUG_DIR / out_name
            vutils.save_image(imgs[i], out_path, normalize=True)
            saved_paths.append(out_path)
        remaining -= cur
    return saved_paths


def balance_with_tiers(train_df: pd.DataFrame, classes_map: dict, thresholds=(500, 1000, 1500), gan_batch_size=128):
    generator = load_trained_generator(num_classes=len(classes_map))
    class_counts = train_df["label"].value_counts().to_dict()
    new_rows = []
    summaries: list[AugSummary] = []
    for cls_name, cls_id in classes_map.items():
        real = class_counts.get(cls_id, 0)
        target = tier_target(real, thresholds=thresholds)
        need = target - real
        synth_paths = []
        if need > 0:
            print(f"Classe {cls_name} (id={cls_id}): gerar {need} para equilibrar {real}->{target}")
            synth_paths = generate_class_samples(generator, cls_id, need, batch_size=gan_batch_size)
            for p in synth_paths:
                new_rows.append({"path": p, "class_name": cls_name, "label": cls_id})
        summaries.append(AugSummary(cls_name, cls_id, real_count=real, target=target, synth_count=len(synth_paths)))
    synth_df = pd.DataFrame(new_rows)
    balanced_df = pd.concat([train_df, synth_df], ignore_index=True) if len(synth_df) else train_df
    return balanced_df, summaries


def build_train_df(base_df: pd.DataFrame, base_key: str, classes_map: dict, classes_map_B: dict):
    base_key = base_key.upper()
    if base_key == "A":  # binario
        df = base_df.assign(label=base_df["class_name"].apply(lambda c: 0 if c == "No-Anomaly" else 1))
    elif base_key == "B":  # 11 classes anomalias
        df = base_df[base_df["class_name"] != "No-Anomaly"].assign(label=lambda d: d["class_name"].map(classes_map_B))
    elif base_key == "C":  # 12 classes
        df = base_df.assign(label=lambda d: d["class_name"].map(classes_map))
    else:
        raise ValueError(f"Base invalida: {base_key}.")
    return df.reset_index(drop=True)


def plot_training_curves(history_df: pd.DataFrame, out_path: Path, model_name: str):
    """Salva curvas de perda/metricas para anexar no relatório."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train loss", marker="o")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val loss", marker="o")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val acc", marker="o")
    axes[1].plot(history_df["epoch"], history_df["val_f1_macro"], label="Val f1_macro", marker="o")
    axes[1].set_title(f"{model_name} - Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train_model(model_name: str, base_df: pd.DataFrame, num_classes: int, epochs: int = 3, lr: float = 1e-3, use_weighted_sampler: bool = True, batch_size: int = 128):
    train_split, val_split = train_test_split(
        base_df,
        test_size=0.2,
        stratify=base_df["label"],
        random_state=SEED,
    )
    sampler = make_weighted_sampler(train_split) if use_weighted_sampler else None
    train_loader = make_loader(
        train_split,
        train_transform,
        batch_size=batch_size,
        shuffle=not use_weighted_sampler,
        sampler=sampler,
    )
    val_loader = make_loader(val_split, test_transform, batch_size=batch_size)

    model = NetworkCNN(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        train_loss = running_loss / seen if seen else 0.0
        val_metrics = evaluate_loader_metrics(model, val_loader, criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )
        print(
            f"{model_name} | Epoch {epoch + 1}/{epochs} | "
            f"loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.3f} | val_f1_macro={val_metrics['f1_macro']:.3f}"
        )

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"{model_name}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Modelo guardado em {out_path}")
    return model, pd.DataFrame(history)


def save_summary(summaries: list[AugSummary], path: Path):
    METRICS_DIR.mkdir(exist_ok=True)
    rows = []
    for s in summaries:
        rows.append(
            {
                "class_name": s.class_name,
                "label": s.label,
                "real_count": s.real_count,
                "target": s.target,
                "synth_count": s.synth_count,
                "total": s.total,
                "synth_ratio": s.synth_ratio,
                "real_ratio": s.real_ratio,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Resumo guardado em {path}")
    print(df)
    print(f"Media synth_ratio: {df['synth_ratio'].mean():.3f}")
    return df


def evaluate_on_final_test(model: nn.Module, model_name: str, base_key: str, classes_map: dict, classes_map_B: dict, batch_size: int = 128) -> tuple[dict, Path]:
    """Avaliacao final para anexar métricas (accuracy/f1/loss) no conjunto de teste."""
    test_df = load_final_test_df()
    eval_df = build_train_df(test_df, base_key, classes_map, classes_map_B)
    test_loader = make_loader(eval_df, test_transform, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_loader_metrics(model, test_loader, criterion)
    metrics_with_name = {"model": model_name, **metrics}
    out_path = METRICS_DIR / f"{model_name}_final_test_metrics.csv"
    METRICS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([metrics_with_name]).to_csv(out_path, index=False)
    print(f"Métricas finais guardadas em {out_path}: {metrics_with_name}")
    return metrics_with_name, out_path


def main():
    parser = argparse.ArgumentParser(description="Balancear com CGAN por patamares e treinar modelos D e E.")
    parser.add_argument("--thresholds", type=str, default="500,1000,1500", help="Patamares alvo separados por virgula.")
    parser.add_argument("--gan-batch-size", type=int, default=128, help="Batch para gerar imagens CGAN.")
    parser.add_argument("--epochs", type=int, default=3, help="Epocas para treinar os modelos D/E.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size para treino/val.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--output-summary", type=Path, default=METRICS_DIR / "cgan_tiered_summary.csv", help="Caminho do CSV de resumo.")
    args = parser.parse_args()

    thresholds = tuple(int(x) for x in args.thresholds.split(","))
    seed_everything()
    train_df, classes_map, idx_to_class, anomaly_classes, classes_map_B = load_dataframes()
    METRICS_DIR.mkdir(exist_ok=True)

    print(f"Balancing classes com thresholds {thresholds} ...")
    balanced_df, summaries = balance_with_tiers(train_df, classes_map, thresholds=thresholds, gan_batch_size=args.gan_batch_size)
    summary_df = save_summary(summaries, args.output_summary)

    # Train Model D (11 classes anomalas) e Model E (12 classes)
    model_configs = [
        ("model_D_tiered", "B", len(anomaly_classes)),
        ("model_E_tiered", "C", len(classes_map)),
    ]
    final_test_rows = []
    for name, base_key, num_classes in model_configs:
        df = build_train_df(balanced_df, base_key, classes_map, classes_map_B)
        model, history_df = train_model(
            model_name=name,
            base_df=df,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            use_weighted_sampler=True,
            batch_size=args.batch_size,
        )
        history_csv = METRICS_DIR / f"{name}_history.csv"
        history_df.to_csv(history_csv, index=False)
        plot_path = METRICS_DIR / f"{name}_loss_curves.png"
        plot_training_curves(history_df, plot_path, model_name=name)
        print(f"Curvas de treino guardadas em {plot_path} e CSV em {history_csv}")

        # Avaliação final para anexar métricas
        metrics, metrics_path = evaluate_on_final_test(model, name, base_key, classes_map, classes_map_B, batch_size=args.batch_size)
        final_test_rows.append(metrics)

    if final_test_rows:
        all_metrics_path = METRICS_DIR / "tiered_final_test_metrics.csv"
        pd.DataFrame(final_test_rows).to_csv(all_metrics_path, index=False)
        print(f"Métricas finais consolidadas em {all_metrics_path}")

    print("Pipeline completo.")


if __name__ == "__main__":
    main()
