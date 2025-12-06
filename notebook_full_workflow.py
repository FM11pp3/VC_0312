"""Conteúdo completo que estava no notebook original (partes pesadas).

Inclui:
- Download/carregamento de dados
-, transforms e modelo CNN
- Avaliação de modelos pré-treinados A/B/C
- Gerador CGAN: balanceamento simples (ate max count) e treino condicional
- Treino de variantes A/B/C/D/E e K-fold

Uso via CLI (exemplos):
- Avaliar modelos pré-treinados no teste:    python notebook_full_workflow.py eval
- Balancear com CGAN até max de classe:       python notebook_full_workflow.py balance_cgan
- Treinar variante (ex.: E, 12 classes):      python notebook_full_workflow.py train_variant --key E --epochs 5
- K-fold numa variante (ex.: B):              python notebook_full_workflow.py kfold --key B --epochs 3 --splits 5
- Treinar CGAN condicional:                   python notebook_full_workflow.py train_cgan --epochs 5 --batch-size 128
"""
from __future__ import annotations

import argparse
import random
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Caminhos e constantes
REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR = Path("/content/InfraredSolarModules") if Path("/content").exists() else REPO_ROOT / "InfraredSolarModules"
DATA_URL = "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip"
BASE_IMAGE_DIR = DATA_DIR / "images"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"
TRAIN_CSV = REPO_ROOT / "full_train_data_list.csv"
TEST_CSV = REPO_ROOT / "final_test_data_list.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMAGE_SIZE = (64, 64)
LATENT_DIM = 128
WEIGHT_URLS = {
    "model_A_final.pth": "https://raw.githubusercontent.com/FM11pp3/VC_0312/main/models/model_A_final.pth",
    "model_B_final.pth": "https://raw.githubusercontent.com/FM11pp3/VC_0312/main/models/model_B_final.pth",
    "model_C_final.pth": "https://raw.githubusercontent.com/FM11pp3/VC_0312/main/models/model_C_final.pth",
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dataset() -> None:
    """Descarrega o dataset se ainda não existir."""
    if BASE_IMAGE_DIR.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.with_suffix(".zip")
    print("A descarregar InfraredSolarModules (pode demorar)...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR.parent)
    print(f"Dataset extraído para {DATA_DIR}")


def load_dataframes(image_dir: Path = BASE_IMAGE_DIR):
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    for df in (train_df, test_df):
        df["filename"] = df["path"].apply(lambda p: Path(p).name)
        df["path"] = df["filename"].apply(lambda n: image_dir / n)
    class_pairs = train_df[["class_name", "label"]].drop_duplicates().sort_values("label")
    classes_map = {row.class_name: int(row.label) for row in class_pairs.itertuples()}
    idx_to_class = {v: k for k, v in classes_map.items()}
    return train_df, test_df, classes_map, idx_to_class


def ensure_weights():
    MODELS_DIR.mkdir(exist_ok=True)
    for fname, url in WEIGHT_URLS.items():
        dest = MODELS_DIR / fname
        if dest.exists():
            print(f"{fname} já existe")
            continue
        print(f"A descarregar {fname}...")
        urllib.request.urlretrieve(url, dest)
    print("Pesos prontos.")


# Transforms base (sem augmentations aleatórias)
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


def evaluate_model_metrics(model: nn.Module, loader: DataLoader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    if not all_labels:
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0, "f1_weighted": 0.0}
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
    }


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    class_counts = df["label"].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = df["label"].map(class_weights).astype(float)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights.values, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ===== CGAN condicional simples =====
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


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(1 * 64 * 64 + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        x = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), dim=1)
        validity = self.model(x)
        return validity


def generate_class_samples(generator, class_id: int, num_images: int, batch_size: int = 64, out_dir: Path | None = None):
    saved_paths = []
    remaining = num_images
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    while remaining > 0:
        cur = min(batch_size, remaining)
        noise = torch.randn(cur, LATENT_DIM, device=DEVICE)
        labels = torch.full((cur,), class_id, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            imgs = generator(noise, labels).cpu()
        for i in range(cur):
            if out_dir:
                out_name = f"cgan_{class_id:02d}_{len(saved_paths)+1:05d}.png"
                out_path = out_dir / out_name
                vutils.save_image(imgs[i], out_path, normalize=True)
                saved_paths.append(out_path)
        remaining -= cur
    return saved_paths


def balance_with_cgan(train_df: pd.DataFrame, classes_map: dict, weights_path: Path, batch_size: int = 64, out_dir: Path | None = None):
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesos do CGAN não encontrados: {weights_path}")
    generator = ConditionalGenerator(LATENT_DIM, len(classes_map)).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    generator.load_state_dict(state)
    generator.eval()

    class_counts = train_df["label"].value_counts().to_dict()
    target_per_class = max(class_counts.values())
    new_rows = []
    for cls_name, cls_id in classes_map.items():
        cur = class_counts.get(cls_id, 0)
        need = target_per_class - cur
        if need <= 0:
            continue
        print(f"Classe {cls_name} (id={cls_id}): gerar {need} para {cur}->{target_per_class}")
        paths = generate_class_samples(generator, cls_id, need, batch_size=batch_size, out_dir=out_dir)
        for p in paths:
            new_rows.append({"path": p, "class_name": cls_name, "label": cls_id})
    if not new_rows:
        print("Dataset já estava balanceado; nada gerado.")
        return train_df
    synth_df = pd.DataFrame(new_rows)
    balanced_df = pd.concat([train_df, synth_df], ignore_index=True)
    print(f"Geradas {len(synth_df)} imagens sintéticas. Total final: {len(balanced_df)}")
    return balanced_df


def build_train_df(base_df: pd.DataFrame, base_key: str, classes_map: dict, classes_map_B: dict):
    base_key = base_key.upper()
    if base_key == "A":
        df = base_df.assign(label=base_df["class_name"].apply(lambda c: 0 if c == "No-Anomaly" else 1))
    elif base_key == "B":
        df = base_df[base_df["class_name"] != "No-Anomaly"].assign(label=lambda d: d["class_name"].map(classes_map_B))
    elif base_key == "C":
        df = base_df.assign(label=lambda d: d["class_name"].map(classes_map))
    else:
        raise ValueError(f"Base inválida: {base_key}.")
    return df.reset_index(drop=True)


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

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        val_acc = evaluate_model(model, val_loader)
        print(f"{model_name} | Epoch {epoch + 1}/{epochs} | loss={running_loss / len(train_loader.dataset):.4f} | val_acc={val_acc:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"{model_name}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Modelo guardado em {out_path}")
    return model


def run_kfold_cv(model_name: str, base_df: pd.DataFrame, num_classes: int, n_splits: int = 5, epochs: int = 3, batch_size: int = 128, lr: float = 1e-3, use_weighted_sampler: bool = True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(base_df, base_df["label"]), start=1):
        print(f"\nFold {fold_idx}/{n_splits}")
        train_split = base_df.iloc[train_idx].reset_index(drop=True)
        val_split = base_df.iloc[val_idx].reset_index(drop=True)
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

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
            val_acc = evaluate_model(model, val_loader)
            print(f"  Epoch {epoch + 1}/{epochs} | loss={running_loss / len(train_loader.dataset):.4f} | val_acc={val_acc:.3f}")

        metrics = evaluate_model_metrics(model, val_loader)
        fold_rows.append({"fold": fold_idx, **metrics})
        print(f"Fold {fold_idx} metrics: acc={metrics['accuracy']:.3f} | f1_macro={metrics['f1_macro']:.3f} | f1_weighted={metrics['f1_weighted']:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    metrics_path = METRICS_DIR / f"kfold_results_{model_name}.csv"
    METRICS_DIR.mkdir(exist_ok=True)
    fold_df.to_csv(metrics_path, index=False)
    print(f"Métricas K-fold guardadas em {metrics_path}")
    return fold_df


def train_model_variant(key: str, base_df: pd.DataFrame, classes_map: dict, classes_map_B: dict, use_cgan_balance: bool = False, cgan_weights: Path | None = None, **kwargs):
    key = key.upper()
    anomaly_classes = sorted([c for c in classes_map if c != "No-Anomaly"])
    if key == "A":
        base_key, num_classes = "A", 2
    elif key == "B":
        base_key, num_classes = "B", len(anomaly_classes)
    elif key == "C":
        base_key, num_classes = "C", len(classes_map)
    elif key == "D":  # 11 classes com CGAN
        base_key, num_classes, use_cgan_balance = "B", len(anomaly_classes), True
    elif key == "E":  # 12 classes com CGAN
        base_key, num_classes, use_cgan_balance = "C", len(classes_map), True
    else:
        raise ValueError("Modelo inválido: use A, B, C, D ou E.")

    if use_cgan_balance:
        if cgan_weights is None:
            cgan_weights = REPO_ROOT / "cgan_generated_outputs" / "cgan_generator_minority_classes.pth"
        base_df = balance_with_cgan(base_df, classes_map, weights_path=cgan_weights, batch_size=kwargs.get("gan_batch_size", 64), out_dir=REPO_ROOT / "InfraredSolarModules" / "cgan_augmented")

    df = build_train_df(base_df, base_key, classes_map, classes_map_B)
    name = f"model_{key}_scratch"
    return train_model(name, df, num_classes=num_classes, **kwargs)


def run_kfold_variant(key: str, base_df: pd.DataFrame, classes_map: dict, classes_map_B: dict, use_cgan_balance: bool = False, cgan_weights: Path | None = None, **kwargs):
    key = key.upper()
    anomaly_classes = sorted([c for c in classes_map if c != "No-Anomaly"])
    if key == "A":
        base_key, num_classes = "A", 2
    elif key == "B":
        base_key, num_classes = "B", len(anomaly_classes)
    elif key == "C":
        base_key, num_classes = "C", len(classes_map)
    elif key == "D":
        base_key, num_classes, use_cgan_balance = "B", len(anomaly_classes), True
    elif key == "E":
        base_key, num_classes, use_cgan_balance = "C", len(classes_map), True
    else:
        raise ValueError("Modelo inválido: use A, B, C, D ou E.")

    if use_cgan_balance:
        if cgan_weights is None:
            cgan_weights = REPO_ROOT / "cgan_generated_outputs" / "cgan_generator_minority_classes.pth"
        base_df = balance_with_cgan(base_df, classes_map, weights_path=cgan_weights, batch_size=kwargs.get("gan_batch_size", 64), out_dir=REPO_ROOT / "InfraredSolarModules" / "cgan_augmented")

    df = build_train_df(base_df, base_key, classes_map, classes_map_B)
    return run_kfold_cv(f"{key}", df, num_classes=num_classes, **kwargs)


def evaluate_pretrained(test_df: pd.DataFrame, classes_map: dict):
    ensure_weights()
    anomaly_classes = sorted([c for c in classes_map if c != "No-Anomaly"])
    classes_map_B = {cls: idx for idx, cls in enumerate(anomaly_classes)}
    model_frames = {
        "A": {
            "num_classes": 2,
            "df": test_df.assign(label=test_df["class_name"].apply(lambda c: 0 if c == "No-Anomaly" else 1)),
            "weights": MODELS_DIR / "model_A_final.pth",
        },
        "B": {
            "num_classes": len(anomaly_classes),
            "df": test_df[test_df["class_name"] != "No-Anomaly"].assign(label=lambda d: d["class_name"].map(classes_map_B)),
            "weights": MODELS_DIR / "model_B_final.pth",
        },
        "C": {
            "num_classes": len(classes_map),
            "df": test_df.assign(label=lambda d: d["class_name"].map(classes_map)),
            "weights": MODELS_DIR / "model_C_final.pth",
        },
    }

    results = []
    for key, cfg in model_frames.items():
        loader = make_loader(cfg["df"], test_transform, batch_size=256)
        model = NetworkCNN(cfg["num_classes"]).to(DEVICE)
        state = torch.load(cfg["weights"], map_location=DEVICE)
        model.load_state_dict(state)
        metrics = evaluate_model_metrics(model, loader)
        results.append({"Model": f"Model {key}", **metrics})
        print(f"Model {key}: acc={metrics['accuracy']:.3f} | f1_macro={metrics['f1_macro']:.3f} | f1_weighted={metrics['f1_weighted']:.3f}")

    results_df = pd.DataFrame(results)
    METRICS_DIR.mkdir(exist_ok=True)
    out_path = METRICS_DIR / "final_test_metrics.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Métricas guardadas em {out_path}")
    return results_df


def train_cgan(train_df: pd.DataFrame, classes_map: dict, num_epochs: int = 5, batch_size: int = 64, lr: float = 2e-4, sample_every: int = 1):
    num_classes = len(classes_map)
    loader = DataLoader(
        SolarDataset(train_df, train_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    generator = ConditionalGenerator(LATENT_DIM, num_classes).to(DEVICE)
    discriminator = ConditionalDiscriminator(num_classes).to(DEVICE)
    adversarial_loss = nn.BCELoss()

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(num_classes, LATENT_DIM, device=DEVICE)
    fixed_labels = torch.arange(num_classes, device=DEVICE)
    GAN_OUT_DIR = REPO_ROOT / "cgan_generated_outputs"
    GAN_OUT_DIR.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            valid = torch.ones(imgs.size(0), 1, device=DEVICE)
            fake = torch.zeros(imgs.size(0), 1, device=DEVICE)

            # Train generator
            opt_G.zero_grad()
            z = torch.randn(imgs.size(0), LATENT_DIM, device=DEVICE)
            gen_labels = labels
            gen_imgs = generator(z, gen_labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            opt_G.step()

            # Train discriminator
            opt_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

        print(f"Epoch {epoch+1}/{num_epochs} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

        if (epoch + 1) % sample_every == 0:
            with torch.no_grad():
                samples = generator(fixed_noise, fixed_labels).cpu()
                grid = vutils.make_grid(samples, nrow=4, normalize=True)
                out_path = GAN_OUT_DIR / f"cgan_samples_epoch_{epoch+1:03}.png"
                vutils.save_image(grid, out_path)
                print(f"Samples guardados em {out_path}")

    torch.save(generator.state_dict(), MODELS_DIR / "cgan_generator.pth")
    print("Gerador guardado em models/cgan_generator.pth")
    return generator


def parse_args():
    p = argparse.ArgumentParser(description="Script consolidado do notebook original.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("eval", help="Avalia modelos pré-treinados A/B/C no teste.")

    b = sub.add_parser("balance_cgan", help="Balanceia com CGAN até o máximo de exemplos por classe.")
    b.add_argument("--weights", type=Path, default=REPO_ROOT / "cgan_generated_outputs" / "cgan_generator_minority_classes.pth", help="Pesos do gerador CGAN.")
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--out-dir", type=Path, default=REPO_ROOT / "InfraredSolarModules" / "cgan_augmented")

    t = sub.add_parser("train_variant", help="Treina variante A/B/C/D/E.")
    t.add_argument("--key", type=str, required=True, help="A, B, C, D ou E")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--gan-batch-size", type=int, default=64)
    t.add_argument("--cgan-weights", type=Path, default=REPO_ROOT / "cgan_generated_outputs" / "cgan_generator_minority_classes.pth")

    kf = sub.add_parser("kfold", help="K-fold numa variante A/B/C/D/E.")
    kf.add_argument("--key", type=str, required=True)
    kf.add_argument("--epochs", type=int, default=3)
    kf.add_argument("--batch-size", type=int, default=128)
    kf.add_argument("--splits", type=int, default=5)
    kf.add_argument("--lr", type=float, default=1e-3)
    kf.add_argument("--gan-batch-size", type=int, default=64)
    kf.add_argument("--cgan-weights", type=Path, default=REPO_ROOT / "cgan_generated_outputs" / "cgan_generator_minority_classes.pth")

    cg = sub.add_parser("train_cgan", help="Treina CGAN condicional simples.")
    cg.add_argument("--epochs", type=int, default=5)
    cg.add_argument("--batch-size", type=int, default=64)
    cg.add_argument("--lr", type=float, default=2e-4)
    cg.add_argument("--sample-every", type=int, default=1)

    return p.parse_args()


def main():
    seed_everything()
    ensure_dataset()
    train_df, test_df, classes_map, idx_to_class = load_dataframes()
    anomaly_classes = sorted([c for c in classes_map if c != "No-Anomaly"])
    classes_map_B = {cls: idx for idx, cls in enumerate(anomaly_classes)}

    args = parse_args()

    if args.cmd == "eval":
        evaluate_pretrained(test_df, classes_map)

    elif args.cmd == "balance_cgan":
        balanced = balance_with_cgan(
            train_df,
            classes_map,
            weights_path=args.weights,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
        )
        out_path = REPO_ROOT / "balanced_with_cgan.csv"
        balanced.to_csv(out_path, index=False)
        print(f"Balanced train salvo em {out_path}")

    elif args.cmd == "train_variant":
        train_model_variant(
            key=args.key,
            base_df=train_df,
            classes_map=classes_map,
            classes_map_B=classes_map_B,
            use_cgan_balance=False,  # D/E forçam True dentro da função
            cgan_weights=args.cgan_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gan_batch_size=args.gan_batch_size,
        )

    elif args.cmd == "kfold":
        run_kfold_variant(
            key=args.key,
            base_df=train_df,
            classes_map=classes_map,
            classes_map_B=classes_map_B,
            use_cgan_balance=False,  # D/E forçam True dentro da função
            cgan_weights=args.cgan_weights,
            n_splits=args.splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gan_batch_size=args.gan_batch_size,
        )

    elif args.cmd == "train_cgan":
        train_cgan(
            train_df=train_df,
            classes_map=classes_map,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sample_every=args.sample_every,
        )


if __name__ == "__main__":
    main()
