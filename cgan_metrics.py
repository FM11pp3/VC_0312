"""Quick CGAN evaluation: generates samples and reports FID/diversity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3, Inception_V3_Weights

# Paths and constants
REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR = Path("/content/InfraredSolarModules") if Path("/content").exists() else REPO_ROOT / "InfraredSolarModules"
DATA_URL = "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip"
BASE_IMAGE_DIR = DATA_DIR / "images"
TRAIN_CSV = REPO_ROOT / "full_train_data_list.csv"
CGAN_WEIGHTS = REPO_ROOT / "cgan_generated_outputs" / "cgan_generator_minority_classes.pth"
METRICS_DIR = REPO_ROOT / "metrics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
LATENT_DIM = 128


def seed_everything(seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # sync Python's default RNG too
    _ = rng.random()


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


def load_train_df():
    ensure_dataset()
    df = pd.read_csv(TRAIN_CSV)
    df["filename"] = df["path"].apply(lambda p: Path(p).name)
    df["path"] = df["filename"].apply(lambda n: BASE_IMAGE_DIR / n)
    class_pairs = df[["class_name", "label"]].drop_duplicates().sort_values("label")
    classes_map = {row.class_name: int(row.label) for row in class_pairs.itertuples()}
    idx_to_class = {v: k for k, v in classes_map.items()}
    return df, classes_map, idx_to_class


class SolarDataset(Dataset):
    """Loads real images from the CSV with a given transform."""

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


class GeneratedDataset(Dataset):
    """Wraps generated tensors so we can reuse torchvision transforms."""

    def __init__(self, images: list[torch.Tensor], labels: list[int], transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


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


def select_minority_classes(df: pd.DataFrame, top_k: int) -> list[str]:
    counts = df["class_name"].value_counts().sort_values()
    return counts.head(top_k).index.tolist()


def generate_samples(generator, class_id: int, num_images: int, batch_size: int = 128):
    """Return a list of tensors in [0,1]."""
    remaining = num_images
    out = []
    while remaining > 0:
        cur = min(batch_size, remaining)
        noise = torch.randn(cur, LATENT_DIM, device=DEVICE)
        labels = torch.full((cur,), class_id, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            imgs = generator(noise, labels).cpu()
        imgs = (imgs + 1) / 2.0  # map from [-1,1] to [0,1]
        out.extend(list(imgs))
        remaining -= cur
    return out


def get_inception_model():
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    return model.to(DEVICE)


FID_REAL_TRANSFORM = T.Compose(
    [
        T.Resize((299, 299)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ]
)

FID_GEN_TRANSFORM = T.Compose(
    [
        T.Resize((299, 299)),
        T.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),
    ]
)


def compute_activations(loader: DataLoader, feature_model: nn.Module) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            acts = feature_model(imgs)
            acts = torch.flatten(acts, 1)
            feats.append(acts.cpu())
    if not feats:
        return np.zeros((0, 2048), dtype=np.float64)
    return torch.cat(feats, dim=0).numpy().astype(np.float64)


def calc_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """FID without relying on scipy; uses eigen-decomposition."""
    diff = mu1 - mu2
    cov_prod = sigma1 @ sigma2
    # stabilise in case of numerical noise
    cov_prod += np.eye(cov_prod.shape[0]) * eps
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals = np.clip(eigvals, a_min=0, a_max=None)
    covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.real(fid))


def stats_for_loader(loader: DataLoader, feature_model: nn.Module):
    acts = compute_activations(loader, feature_model)
    if len(acts) == 0:
        return acts, np.zeros(1), np.zeros((1, 1))
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return acts, mu, sigma


def mean_pairwise_distance(features: np.ndarray) -> float:
    """Simple diversity proxy: mean L2 between feature pairs."""
    if features.shape[0] < 2:
        return 0.0
    feats = torch.tensor(features, dtype=torch.float32)
    return torch.pdist(feats, p=2).mean().item()


def build_loader(df: pd.DataFrame, transform, batch_size: int):
    ds = SolarDataset(df, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())


def build_generated_loader(images: list[torch.Tensor], labels: list[int], transform, batch_size: int):
    ds = GeneratedDataset(images, labels, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())


def evaluate_cgan(
    target_classes: list[str],
    samples_per_class: int,
    batch_size: int,
    gan_batch_size: int,
):
    train_df, classes_map, idx_to_class = load_train_df()
    generator = load_trained_generator(num_classes=len(classes_map))
    feature_model = get_inception_model()

    # Build real subset
    real_df = train_df[train_df["class_name"].isin(target_classes)].reset_index(drop=True)
    real_loader = build_loader(real_df, FID_REAL_TRANSFORM, batch_size=batch_size)

    # Generate synthetic tensors for target classes
    gen_images, gen_labels = [], []
    for cls_name in target_classes:
        cls_id = classes_map[cls_name]
        imgs = generate_samples(generator, cls_id, samples_per_class, batch_size=gan_batch_size)
        gen_images.extend(imgs)
        gen_labels.extend([cls_id] * len(imgs))
    gen_loader = build_generated_loader(gen_images, gen_labels, FID_GEN_TRANSFORM, batch_size=batch_size)

    # Overall FID
    gen_acts, gen_mu, gen_sigma = stats_for_loader(gen_loader, feature_model)
    real_acts, real_mu, real_sigma = stats_for_loader(real_loader, feature_model)
    overall_fid = calc_fid(real_mu, real_sigma, gen_mu, gen_sigma)

    # Per-class FID + diversity
    fid_per_class = {}
    diversity_per_class = {}
    for cls_name in target_classes:
        cls_id = classes_map[cls_name]
        real_cls_df = real_df[real_df["class_name"] == cls_name]
        real_cls_loader = build_loader(real_cls_df, FID_REAL_TRANSFORM, batch_size=batch_size)
        fake_cls_imgs = [img for img, lbl in zip(gen_images, gen_labels) if lbl == cls_id]
        fake_cls_lbls = [cls_id] * len(fake_cls_imgs)
        fake_cls_loader = build_generated_loader(fake_cls_imgs, fake_cls_lbls, FID_GEN_TRANSFORM, batch_size=batch_size)

        _, r_mu, r_sigma = stats_for_loader(real_cls_loader, feature_model)
        cls_acts, f_mu, f_sigma = stats_for_loader(fake_cls_loader, feature_model)
        fid_per_class[cls_name] = calc_fid(r_mu, r_sigma, f_mu, f_sigma)
        diversity_per_class[cls_name] = mean_pairwise_distance(cls_acts)

    return {
        "device": str(DEVICE),
        "target_classes": target_classes,
        "samples_per_class": samples_per_class,
        "overall_fid": overall_fid,
        "fid_per_class": fid_per_class,
        "diversity_l2_per_class": diversity_per_class,
        "real_counts": real_df["class_name"].value_counts().to_dict(),
        "generated_counts": {idx_to_class[lbl]: count for lbl, count in pd.Series(gen_labels).value_counts().items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CGAN quality (FID + diversity).")
    parser.add_argument("--samples-per-class", type=int, default=256, help="Number of synthetic images per class to sample.")
    parser.add_argument("--top-k-classes", type=int, default=4, help="Evaluate the K most minority classes (ignored if --classes is set).")
    parser.add_argument("--classes", nargs="+", help="Explicit class names to evaluate (case-sensitive).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction.")
    parser.add_argument("--gan-batch-size", type=int, default=128, help="Batch size for GAN sampling.")
    parser.add_argument("--output", type=Path, default=METRICS_DIR / "cgan_fid_metrics.json", help="Where to store the metrics JSON.")
    args = parser.parse_args()

    seed_everything()
    train_df, _, _ = load_train_df()
    target_classes = args.classes or select_minority_classes(train_df, top_k=args.top_k_classes)

    print(f"Running CGAN eval on {len(target_classes)} classes: {target_classes}")
    metrics = evaluate_cgan(
        target_classes=target_classes,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        gan_batch_size=args.gan_batch_size,
    )

    METRICS_DIR.mkdir(exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"Done. Metrics saved to {args.output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
