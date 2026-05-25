from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from perceiver.data.vision.mvtec import MVTecDataModule
from perceiver.model.vision.image_classifier import LitImageClassifier


CKPT_PATH = r"C:\Users\20763\Desktop\zero-shot\perceiver-io\logs\mvtec_224_10class_pretrain_fixed\version_0\checkpoints\epoch=05-val_acc=1.0000.ckpt"
DATASET_DIR = r"C:\Users\20763\Desktop\zero-shot\MVtec_ad\data"

SAVE_DIR = "pretrain_classifier_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def denormalize_image(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 3 and x.shape[-1] == 3:
        x = x.permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()
    return x


def collect_predictions(model, dataloader, device):
    y_true = []
    y_pred = []
    probs_all = []
    paths = []
    categories = []
    images_for_grid = []

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images)

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            batch_categories = batch["category"]
            batch_paths = batch["path"]

            for i, cat in enumerate(batch_categories):
                if cat not in CLASS_TO_ID:
                    continue

                y_true.append(CLASS_TO_ID[cat])
                y_pred.append(int(preds[i].cpu()))
                probs_all.append(probs[i].cpu().numpy())
                paths.append(batch_paths[i])
                categories.append(cat)

                if len(images_for_grid) < 40:
                    images_for_grid.append(
                        (
                            batch["image"][i].cpu(),
                            cat,
                            CLASS_NAMES[int(preds[i].cpu())],
                            float(probs[i].max().cpu()),
                        )
                    )

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(probs_all),
        paths,
        categories,
        images_for_grid,
    )


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True category")
    plt.title("Pretrained MVTec 10-class Classifier Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_10class.png"), dpi=200)
    plt.close()


def plot_prediction_distribution(y_pred):
    counts = np.bincount(y_pred, minlength=len(CLASS_NAMES))

    plt.figure(figsize=(12, 5))
    plt.bar(CLASS_NAMES, counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of predictions")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "prediction_distribution.png"), dpi=200)
    plt.close()


def plot_probability_heatmap(probs_all, y_true):
    class_mean_probs = []

    for cls_id in range(len(CLASS_NAMES)):
        mask = y_true == cls_id
        if mask.sum() == 0:
            class_mean_probs.append(np.zeros(len(CLASS_NAMES)))
        else:
            class_mean_probs.append(probs_all[mask].mean(axis=0))

    class_mean_probs = np.stack(class_mean_probs, axis=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        class_mean_probs,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted probability class")
    plt.ylabel("True category")
    plt.title("Mean Softmax Probability per True Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "class_probability_heatmap.png"), dpi=200)
    plt.close()


def plot_sample_predictions(images_for_grid):
    n = min(len(images_for_grid), 40)
    cols = 5
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(n):
        img, true_cat, pred_cat, conf = images_for_grid[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(denormalize_image(img))
        color = "green" if true_cat == pred_cat else "red"
        plt.title(f"T:{true_cat}\nP:{pred_cat} {conf:.2f}", color=color, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "sample_predictions.png"), dpi=200)
    plt.close()


def collect_encoder_features(model, dataloader, device, max_samples):
    features = []
    labels = []

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            cats = batch["category"]

            latents = model.model.encoder(images)
            feat = latents.mean(dim=1)

            for i, cat in enumerate(cats):
                if cat not in CLASS_TO_ID:
                    continue

                features.append(feat[i].cpu().numpy())
                labels.append(CLASS_TO_ID[cat])

                if len(features) >= max_samples:
                    return np.array(features), np.array(labels)

    return np.array(features), np.array(labels)


def plot_tsne(features, labels):
    if len(features) < 5:
        return

    if features.shape[1] > 50:
        features_50 = PCA(n_components=50, random_state=42).fit_transform(features)
    else:
        features_50 = features

    perplexity = min(30, max(2, len(features_50) // 10))
    emb = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(features_50)

    plt.figure(figsize=(10, 8))

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_id
        if mask.sum() == 0:
            continue
        plt.scatter(emb[mask, 0], emb[mask, 1], s=12, label=cls_name, alpha=0.75)

    plt.legend(markerscale=2, fontsize=8)
    plt.title("t-SNE of Encoder Features")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "tsne_encoder_features.png"), dpi=200)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = MVTecDataModule(
        dataset_dir=DATASET_DIR,
        train_categories=[],
        test_categories=CLASS_NAMES,
        image_size=224,
        batch_size=8,
        num_workers=0,
        pin_memory=True,
        train_augment=False,
        include_test_good=True,
        use_synthetic_anomaly=False,
    )
    dm.setup()

    model = LitImageClassifier.load_from_checkpoint(CKPT_PATH, params=None)
    model = model.to(device)
    model.eval()

    loader = dm.test_dataloader()

    y_true, y_pred, probs_all, paths, categories, images_for_grid = collect_predictions(
        model, loader, device
    )

    print("Number of evaluated samples:", len(y_true))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

     # 获取测试集总样本数
    total_samples = len(dm.test_dataloader().dataset)   # 或 len(dm.ds_test)
    print(f"Total test samples: {total_samples}")

    # 将 max_samples 设为总样本数（或更大）
    features, labels = collect_encoder_features(model, loader, device, max_samples=total_samples)

    plot_tsne(features, labels)

    plot_confusion_matrix(y_true, y_pred)
    plot_prediction_distribution(y_pred)
    plot_probability_heatmap(probs_all, y_true)
    plot_sample_predictions(images_for_grid)

   


    print(f"\nVisualization results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()