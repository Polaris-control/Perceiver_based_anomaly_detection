from __future__ import annotations

import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from perceiver.data.vision.mvtec import MVTecDataModule
from perceiver.model.vision.anomaly_detector.backend import (
    AnomalyDecoderConfig,
    AnomalyEncoderConfig,
)
from perceiver.model.vision.anomaly_detector.lightning import LitAnomalyDetector


# 保存排序可视化结果（可选）
SORTING_VIZ_DIR = "./sorting_viz"
os.makedirs(SORTING_VIZ_DIR, exist_ok=True)


def plot_logit_histogram(logits, true_mask, save_path):
    """绘制正常像素与异常像素的 logits 分布直方图"""
    logits_np = logits.detach().cpu().numpy().flatten()
    mask_np = true_mask.detach().cpu().numpy().flatten()
    normal_logits = logits_np[mask_np == 0]
    anomaly_logits = logits_np[mask_np == 1]
    
    plt.figure(figsize=(8,6))
    plt.hist(normal_logits, bins=100, alpha=0.6, label='Normal', color='blue')
    plt.hist(anomaly_logits, bins=100, alpha=0.6, label='Anomaly', color='red')
    plt.xlabel('Raw logits')
    plt.ylabel('Frequency')
    plt.title('Logits Distribution (Normal vs Anomaly)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(logits, true_mask, save_path):
    """绘制 ROC 曲线并计算 AUC"""
    pred = logits.detach().cpu().numpy().flatten()
    true = true_mask.detach().cpu().numpy().flatten()
    fpr, tpr, _ = roc_curve(true, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Pixel-level ROC Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    return roc_auc


def plot_pr_curve(logits, true_mask, save_path):
    """绘制 Precision-Recall 曲线并计算 AP"""
    pred = logits.detach().cpu().numpy().flatten()
    true = true_mask.detach().cpu().numpy().flatten()
    precision, recall, _ = precision_recall_curve(true, pred)
    ap = average_precision_score(true, pred)
    plt.figure()
    plt.plot(recall, precision, label=f'PR (AP={ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Pixel-level PR Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    return ap


def main():
    dataset_dir = "C:/Users/20763/Desktop/zero-shot/MVtec_ad/data"
    train_categories = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
    ]
    test_categories = ["tile", "toothbrush", "transistor", "wood", "zipper"]

    # 数据模块（测试模式，不使用合成异常）
    dm = MVTecDataModule(
        dataset_dir=dataset_dir,
        train_categories=train_categories,
        test_categories=test_categories,
        image_size=224,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        train_augment=False,
        include_test_good=True,
        use_synthetic_anomaly=False,
    )
    dm.setup()

    # 模型配置（应与训练时一致）
    encoder = AnomalyEncoderConfig(
        image_shape=dm.image_shape,
        num_frequency_bands=64,
        num_cross_attention_heads=1,
        num_self_attention_heads=8,
        num_self_attention_layers_per_block=4,
        num_self_attention_blocks=1,
        num_cross_attention_layers=1,
        dropout=0.1,
    )
    decoder = AnomalyDecoderConfig(
        map_shape=(112, 112),
        num_output_query_channels=256,
        num_output_channels=1,
        num_cross_attention_heads=1,
        score_pool="topk_mean",
        score_topk_ratio=0.01,
        dropout=0.1,
    )

    # 加载模型（使用训练好的 checkpoint）
    model = LitAnomalyDetector.load_from_checkpoint(
        checkpoint_path=r"C:\Users\20763\Desktop\zero-shot\perceiver-io\logs\one-class-anomaly_test\focal_qkv_5_seed42_best\checkpoints\last.ckpt",
        encoder=encoder,
        decoder=decoder,
        num_latents=512,
        num_latent_channels=1024,
        pixel_loss_weight=1.0,
        image_loss_weight=0.0,
        pixel_pos_weight=5.0,
        loss_type="focal",
        focal_gamma=1.5,
        area_loss_weight=1.0,
        use_lora=True,
        lora_rank=8,
        lora_alpha=8.0,
        lora_dropout=0.05,
        lora_target_projs=("q_proj","k_proj","v_proj"),
        lora_lr=5e-5,
        decoder_lr=1e-4,
        strict=False,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 测试数据加载器
    dataloader = dm.test_dataloader()

    # ========== 1. 排序性能收集 ==========
    all_logits = []
    all_masks = []
    all_pred_probs = []
    all_image_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            logits = outputs["anomaly_logits"]          # (B, H, W, 1)
            probs = outputs["anomaly_prob"]             # (B, H, W, 1)
            img_score = outputs["image_score"]          # (B,)

            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())
            all_pred_probs.append(probs.cpu())
            all_image_scores.append(img_score.cpu())
            all_labels.append(batch["label"].cpu())

    # 合并所有样本
    all_logits = torch.cat(all_logits, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)
    all_image_scores = torch.cat(all_image_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # ========== 2. 使用 Lightning Trainer 计算官方 AUROC ==========
    # 注意：由于我们已经手动推理，trainer.test 会重新加载数据，但为了简便，直接使用 Lightning 内置测试
    # 为了避免重复，也可以直接计算指标，但为了与训练时一致，调用 trainer.test
    trainer = Trainer(accelerator="auto", devices=1, logger=TensorBoardLogger("logs", name="final_test"))
    trainer.test(model, datamodule=dm)   # 这会输出 test_pixel_auroc 和 test_image_auroc

    # ========== 3. 额外排序性能可视化（可选） ==========
    # 选取前 5 张异常图像（有 mask 的）进行 logits 分布分析
    anomaly_indices = [i for i in range(len(all_masks)) if all_masks[i].sum() > 0][:5]
    for i, idx in enumerate(anomaly_indices):
        logits_img = all_logits[idx, ..., 0]   # (H, W)
        mask_img = all_masks[idx, ..., 0]      # (H, W)
        plot_logit_histogram(logits_img, mask_img, f"{SORTING_VIZ_DIR}/logits_hist_{i}.png")
        plot_roc_curve(logits_img, mask_img, f"{SORTING_VIZ_DIR}/roc_{i}.png")
        plot_pr_curve(logits_img, mask_img, f"{SORTING_VIZ_DIR}/pr_{i}.png")

    # 输出全局统计
    pixel_auc_from_logits = []
    for logits_img, mask_img in zip(all_logits, all_masks):
        if mask_img.sum() == 0 or (mask_img == 0).sum() == 0:
            continue
        pred = logits_img.flatten().cpu().numpy()
        true = mask_img.flatten().cpu().numpy()
        pixel_auc_from_logits.append(roc_auc_score(true, pred))
    if pixel_auc_from_logits:
        print(f"\nManual Pixel AUROC (average over {len(pixel_auc_from_logits)} images with both classes): {np.mean(pixel_auc_from_logits):.4f}")

    print(f"\nImage-level AUROC (from trainer.test) should be shown above.")


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score   # 注意导入
    main()