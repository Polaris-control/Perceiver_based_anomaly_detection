"""
基线训练脚本：仅使用预训练编码器 + 从头训练解码器
- 无 LoRA（编码器完全冻结，不注入任何可训练参数）
- 损失：简单 BCE（不使用 focal、area loss、dice loss）
- 目标：评估分类器预训练权重本身对异常检测的提升效果
"""

from __future__ import annotations

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from perceiver.data.vision.mvtec import MVTecDataModule
from perceiver.model.vision.anomaly_detector.backend import (
    AnomalyDecoderConfig,
    AnomalyEncoderConfig,
)
from perceiver.model.vision.anomaly_detector.lightning import LitAnomalyDetector


def main():
    # 固定随机种子，保证可复现
    seed_everything(42, workers=True)

    dataset_dir = "C:/Users/20763/Desktop/zero-shot/MVtec_ad/data"

    # 训练类别（前10类）与测试类别（后5类）
    train_categories = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
    ]
    test_categories = ["tile", "toothbrush", "transistor", "wood", "zipper"]

    # 数据模块配置（使用与之前相同的合成异常，但可保持默认值）
    dm = MVTecDataModule(
        dataset_dir=dataset_dir,
        train_categories=train_categories,
        test_categories=test_categories,
        image_size=224,
        batch_size=8,
        num_workers=0,
        pin_memory=True,
        train_augment=True,                # 使用数据增强
        include_test_good=True,
        use_synthetic_anomaly=True,
        synthetic_anomaly_prob=0.5,
        synthetic_min_size_ratio=0.02,
        synthetic_max_size_ratio=0.12,
        synthetic_max_patches=3,
        synthetic_noise_std=0.25,
    )
    dm.setup()

    # 编码器配置（必须与分类器训练时的配置完全一致）
    encoder = AnomalyEncoderConfig(
        image_shape=dm.image_shape,
        num_frequency_bands=64,            # 与分类器一致
        num_cross_attention_heads=1,
        num_self_attention_heads=8,
        num_self_attention_layers_per_block=4,
        num_self_attention_blocks=1,
        num_cross_attention_layers=1,
        dropout=0.1,
        # 关键：加载分类器预训练权重
        params=r"C:\Users\20763\Desktop\zero-shot\perceiver-io\logs\mvtec_224_10class_pretrain_fixed\version_0\checkpoints\epoch=epoch=05-val_acc=val_acc=1.0000.ckpt",
    )

    # 解码器配置（标准配置，输出112x112热力图）
    decoder = AnomalyDecoderConfig(
        map_shape=(112, 112),
        num_output_query_channels=256,
        num_output_channels=1,
        num_cross_attention_heads=1,
        score_pool="topk_mean",
        score_topk_ratio=0.01,
        dropout=0.1,
    )

    # 模型配置：基线版本
    # - 关闭 LoRA（use_lora=False），编码器完全冻结
    # - 使用普通 BCE 损失（loss_type="bce"）
    # - 关闭面积损失、Dice损失、边缘损失
    # - 仅保留像素级损失，图像级损失关闭
    model = LitAnomalyDetector(
        encoder=encoder,
        decoder=decoder,
        num_latents=512,
        num_latent_channels=1024,
        pixel_loss_weight=1.0,
        image_loss_weight=0.0,
        pixel_pos_weight=5.0,              # 适中正样本权重，平衡正负像素
        loss_type="bce",                   # 普通二值交叉熵
        focal_gamma=1.0,                  # 无效（非focal模式）
        dice_loss_weight=0.0,              # 关闭 Dice
        dice_smooth=1.0,
        encoder_lr=None,
        use_lora=True,                    # 关键：关闭 LoRA
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target_projs=("q_proj", "v_proj"),
        lora_lr=None,
        area_loss_weight=0.0,              # 关闭面积损失
        edge_loss_weight=0.0,              # 关闭边缘损失
    )

    # 模型保存：监控像素级 AUROC
    checkpoint_callback = ModelCheckpoint(
        monitor="val_pixel_auroc",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch:02d}-val_pixel_auroc={val_pixel_auroc:.4f}",
    )

    # 训练器：与之前保持一致
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=10,                     # 训练20个epoch
        logger=TensorBoardLogger(save_dir="logs", name="anomaly_baseline", version="bce_pretrained_use_lora"),
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()