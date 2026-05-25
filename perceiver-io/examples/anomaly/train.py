from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from perceiver.data.vision.mvtec import MVTecDataModule
from perceiver.model.vision.anomaly_detector.backend import (
    AnomalyDecoderConfig,
    AnomalyEncoderConfig,
)
from perceiver.model.vision.anomaly_detector.lightning import LitAnomalyDetector


def main():
    seed_everything(42,workers=True)

    dataset_dir = "C:/Users/20763/Desktop/zero-shot/MVtec_ad/data"

    single_category = None

    train_categories = [
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
    test_categories = ["tile", "toothbrush", "transistor", "wood", "zipper"]

    if single_category is not None:
        train_categories = [single_category]
        test_categories = [single_category]

    dm = MVTecDataModule(
        dataset_dir=dataset_dir,
        train_categories=train_categories,
        test_categories=test_categories,
        image_size=224,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        train_augment=True,
        include_test_good=True,
        use_synthetic_anomaly=True,
        synthetic_anomaly_prob=0.5,
        synthetic_min_size_ratio=0.03,
        synthetic_max_size_ratio=0.15,
        synthetic_max_patches=3,
        synthetic_noise_std=0.25,
    )
    dm.setup()

    encoder = AnomalyEncoderConfig(
        image_shape=dm.image_shape,
        num_frequency_bands=64,
        num_cross_attention_heads=1,
        num_self_attention_heads=8,
        num_self_attention_layers_per_block=4,
        num_self_attention_blocks=1,
        num_cross_attention_layers=1,
        dropout=0.1,
        params=r"C:\Users\20763\Desktop\zero-shot\perceiver-io\logs\mvtec_224_10class_pretrain_fixed\version_0\checkpoints\epoch=05-val_acc=1.0000.ckpt",
    )

    decoder = AnomalyDecoderConfig(
        map_shape=(112,112),
        num_output_query_channels=256,
        num_output_channels=1,
        num_cross_attention_heads=1,
        score_pool="topk_mean",
        score_topk_ratio=0.01,
        dropout=0.1,
        use_global_bias=False,
        trainable_query=False,
    )

    model = LitAnomalyDetector(
        encoder=encoder,
        decoder=decoder,
        num_latents=512,
        num_latent_channels=1024,

        pixel_loss_weight=1.0,
        image_loss_weight=0.0,
        pixel_pos_weight=20.0,
        loss_type="focal",
        focal_gamma=2.0,
        focal_alpha=0.75,

        area_loss_weight=1.0,
        dice_loss_weight=0.05,
        dice_smooth=1.0,
        edge_loss_weight=0.0,
        ranking_loss_weight=0.5, 
        hard_neg_ratio=0.05,       


        encoder_lr=None,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        lora_target_projs=("q_proj","k_proj","v_proj"),
        lora_lr=5e-5,
        decoder_lr=1e-4,
    )

    # 自动保存最优模型，根据 val_pixel_auroc 排序
    checkpoint_callback = ModelCheckpoint(
        monitor="val_pixel_auroc",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch:02d}-val_pixel_auroc={val_pixel_auroc:.4f}",
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=20,
        logger=TensorBoardLogger(save_dir="logs", name="anomaly_new", version="focal_use_lora_final"),
        callbacks=[checkpoint_callback], #加入回调
        log_every_n_steps=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()