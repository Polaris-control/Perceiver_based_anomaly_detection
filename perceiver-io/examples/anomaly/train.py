from __future__ import annotations

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW

from perceiver.data.vision.mvtec import MVTecDataModule
from perceiver.model.vision.anomaly_detector.backend import AnomalyDecoderConfig, AnomalyEncoderConfig
from perceiver.model.vision.anomaly_detector.lightning import LitAnomalyDetector


def main():
    dataset_dir = "C:/Users/20763/Desktop/zero-shot/MVtec_ad/data"
    single_category = "bottle"  # e.g. "bottle" for Line-2 one-class validation

    # Example: 10 seen (train), 5 unseen (val/test) — adjust freely
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
        num_workers=0,
        pin_memory=True,
        train_augment=True,
        include_test_good=True,
    )
    dm.setup()

    encoder = AnomalyEncoderConfig(
        image_shape=dm.image_shape,  # (224,224,3)
        num_frequency_bands=64,
        num_cross_attention_heads=1,
        num_self_attention_heads=8,
        num_self_attention_layers_per_block=4,
        num_self_attention_blocks=1,
        num_cross_attention_layers=1,
        dropout=0.1,
        params="C:/Users/20763/Desktop/zero-shot/perceiver-io/logs/mvtec_224_10class_pretrain/version_0/checkpoints/epoch=19-step=6600.ckpt",
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

    model = LitAnomalyDetector(
        encoder=encoder,
        decoder=decoder,
        num_latents=512,
        num_latent_channels=1024,
        pixel_loss_weight=1.0,
        image_loss_weight=0.1,
        pixel_pos_weight=20.0,
        loss_type="focal",
        focal_gamma=1.5,
        encoder_lr=None,    # Encoder 不训练
        decoder_lr=1e-4,    # 只训练 Decoder
    )
    
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=10,
        logger=TensorBoardLogger(save_dir="logs", name="one-class-anomaly"),
        log_every_n_steps=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()