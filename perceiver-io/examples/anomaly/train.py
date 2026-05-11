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
        image_size=256,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        train_augment=True,
        include_test_good=True,
    )
    dm.setup()

    encoder = AnomalyEncoderConfig(
        image_shape=dm.image_shape,  # (256,256,3)
        num_frequency_bands=64,
        num_cross_attention_heads=1,
        num_self_attention_heads=8,
        num_self_attention_layers_per_block=2,
        num_self_attention_blocks=1,
        num_cross_attention_layers=1,
        dropout=0.1,
        params="deepmind/vision-perceiver-fourier",
    )
    decoder = AnomalyDecoderConfig(
        map_shape=(128, 128),
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
        num_latents=256,
        num_latent_channels=512,
        pixel_loss_weight=1.0,
        image_loss_weight=0.1,
        pixel_pos_weight=20.0,
        loss_type="focal",
        focal_gamma=1.5,
    )

    # Lightning requires configure_optimizers OR pass optimizer via CLI.
    # Here we define it by monkey-patching for minimal reproducibility:
    def configure_optimizers():
        return AdamW(model.parameters(), lr=5e-5)

    model.configure_optimizers = configure_optimizers  # minimal, explicit

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=20,
        logger=TensorBoardLogger(save_dir="logs", name="anomaly"),
        log_every_n_steps=1,
        limit_train_batches=0.5,
        limit_val_batches=1.0,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()