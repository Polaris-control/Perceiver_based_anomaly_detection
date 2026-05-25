from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.vision.image_classifier import ImageClassifierConfig, ImageEncoderConfig, LitImageClassifier
from perceiver.data.vision.mvtec_category import MVTecCategoryDataModule
from perceiver.scripts.lrs import ConstantWithWarmupLR


class CustomLitImageClassifier(LitImageClassifier):
    """重写优化器配置方法"""
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
        # 使用预热调度器（如果需要）
        scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=500)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def main():
    CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw"
    ]

    dm = MVTecCategoryDataModule(
        dataset_dir="C:/Users/20763/Desktop/zero-shot/MVtec_ad/data",
        categories=CATEGORIES,
        image_size=224,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        train_augment=True,
        val_split=0.1,
    )
    dm.setup()

    # 模型配置（必须与异常检测模型编码器一致）
    config = ImageClassifierConfig(
        encoder=ImageEncoderConfig(
            image_shape=(224, 224, 3),
            num_frequency_bands=64,
            num_cross_attention_layers=1,
            num_cross_attention_heads=1,
            num_self_attention_blocks=1,
            num_self_attention_layers_per_block=4,
            dropout=0.1,
        ),
        decoder=ClassificationDecoderConfig(
            num_output_query_channels=256,
            num_cross_attention_heads=1,
            num_classes=10,
            dropout=0.1,
        ),
        num_latents=512,
        num_latent_channels=1024,
    )

    # 使用自定义类
    lit_model = CustomLitImageClassifier.create(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch:02d}-val_acc={val_acc:.4f}",
    )
    early_stop = EarlyStopping(
        monitor="val_acc",
        patience=5,
        mode="max",
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",          # 自动选择 GPU/CPU
        devices=1,
        max_epochs=20,
        logger=TensorBoardLogger("logs", name="mvtec_224_10class_pretrain_fixed"),
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10,
        precision="16-mixed",
    )

    trainer.fit(lit_model, datamodule=dm)


if __name__ == "__main__":
    main()