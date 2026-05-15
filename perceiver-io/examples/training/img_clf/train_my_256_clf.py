import examples.training
import pytorch_lightning as pl

from perceiver.model.core import ClassificationDecoderConfig
from perceiver.model.vision.image_classifier import ImageClassifierConfig, ImageEncoderConfig, LitImageClassifier
from perceiver.scripts.lrs import ConstantWithWarmupLR
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW

from perceiver.data.vision.mvtec import MVTecDataModule



# 优化器配置

def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=1e-4)
    scheduler = ConstantWithWarmupLR(optimizer, warmup_steps=500)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }

setattr(LitImageClassifier, "configure_optimizers", configure_optimizers)


#MVTecDataModule 的官方实现里 只加载train_dataloader 10 类训练集的 good 样本

dm = MVTecDataModule(
    dataset_dir="C:/Users/20763/Desktop/zero-shot/MVtec_ad/data",
    image_size=224,                # 与异常检测模型一致
    batch_size=8,
    num_workers=0,
    pin_memory=True,
    train_augment=True,

    # 10 类训练集（只使用 train/good 样本）
    train_categories=[
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
    ],
    test_categories=[],  # 不使用测试集
)



# 模型参数 匹配的异常检测

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



# 训练

if __name__ == "__main__":
    lit_model = LitImageClassifier.create(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=20,
        logger=TensorBoardLogger("logs", name="mvtec_224_10class_pretrain"),
    )

    trainer.fit(lit_model, datamodule=dm)