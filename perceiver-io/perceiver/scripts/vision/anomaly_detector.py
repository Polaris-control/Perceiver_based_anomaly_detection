from pytorch_lightning.cli import LightningArgumentParser

# Import data module class so Lightning CLI can resolve --data=MVTecDataModule
from perceiver.data.vision.mvtec import MVTecDataModule  # noqa: F401
from perceiver.model.vision.anomaly_detector import LitAnomalyDetector
from perceiver.scripts.cli import CLI
from perceiver.scripts.lrs import ConstantWithWarmupLR


class AnomalyDetectorCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        parser.add_lr_scheduler_args(ConstantWithWarmupLR)

        # Required shape link (same pattern as image_classifier.py)
        parser.link_arguments("data.image_shape", "model.encoder.image_shape", apply_on="instantiate")

        # Optional but recommended: keep decoder map_shape and image_size consistent by defaults
        parser.set_defaults(
            {
                "model.num_latents": 512,
                "model.num_latent_channels": 1024,
                "model.encoder.num_frequency_bands": 64,
                "model.encoder.num_cross_attention_layers": 1,
                "model.encoder.num_cross_attention_heads": 1,
                "model.encoder.num_self_attention_heads": 8,
                "model.encoder.num_self_attention_layers_per_block": 4,
                "model.encoder.num_self_attention_blocks": 1,
                "model.encoder.dropout": 0.1,
                "model.decoder.num_output_query_channels": 256,
                "model.decoder.num_cross_attention_heads": 1,
                "model.decoder.dropout": 0.1,
                "model.decoder.map_shape": [112, 112],
                "model.decoder.num_output_channels": 1, #输出异常图
                "model.decoder.score_pool": "topk_mean",
                "model.decoder.score_topk_ratio": 0.01,
                "model.pixel_loss_weight": 1.0, 
                "model.image_loss_weight": 0.1,
                "model.pixel_pos_weight": 2.0,  # 像素正类权重（mask 稀疏）
                "model.loss_type": "bce",
                "model.focal_gamma": 1.0,
                "model.use_lora": False,
                "model.lora_rank": 8,
                "model.lora_alpha": 16.0,
                "model.lora_dropout": 0.0,
                "model.lora_target_projs": ["q_proj", "v_proj"],
                "model.lora_lr": None,
            }
        )


if __name__ == "__main__":
    AnomalyDetectorCLI(LitAnomalyDetector, run=True)