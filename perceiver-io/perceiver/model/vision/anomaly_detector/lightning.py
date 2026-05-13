from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from perceiver.model.core.lightning import LitPerceiverIO, is_checkpoint
from perceiver.model.vision.anomaly_detector.backend import (
    AnomalyDecoderConfig,
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyEncoderConfig,
)
from perceiver.model.vision.anomaly_detector.metrics import (
    compute_and_reset,
    make_auroc_metrics,
    update_image_auroc,
    update_pixel_auroc,
)


class LitAnomalyDetector(LitPerceiverIO):
    def __init__(
        self,
        encoder: AnomalyEncoderConfig,
        decoder: AnomalyDecoderConfig,
        pixel_loss_weight: float = 1.0,
        image_loss_weight: float = 0.1,
        pixel_pos_weight: float = 20.0,  # E1: class imbalance fix
        loss_type: str = "focal",
        focal_gamma: float = 1.5,
        encoder_lr: Optional[float] = None,
        decoder_lr: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.save_hyperparameters()
        super().__init__(encoder, decoder, *args, **kwargs)

        self.model = AnomalyDetector(
            AnomalyDetectorConfig(
                encoder=encoder,
                decoder=decoder,
                num_latents=self.hparams.num_latents,
                num_latent_channels=self.hparams.num_latent_channels,
                activation_checkpointing=self.hparams.activation_checkpointing,
                activation_offloading=self.hparams.activation_offloading,
            )
        )

        encoder_params = getattr(self.hparams.encoder, "params", None)
        if encoder_params is not None:
            if is_checkpoint(encoder_params):
                ckpt_path = Path(encoder_params)
                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        "Encoder checkpoint not found: "
                        f"{ckpt_path}. Train a classifier first (for example, "
                        "`python examples/training/img_clf/train_my_256_clf.py`) "
                        "or set `encoder.params` to a valid HuggingFace model id."
                    )
                #加载本地训练的 .ckpt 
                from perceiver.model.vision.image_classifier.lightning import LitImageClassifier

                source_model = LitImageClassifier.load_from_checkpoint(encoder_params, params=None)
                state_dict = source_model.model.encoder.state_dict()
            else:
                from perceiver.model.vision.image_classifier.huggingface import PerceiverImageClassifier
                #直接加载 HuggingFace模型 
                source_model = PerceiverImageClassifier.from_pretrained(encoder_params)
                state_dict = source_model.backend_model.encoder.state_dict()

            missing, unexpected = self.model.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained encoder. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        if self.hparams.params is not None:
            if is_checkpoint(self.hparams.params):
                wrapper = LitAnomalyDetector.load_from_checkpoint(self.hparams.params, params=None)
                self.model.load_state_dict(wrapper.model.state_dict())
            else:
                raise ValueError(
                    "Only checkpoint loading is supported currently for LitAnomalyDetector. "
                    "Provide a .ckpt path via --model.params."
                )

        self.pixel_loss_weight = pixel_loss_weight
        self.image_loss_weight = image_loss_weight

        # Register as buffer so it moves with device
        self.register_buffer("pixel_pos_weight", torch.tensor(float(pixel_pos_weight), dtype=torch.float32))

        self.image_loss_fn = nn.BCEWithLogitsLoss()
        self.pixel_auroc, self.image_auroc = make_auroc_metrics()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    @staticmethod
    def _mask_to_channels_last(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 4:
            raise ValueError(f"Expected 4D mask, got shape={tuple(mask.shape)}")
        if mask.shape[-1] == 1:
            return mask
        if mask.shape[1] == 1:
            return mask.permute(0, 2, 3, 1).contiguous()
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    def step(self, batch: Dict[str, Any], stage: str):
        x = batch["image"]
        y_mask = self._mask_to_channels_last(batch["mask"]).float()

        outputs = self(x)
        if "anomaly_logits" not in outputs:
            raise KeyError('Model output must include key "anomaly_logits".')

        pred_logits = outputs["anomaly_logits"]
        if pred_logits.shape != y_mask.shape:
            raise ValueError(
                f"Shape mismatch: anomaly_logits={tuple(pred_logits.shape)} vs mask={tuple(y_mask.shape)}"
            )

        if self.hparams.loss_type == "focal":
            loss_pix = self._focal_bce_with_logits(
                pred_logits,
                y_mask,
                gamma=float(self.hparams.focal_gamma),
            )
        else:
            # E1: weighted BCE for sparse anomaly pixels
            loss_pix = F.binary_cross_entropy_with_logits(
                pred_logits,
                y_mask,
                pos_weight=self.pixel_pos_weight.to(pred_logits.device),
            )

        loss_img = torch.tensor(0.0, device=pred_logits.device)
        use_image_loss = (
            self.image_loss_weight > 0.0
            and "image_score" in outputs
            and "label" in batch
        )
        if use_image_loss:
            image_score = outputs["image_score"].float().view(-1)
            image_label = batch["label"].float().view(-1)
            loss_img = self.image_loss_fn(image_score, image_label)

        loss = self.pixel_loss_weight * loss_pix + self.image_loss_weight * loss_img

        bs = x.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), sync_dist=(stage != "train"), batch_size=bs)
        self.log(f"{stage}_loss_pix", loss_pix, prog_bar=(stage != "train"), sync_dist=(stage != "train"), batch_size=bs)
        if use_image_loss:
            self.log(f"{stage}_loss_img", loss_img, prog_bar=False, sync_dist=(stage != "train"), batch_size=bs)

        with torch.no_grad():
            pred_prob = torch.sigmoid(pred_logits)
            update_pixel_auroc(self.pixel_auroc, pred_prob, y_mask)

            if "image_score" in outputs and "label" in batch:
                update_image_auroc(self.image_auroc, outputs["image_score"], batch["label"])

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.step(batch, stage="test")

    def on_validation_epoch_end(self):
        pixel_auc = compute_and_reset(self.pixel_auroc)
        if pixel_auc is not None:
            self.log("val_pixel_auroc", pixel_auc, prog_bar=True, sync_dist=True)

        image_auc = compute_and_reset(self.image_auroc)
        if image_auc is not None:
            self.log("val_image_auroc", image_auc, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        pixel_auc = compute_and_reset(self.pixel_auroc)
        if pixel_auc is not None:
            self.log("test_pixel_auroc", pixel_auc, sync_dist=True)

        image_auc = compute_and_reset(self.image_auroc)
        if image_auc is not None:
            self.log("test_image_auroc", image_auc, sync_dist=True)

    def _focal_bce_with_logits(self, logits, targets, gamma: float):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pixel_pos_weight.to(logits.device),
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        return ((1.0 - pt).pow(gamma) * bce).mean()

    """ def configure_optimizers(self):
        if self.hparams.encoder_lr is not None and self.hparams.decoder_lr is not None:
            encoder_params = list(self.model.encoder.parameters())
            encoder_param_ids = {id(p) for p in encoder_params}
            decoder_params = [p for p in self.model.parameters() if id(p) not in encoder_param_ids]

            return AdamW(
                [
                    {"params": encoder_params, "lr": float(self.hparams.encoder_lr)},
                    {"params": decoder_params, "lr": float(self.hparams.decoder_lr)},
                ]
            )

        return AdamW(self.parameters(), lr=1e-4) """

    def configure_optimizers(self):
    # 冻结 Encoder！
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # 只训练 decoder
        from torch.optim import AdamW
        return AdamW(self.model.decoder.parameters(), lr=1e-4)