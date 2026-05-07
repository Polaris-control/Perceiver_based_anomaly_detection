from typing import Any, Dict

import torch
import torchmetrics as tm
from torch import nn

from perceiver.model.core.lightning import LitPerceiverIO, is_checkpoint
from perceiver.model.vision.anomaly_detector.backend import (
    AnomalyDecoderConfig,
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyEncoderConfig,
)


class LitAnomalyDetector(LitPerceiverIO):
    """
    Expected batch fields:
      - batch["image"]: (B, H, W, C), float
      - batch["mask"]:  (B, H, W, 1) or (B, 1, H, W), float in {0,1}
      - batch["label"]: (B,), optional (image-level 0/1)

    Expected model outputs (dict):
      - outputs["anomaly_logits"]: (B, H, W, 1)  # pixel-level logits
      - outputs["image_score"]: (B,), optional   # image-level score/logit
    """

    def __init__(
        self,
        encoder: AnomalyEncoderConfig,
        decoder: AnomalyDecoderConfig,
        pixel_loss_weight: float = 1.0,
        image_loss_weight: float = 0.0,
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

        # Optional checkpoint init (same style as existing modules)
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

        # Pixel segmentation loss on logits + binary mask
        self.pixel_loss_fn = nn.BCEWithLogitsLoss()
        # Optional image-level loss if label/image_score exists
        self.image_loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.pixel_auroc = tm.AUROC(pos_label=1)
        self.image_auroc = tm.AUROC(pos_label=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    @staticmethod
    def _mask_to_channels_last(mask: torch.Tensor) -> torch.Tensor:
        # Accept both (B, H, W, 1) and (B, 1, H, W), return (B, H, W, 1)
        if mask.ndim != 4:
            raise ValueError(f"Expected 4D mask, got shape={tuple(mask.shape)}")
        if mask.shape[-1] == 1:
            return mask
        if mask.shape[1] == 1:
            return mask.permute(0, 2, 3, 1).contiguous()
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    @staticmethod
    def _flatten_pixel_scores(x: torch.Tensor) -> torch.Tensor:
        # (B, H, W, 1) -> (B*H*W,)
        return x.reshape(-1)

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

        # Pixel-level loss
        loss_pix = self.pixel_loss_fn(pred_logits, y_mask)

        # Optional image-level loss (if both provided)
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

        # Logs
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), sync_dist=(stage != "train"))
        self.log(f"{stage}_loss_pix", loss_pix, prog_bar=(stage != "train"), sync_dist=(stage != "train"))
        if use_image_loss:
            self.log(f"{stage}_loss_img", loss_img, prog_bar=False, sync_dist=(stage != "train"))

        # Metrics (AUROC)
        with torch.no_grad():
            pred_prob = torch.sigmoid(pred_logits)
            pix_pred = self._flatten_pixel_scores(pred_prob)
            pix_true = self._flatten_pixel_scores(y_mask).long()
            
            self.pixel_auroc.update(pix_pred, pix_true)
            if "image_score" in outputs and "label" in batch:
                img_pred = outputs["image_score"].float().view(-1)
                img_true = batch["label"].long().view(-1)
                self.image_auroc.update(img_pred, img_true)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.step(batch, stage="test")

    def on_validation_epoch_end(self):
        pixel_auc = self.pixel_auroc.compute()
        self.log("val_pixel_auroc", pixel_auc, prog_bar=True, sync_dist=True)
        self.pixel_auroc.reset()

        # image AUROC is optional (only if image_score/label were present)
        try:
            image_auc = self.image_auroc.compute()
            self.log("val_image_auroc", image_auc, prog_bar=True, sync_dist=True)
        except Exception:
            pass
        self.image_auroc.reset()

    def on_test_epoch_end(self):
        pixel_auc = self.pixel_auroc.compute()
        self.log("test_pixel_auroc", pixel_auc, sync_dist=True)
        self.pixel_auroc.reset()

        try:
            image_auc = self.image_auroc.compute()
            self.log("test_image_auroc", image_auc, sync_dist=True)
        except Exception:
            pass
        self.image_auroc.reset()