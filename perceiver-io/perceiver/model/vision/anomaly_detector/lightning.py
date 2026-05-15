from pathlib import Path
import json
import time
import warnings
from typing import Any, Dict, Optional, Tuple

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
from perceiver.model.vision.anomaly_detector.lora import (
    LinearWithLoRA,
    inject_lora_into_encoder,
    iter_lora_parameters,
    lora_parameter_count,
)
from perceiver.model.vision.anomaly_detector.metrics import (
    compute_and_reset,
    make_auroc_metrics,
    update_image_auroc,
    update_pixel_auroc,
)

_DEBUG_LOG_PATH = Path(r"C:\Users\20763\Desktop\zero-shot\debug-198037.log")

#假设并添加 NDJSON 日志：验证 AUROC 是否因跳过 update 而失真、梯度是否爆炸/消失、预测是否塌缩、损失与 mask 分布
def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    # region agent log
    payload = {
        "sessionId": "198037",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": "post-fix-v4",
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


class LitAnomalyDetector(LitPerceiverIO):
    def __init__(
        self,
        encoder: AnomalyEncoderConfig,
        decoder: AnomalyDecoderConfig,
        pixel_loss_weight: float = 1.0,
        image_loss_weight: float = 0.1,
        pixel_pos_weight: float = 2.0,  
        loss_type: str = "bce",
        focal_gamma: float = 1.5,
        encoder_lr: Optional[float] = None,
        decoder_lr: Optional[float] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_projs: Tuple[str, ...] = ("q_proj", "v_proj"),
        lora_lr: Optional[float] = None,
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

        # Inject LoRA before loading a full anomaly ckpt so keys like ``q_proj.base.*`` / ``lora_*`` match.
        if self.hparams.use_lora:
            if self.hparams.lora_rank <= 0:
                raise ValueError("lora_rank must be positive when use_lora=True")
            n_wrapped, n_skip = inject_lora_into_encoder(
                self.model.encoder,
                rank=int(self.hparams.lora_rank),
                alpha=float(self.hparams.lora_alpha),
                dropout=float(self.hparams.lora_dropout),
                target_proj_names=self.hparams.lora_target_projs,
            )
            n_lora = lora_parameter_count(self.model.encoder)
            print(
                f"LoRA: wrapped {n_wrapped} Linear layers on MultiHeadAttention "
                f"(skipped already-wrapped: {n_skip}), lora_param_count={n_lora}"
            )
            if n_wrapped == 0:
                warnings.warn(
                    "use_lora=True but no MultiHeadAttention projections were wrapped. "
                    "Check encoder structure or lora_target_projs.",
                    stacklevel=2,
                )

        if self.hparams.params is not None:
            if is_checkpoint(self.hparams.params):
                wrapper = LitAnomalyDetector.load_from_checkpoint(self.hparams.params, params=None)
                load_strict = not bool(self.hparams.use_lora)
                incompatible = self.model.load_state_dict(wrapper.model.state_dict(), strict=load_strict)
                if not load_strict:
                    print(
                        "Loaded anomaly checkpoint with strict=False (LoRA mode). "
                        f"Missing keys: {len(incompatible.missing_keys)}, "
                        f"Unexpected keys: {len(incompatible.unexpected_keys)}"
                    )
            else:
                raise ValueError(
                    "Only checkpoint loading is supported currently for LitAnomalyDetector. "
                    "Provide a .ckpt path via --model.params."
                )

        self._setup_trainable_parameters()

        self.pixel_loss_weight = pixel_loss_weight
        self.image_loss_weight = image_loss_weight

        # Register as buffer so it moves with device
        self.register_buffer("pixel_pos_weight", torch.tensor(float(pixel_pos_weight), dtype=torch.float32))

        self.image_loss_fn = nn.BCEWithLogitsLoss()
        self.pixel_auroc, self.image_auroc = make_auroc_metrics()
        self._dbg_val_pix_updates = 0
        self._dbg_val_pix_skips = 0
        self._dbg_val_img_updates = 0
        self._dbg_val_img_skips = 0

        

    def on_validation_epoch_start(self) -> None:
        self._dbg_val_pix_updates = 0
        self._dbg_val_pix_skips = 0
        self._dbg_val_img_updates = 0
        self._dbg_val_img_skips = 0

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

    def step(self, batch: Dict[str, Any], stage: str, batch_idx: int = -1):
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
            loss_pix = F.binary_cross_entropy_with_logits(
                pred_logits,
                y_mask,
                pos_weight=self.pixel_pos_weight.to(pred_logits.device),
            )

        #图像集损失不变
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

        #总损失
        loss = self.pixel_loss_weight * loss_pix + self.image_loss_weight * loss_img

        bs = x.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), sync_dist=(stage != "train"), batch_size=bs)
        self.log(f"{stage}_loss_pix", loss_pix, prog_bar=(stage != "train"), sync_dist=(stage != "train"), batch_size=bs)
        if use_image_loss:
            self.log(f"{stage}_loss_img", loss_img, prog_bar=False, sync_dist=(stage != "train"), batch_size=bs)

        with torch.no_grad():
            pred_prob = torch.sigmoid(pred_logits)
            pix_updated = update_pixel_auroc(self.pixel_auroc, pred_prob, y_mask)

            img_updated = False
            if "image_score" in outputs and "label" in batch:
                img_updated = update_image_auroc(self.image_auroc, outputs["image_score"], batch["label"])

        if stage == "val":
            if pix_updated:
                self._dbg_val_pix_updates += 1
            else:
                self._dbg_val_pix_skips += 1
            if "image_score" in outputs and "label" in batch:
                if img_updated:
                    self._dbg_val_img_updates += 1
                else:
                    self._dbg_val_img_skips += 1

        if stage == "val" and 0 <= batch_idx <= 2:
            tmin, tmax = float(y_mask.min().cpu()), float(y_mask.max().cpu())
            _agent_debug_log(
                "H2",
                "lightning.py:step",
                "val_batch_snapshot",
                {
                    "epoch": int(self.current_epoch),
                    "batch_idx": int(batch_idx),
                    "mask_mean": float(y_mask.mean().cpu()),
                    "mask_min": tmin,
                    "mask_max": tmax,
                    "pix_auroc_updated": bool(pix_updated),
                    "img_auroc_updated": bool(img_updated),
                    "pred_prob_mean": float(pred_prob.mean().cpu()),
                    "pred_prob_std": float(pred_prob.std().cpu()),
                    "pred_logit_mean": float(pred_logits.mean().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "loss_pix": float(loss_pix.detach().cpu()),
                    "use_lora": bool(self.hparams.use_lora),
                },
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train", batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        self.step(batch, stage="val", batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        self.step(batch, stage="test", batch_idx=batch_idx)

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.global_step % 25 != 0:
            return

        dec_ids = {id(p) for p in self.model.decoder.parameters()}
        lora_params = list(iter_lora_parameters(self.model.encoder))
        lora_ids = {id(p) for p in lora_params}

        total_sq = 0.0
        lora_sq = 0.0
        dec_sq = 0.0

        n_trainable_missing_grad = 0
        n_lora_total = len(lora_params)
        n_lora_trainable = 0
        n_lora_grad_none = 0
        n_lora_nonzero_grad = 0
        lora_grad_max = 0.0

        for p in self.parameters():
            if not p.requires_grad:
                continue

            if p.grad is None:
                n_trainable_missing_grad += 1
                continue

            gn_sq = float(p.grad.norm().item() ** 2)
            total_sq += gn_sq

            pid = id(p)
            if pid in lora_ids:
                lora_sq += gn_sq
            elif pid in dec_ids:
                dec_sq += gn_sq

        for p in lora_params:
            if p.requires_grad:
                n_lora_trainable += 1
            if p.grad is None:
                n_lora_grad_none += 1
            else:
                grad_abs_max = float(p.grad.detach().abs().max().item())
                lora_grad_max = max(lora_grad_max, grad_abs_max)
                if grad_abs_max > 0:
                    n_lora_nonzero_grad += 1

        _agent_debug_log(
            "H1",
            "lightning.py:on_before_optimizer_step",
            "grad_norms",
            {
                "global_step": int(self.global_step),
                "epoch": int(self.current_epoch),
                "total_grad_norm": total_sq ** 0.5,
                "lora_grad_norm": lora_sq ** 0.5,
                "decoder_grad_norm": dec_sq ** 0.5,
                "n_trainable_params_missing_grad": int(n_trainable_missing_grad),
                "n_lora_total": int(n_lora_total),
                "n_lora_trainable": int(n_lora_trainable),
                "n_lora_grad_none": int(n_lora_grad_none),
                "n_lora_nonzero_grad": int(n_lora_nonzero_grad),
                "lora_grad_max": float(lora_grad_max),
            },
        )

    def on_validation_epoch_end(self):
        pixel_auc = compute_and_reset(self.pixel_auroc)
        image_auc = compute_and_reset(self.image_auroc)
        _agent_debug_log(
            "H2",
            "lightning.py:on_validation_epoch_end",
            "val_auroc_rollup",
            {
                "epoch": int(self.current_epoch),
                "val_pix_updates": int(self._dbg_val_pix_updates),
                "val_pix_skips": int(self._dbg_val_pix_skips),
                "val_img_updates": int(self._dbg_val_img_updates),
                "val_img_skips": int(self._dbg_val_img_skips),
                "pixel_auc": None if pixel_auc is None else float(pixel_auc.cpu()),
                "image_auc": None if image_auc is None else float(image_auc.cpu()),
            },
        )
        if pixel_auc is not None:
            self.log("val_pixel_auroc", pixel_auc, prog_bar=True, sync_dist=True)

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
    
    def _setup_trainable_parameters(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        lora_params = list(iter_lora_parameters(self.model.encoder))
        if self.hparams.use_lora:
            for p in lora_params:
                p.requires_grad = True

        for p in self.model.decoder.parameters():
            p.requires_grad = True

        n_lora_total = sum(p.numel() for p in lora_params)
        n_lora_trainable = sum(p.numel() for p in lora_params if p.requires_grad)
        n_decoder_trainable = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())

        print(
            "[trainable] "
            f"use_lora={bool(self.hparams.use_lora)}, "
            f"lora_tensors={len(lora_params)}, "
            f"lora_params={n_lora_total:,}, "
            f"lora_trainable={n_lora_trainable:,}, "
            f"decoder_trainable={n_decoder_trainable:,}, "
            f"total_trainable={n_trainable:,}/{n_total:,}"
        )

        _agent_debug_log(
            "E1",
            "lightning.py:_setup_trainable_parameters",
            "trainable_summary",
            {
                "use_lora": bool(self.hparams.use_lora),
                "lora_tensors": int(len(lora_params)),
                "lora_params": int(n_lora_total),
                "lora_trainable": int(n_lora_trainable),
                "decoder_trainable": int(n_decoder_trainable),
                "total_trainable": int(n_trainable),
                "total_params": int(n_total),
            },
        )

    def configure_optimizers(self):
        dec_lr = float(self.hparams.decoder_lr) if self.hparams.decoder_lr is not None else 1e-4

        if self.hparams.use_lora:
            lora_lr = self.hparams.lora_lr
            if lora_lr is None and self.hparams.encoder_lr is not None:
                lora_lr = self.hparams.encoder_lr
            if lora_lr is None:
                lora_lr = dec_lr
            lora_lr = float(lora_lr)

            lora_params = list(iter_lora_parameters(self.model.encoder))
            dec_params = list(self.model.decoder.parameters())

            n_lora_trainable = sum(p.requires_grad for p in lora_params)
            n_dec_trainable = sum(p.requires_grad for p in dec_params)

            print(
                "[optimizer] "
                f"lora_tensors={len(lora_params)}, "
                f"lora_trainable_tensors={n_lora_trainable}, "
                f"decoder_tensors={len(dec_params)}, "
                f"decoder_trainable_tensors={n_dec_trainable}, "
                f"lora_lr={lora_lr}, decoder_lr={dec_lr}"
            )

            _agent_debug_log(
                "E1",
                "lightning.py:configure_optimizers",
                "optimizer_summary",
                {
                    "lora_tensors": int(len(lora_params)),
                    "lora_trainable_tensors": int(n_lora_trainable),
                    "decoder_tensors": int(len(dec_params)),
                    "decoder_trainable_tensors": int(n_dec_trainable),
                    "lora_lr": float(lora_lr),
                    "decoder_lr": float(dec_lr),
                },
            )

            return AdamW(
                [
                    {"params": lora_params, "lr": lora_lr},
                    {"params": dec_params, "lr": dec_lr},
                ],
                weight_decay=1e-5,
            )

        return AdamW(self.model.decoder.parameters(), lr=dec_lr, weight_decay=1e-5)