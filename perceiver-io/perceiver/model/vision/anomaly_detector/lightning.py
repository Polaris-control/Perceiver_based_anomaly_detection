from pathlib import Path
import json
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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

def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": "198037",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": "post-fix-v8",
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


class LitAnomalyDetector(LitPerceiverIO):
    def __init__(
        self,
        encoder: AnomalyEncoderConfig,
        decoder: AnomalyDecoderConfig,
        pixel_loss_weight: float = 1.0,
        image_loss_weight: float = 0.0,
        pixel_pos_weight: float = 5.0,
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
        dice_loss_weight: float = 0.05,
        dice_smooth: float = 1.0,
        encoder_lr: Optional[float] = None,
        decoder_lr: Optional[float] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_projs: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj"),
        lora_lr: Optional[float] = None,
        area_loss_weight: float = 1.0,
        edge_loss_weight: float = 0.0,
        focal_alpha: float = 0.25,
        ranking_loss_weight: float = 0.3,          # Ranking loss 权重，推荐 0.3
        use_hard_mining: bool = False,              # Hard Negative Mining 开关（默认关闭）
        hard_neg_ratio: float = 0.05,               # 仅当 use_hard_mining=True 时生效
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

        # 加载预训练编码器权重（分类器）
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
                from perceiver.model.vision.image_classifier.lightning import LitImageClassifier
                source_model = LitImageClassifier.load_from_checkpoint(encoder_params, params=None)
                state_dict = source_model.model.encoder.state_dict()
            else:
                from perceiver.model.vision.image_classifier.huggingface import PerceiverImageClassifier
                source_model = PerceiverImageClassifier.from_pretrained(encoder_params)
                state_dict = source_model.backend_model.encoder.state_dict()
            missing, unexpected = self.model.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained encoder. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # LoRA 注入
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
        self.area_loss_weight = area_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.focal_alpha = focal_alpha
        self.ranking_loss_weight = ranking_loss_weight
        self.use_hard_mining = use_hard_mining
        self.hard_neg_ratio = hard_neg_ratio

        self.register_buffer("pixel_pos_weight", torch.tensor(float(pixel_pos_weight), dtype=torch.float32))
        self.image_loss_fn = nn.BCEWithLogitsLoss()

        self.val_pixel_auroc, self.val_image_auroc = make_auroc_metrics()
        self.test_pixel_auroc, self.test_image_auroc = make_auroc_metrics()

        self._dbg_val_pix_updates = 0
        self._dbg_val_pix_skips = 0
        self._dbg_val_img_updates = 0
        self._dbg_val_img_skips = 0

    def _compute_edge_map(self, x):
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32, device=x.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32, device=x.device,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2)

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

    def _update_image_auroc_no_skip(self, auroc_metric, image_score, true_label):
        pred = image_score.float().view(-1)
        true = true_label.long().view(-1)
        if true.numel() == 0:
            return False
        auroc_metric.update(pred, true)
        return True

    # 均值 Focal Loss（原版，无逐像素）
    def _focal_bce_with_logits(self, logits, targets, gamma, alpha):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pixel_pos_weight.to(logits.device)
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = (1.0 - pt).pow(gamma) * alpha_weight
        return (focal_weight * bce).mean()   # 直接返回均值，不逐像素

    def _compute_ranking_loss(self, pred_logits, y_mask, margin=1.0):
        anom_mask = (y_mask > 0.5).float()
        normal_mask = (y_mask <= 0.5).float()
        if anom_mask.sum() > 0 and normal_mask.sum() > 0:
            anom_mean = (pred_logits * anom_mask).sum() / (anom_mask.sum() + 1e-8)
            normal_mean = (pred_logits * normal_mask).sum() / (normal_mask.sum() + 1e-8)
            loss_rank = torch.relu(margin - (anom_mean - normal_mean))
        else:
            loss_rank = torch.tensor(0.0, device=pred_logits.device)
        return loss_rank

    def step(self, batch: Dict[str, Any], stage: str, batch_idx: int = -1):
        x = batch["image"]
        y_mask = self._mask_to_channels_last(batch["mask"]).float()
        outputs = self(x)
        pred_logits = outputs["anomaly_logits"]

        if pred_logits.shape != y_mask.shape:
            raise ValueError(f"Shape mismatch: {pred_logits.shape} vs {y_mask.shape}")

        # 1. 像素损失（均值 Focal/BCE，不使用逐像素）
        if self.hparams.loss_type == "focal":
            loss_pix_bce = self._focal_bce_with_logits(
                pred_logits, y_mask,
                gamma=self.hparams.focal_gamma,
                alpha=self.focal_alpha,
            )
        else:
            loss_pix_bce = F.binary_cross_entropy_with_logits(
                pred_logits, y_mask,
                pos_weight=self.pixel_pos_weight.to(pred_logits.device),
            )

        dice_weight = self.hparams.dice_loss_weight
        if dice_weight > 0.0:
            loss_pix_dice = self._soft_dice_loss_with_logits(pred_logits, y_mask, smooth=self.hparams.dice_smooth)
        else:
            loss_pix_dice = torch.zeros_like(loss_pix_bce)

        loss_pix = loss_pix_bce + dice_weight * loss_pix_dice

        # 2. 面积损失
        pred_prob = torch.sigmoid(pred_logits)   # 统一计算一次
        pred_area = pred_prob.mean(dim=[1, 2, 3])
        target_area = y_mask.mean(dim=[1, 2, 3])
        area_loss = F.smooth_l1_loss(pred_area, target_area, beta=0.1) if self.area_loss_weight > 0 else torch.tensor(0.0, device=pred_logits.device)

        # 3. 图像级损失（可选）
        loss_img = torch.tensor(0.0, device=pred_logits.device)
        use_image_loss = (self.image_loss_weight > 0.0 and "image_score" in outputs and "label" in batch)
        if use_image_loss:
            image_score = outputs["image_score"].float().view(-1)
            image_label = batch["label"].float().view(-1)
            loss_img = self.image_loss_fn(image_score, image_label)

        # 4. 边缘损失（可选）
        loss_edge = torch.tensor(0.0, device=pred_logits.device)
        if self.edge_loss_weight > 0.0:
            pred_edge = self._compute_edge_map(pred_prob.permute(0, 3, 1, 2))
            true_edge = self._compute_edge_map(y_mask.permute(0, 3, 1, 2))
            loss_edge = F.l1_loss(pred_edge, true_edge)

        # 5. Ranking Loss（直接优化 logits 排序）
        loss_rank = self._compute_ranking_loss(pred_logits, y_mask, margin=1.0)

        # 总损失
        loss = (self.pixel_loss_weight * loss_pix
                + self.image_loss_weight * loss_img
                + self.area_loss_weight * area_loss
                + self.edge_loss_weight * loss_edge
                + self.ranking_loss_weight * loss_rank)

        bs = x.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), batch_size=bs)
        self.log(f"{stage}_loss_pix", loss_pix, prog_bar=(stage != "train"), batch_size=bs)
        self.log(f"{stage}_loss_pix_bce", loss_pix_bce, prog_bar=False, batch_size=bs)
        if dice_weight > 0.0:
            self.log(f"{stage}_loss_pix_dice", loss_pix_dice, prog_bar=False, batch_size=bs)
        if use_image_loss:
            self.log(f"{stage}_loss_img", loss_img, prog_bar=False, batch_size=bs)
        if self.area_loss_weight > 0:
            self.log(f"{stage}_area_loss", area_loss, prog_bar=False, batch_size=bs)
            self.log(f"{stage}_pred_area_mean", pred_area.mean(), prog_bar=False, batch_size=bs)
            self.log(f"{stage}_target_area_mean", target_area.mean(), prog_bar=False, batch_size=bs)
        if self.ranking_loss_weight > 0:
            self.log(f"{stage}_loss_rank", loss_rank, prog_bar=False, batch_size=bs)

        # 更新 AUROC
        with torch.no_grad():
            if stage == "val":
                pix_updated = update_pixel_auroc(self.val_pixel_auroc, pred_prob, y_mask)
                img_updated = False
                if "image_score" in outputs and "label" in batch:
                    img_updated = self._update_image_auroc_no_skip(self.val_image_auroc, outputs["image_score"], batch["label"])
                if pix_updated:
                    self._dbg_val_pix_updates += 1
                else:
                    self._dbg_val_pix_skips += 1
                if img_updated:
                    self._dbg_val_img_updates += 1
                else:
                    self._dbg_val_img_skips += 1
            elif stage == "test":
                update_pixel_auroc(self.test_pixel_auroc, pred_prob, y_mask)
                if "image_score" in outputs and "label" in batch:
                    self._update_image_auroc_no_skip(self.test_image_auroc, outputs["image_score"], batch["label"])

        # 调试日志
        if stage == "val" and 0 <= batch_idx <= 2:
            tmin, tmax = float(y_mask.min().cpu()), float(y_mask.max().cpu())
            _agent_debug_log(
                "H2", "lightning.py:step", "val_batch_snapshot",
                {
                    "epoch": self.current_epoch,
                    "batch_idx": batch_idx,
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
                    "loss_pix_bce": float(loss_pix_bce.detach().cpu()),
                    "loss_pix_dice": float(loss_pix_dice.detach().cpu()),
                    "loss_rank": float(loss_rank.detach().cpu()),
                    "area_loss": float(area_loss.detach().cpu()),
                    "pred_area_mean": float(pred_area.mean().cpu()),
                    "target_area_mean": float(target_area.mean().cpu()),
                    "area_loss_weight": self.area_loss_weight,
                    "ranking_loss_weight": self.ranking_loss_weight,
                }
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
        total_sq = 0.0; lora_sq = 0.0; dec_sq = 0.0
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
            "H1", "lightning.py:on_before_optimizer_step", "grad_norms",
            {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "total_grad_norm": total_sq ** 0.5,
                "lora_grad_norm": lora_sq ** 0.5,
                "decoder_grad_norm": dec_sq ** 0.5,
                "n_trainable_params_missing_grad": n_trainable_missing_grad,
                "n_lora_total": n_lora_total,
                "n_lora_trainable": n_lora_trainable,
                "n_lora_grad_none": n_lora_grad_none,
                "n_lora_nonzero_grad": n_lora_nonzero_grad,
                "lora_grad_max": lora_grad_max,
            }
        )

    def on_validation_epoch_end(self):
        pixel_auc = compute_and_reset(self.val_pixel_auroc)
        image_auc = compute_and_reset(self.val_image_auroc)
        _agent_debug_log(
            "H2", "lightning.py:on_validation_epoch_end", "val_auroc_rollup",
            {
                "epoch": self.current_epoch,
                "val_pix_updates": self._dbg_val_pix_updates,
                "val_pix_skips": self._dbg_val_pix_skips,
                "val_img_updates": self._dbg_val_img_updates,
                "val_img_skips": self._dbg_val_img_skips,
                "pixel_auc": None if pixel_auc is None else float(pixel_auc.cpu()),
                "image_auc": None if image_auc is None else float(image_auc.cpu()),
            }
        )
        if pixel_auc is not None:
            self.log("val_pixel_auroc", pixel_auc, prog_bar=True, sync_dist=True)
        if image_auc is not None:
            self.log("val_image_auroc", image_auc, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        pixel_auc = compute_and_reset(self.test_pixel_auroc)
        image_auc = compute_and_reset(self.test_image_auroc)
        if pixel_auc is not None:
            self.log("test_pixel_auroc", pixel_auc, sync_dist=True)
        if image_auc is not None:
            self.log("test_image_auroc", image_auc, sync_dist=True)

    @staticmethod
    def _soft_dice_loss_with_logits(logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1.0 - dice.mean()

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
            f"[trainable] use_lora={self.hparams.use_lora}, "
            f"lora_params={n_lora_total:,}, lora_trainable={n_lora_trainable:,}, "
            f"decoder_trainable={n_decoder_trainable:,}, "
            f"total_trainable={n_trainable:,}/{n_total:,}"
        )
        _agent_debug_log(
            "E1", "lightning.py:_setup_trainable_parameters", "trainable_summary",
            {
                "use_lora": self.hparams.use_lora,
                "lora_params": n_lora_total,
                "lora_trainable": n_lora_trainable,
                "decoder_trainable": n_decoder_trainable,
                "total_trainable": n_trainable,
                "total_params": n_total,
            }
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
            optimizer = AdamW(
                [{"params": lora_params, "lr": lora_lr}, {"params": dec_params, "lr": dec_lr}],
                weight_decay=1e-5,
            )
        else:
            optimizer = AdamW(self.model.decoder.parameters(), lr=dec_lr, weight_decay=1e-5)

        # 使用余弦退火调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]