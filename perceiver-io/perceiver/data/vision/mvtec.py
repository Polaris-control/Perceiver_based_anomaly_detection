from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from torchvision.transforms.functional import gaussian_blur

import random
import numpy as np
import cv2

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from perceiver.data.vision.common import channels_to_last


MVTEC_CATEGORIES = [
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
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass
class MVTecSample:
    image_path: Path
    mask_path: Optional[Path]
    category: str
    defect_type: str
    is_anomaly: int


def _default_test_categories(train_categories: Sequence[str]) -> List[str]:
    train_set = set(train_categories)
    return [c for c in MVTEC_CATEGORIES if c not in train_set]


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MVTecDataset(Dataset):
    """
    Returns dict:
      - image: FloatTensor, (H, W, C) if channels_last else (C, H, W)
      - label: LongTensor scalar, 0/1 image-level anomaly label
      - mask: FloatTensor, (H, W, 1) if channels_last else (1, H, W)
      - category: str
      - defect_type: str
      - path: str
    """

    def __init__(
        self,
        samples: List[MVTecSample],
        image_size: int = 256,
        channels_last: bool = True,
        normalize_imagenet: bool = True,
        augment: bool = False,
        use_synthetic_anomaly: bool = False,
        synthetic_anomaly_prob: float = 0.15,
        synthetic_min_size_ratio: float = 0.03,
        synthetic_max_size_ratio: float = 0.25,
        synthetic_max_patches: int = 4,
        synthetic_noise_std: float = 0.25,
    ):
        self.samples = samples
        self.image_size = image_size
        self.channels_last = channels_last
        self.use_synthetic_anomaly = use_synthetic_anomaly
        self.synthetic_anomaly_prob = synthetic_anomaly_prob
        self.synthetic_min_size_ratio = synthetic_min_size_ratio
        self.synthetic_max_size_ratio = synthetic_max_size_ratio
        self.synthetic_max_patches = synthetic_max_patches
        self.synthetic_noise_std = synthetic_noise_std

        image_tfm: List[object] = []
        # 训练增强：随机裁剪 + 翻转 + 旋转
        if augment:
            image_tfm.extend(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                ]
            )
        else:
            image_tfm.append(transforms.Resize((image_size, image_size)))

        #统一转换成张量 在做 imageNet归一化
        image_tfm.append(transforms.ToTensor())
        if normalize_imagenet:
            image_tfm.append(
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )

        if channels_last:
            image_tfm.append(channels_to_last)

        self.image_transform = transforms.Compose(image_tfm)

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),  #缩放到相同尺寸  转张量
            ]
        )
     #新增 irregular_mask 函数
    def _generate_irregular_mask(self, h: int, w: int, device) -> torch.Tensor:
        """生成不规则多边形 mask 用于合成异常的软融合"""
        canvas = np.zeros((h, w), dtype=np.uint8)
        num_vertices = random.randint(6, 12)
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        radius_base = min(h, w) * random.uniform(0.2, 0.5) #缺陷大小：占图片宽度的 20%～50%
        points = []
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            radius = radius_base * random.uniform(0.5, 1.5)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(canvas, [points], 1)
        mask = torch.tensor(canvas, dtype=torch.float32, device=device)
        return mask

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Optional[Path], image_size: int) -> torch.Tensor:
        if path is None:
            mask = torch.zeros((1, image_size, image_size), dtype=torch.float32)
        else:
            mask_img = Image.open(path).convert("L")
            mask = self.mask_transform(mask_img)
            mask = (mask > 0.5).float()

        if self.channels_last:
            mask = channels_to_last(mask)

        return mask

    def _synthesize_anomaly(self, x: torch.Tensor):
        """
        Synthesize local anomalies on a normalized channels-last image tensor.
        Input x shape: (H, W, C), value range is roughly [-3, 3].
        """
        if not self.channels_last:
            raise NotImplementedError("Synthetic anomaly currently expects channels_last=True")

        x = x.clone()  # 复制一份，不破坏原图
        h, w, _ = x.shape
        mask = torch.zeros((h, w, 1), dtype=torch.float32, device=x.device)

        max_patches = max(1, int(self.synthetic_max_patches))
        num_patches = torch.randint(1, max_patches + 1, (1,)).item()

        min_w = max(4, int(w * self.synthetic_min_size_ratio))
        max_w = max(min_w + 1, int(w * self.synthetic_max_size_ratio))
        min_h = max(4, int(h * self.synthetic_min_size_ratio))
        max_h = max(min_h + 1, int(h * self.synthetic_max_size_ratio))
        max_w = min(max_w, w)
        max_h = min(max_h, h)

        for _ in range(num_patches):
            patch_w = torch.randint(min_w, max_w + 1, (1,)).item()
            patch_h = torch.randint(min_h, max_h + 1, (1,)).item()

            src_x1 = torch.randint(0, w - patch_w + 1, (1,)).item()
            src_y1 = torch.randint(0, h - patch_h + 1, (1,)).item()
            src_x2 = src_x1 + patch_w
            src_y2 = src_y1 + patch_h
            patch = x[src_y1:src_y2, src_x1:src_x2, :].clone()

            shift_x = torch.randint(-patch_w, patch_w + 1, (1,)).item()
            shift_y = torch.randint(-patch_h, patch_h + 1, (1,)).item()
            dst_x1 = max(0, min(w - patch_w, src_x1 + shift_x))
            dst_y1 = max(0, min(h - patch_h, src_y1 + shift_y))
            dst_x2 = dst_x1 + patch_w
            dst_y2 = dst_y1 + patch_h

            irregular_mask = self._generate_irregular_mask(
                patch_h,
                patch_w,
                x.device,
            ).unsqueeze(-1)

            mode = torch.randint(0, 5, (1,)).item()

            if mode == 0:
                patch_aug = gaussian_blur(
                    patch.permute(2, 0, 1),
                    kernel_size=3, #弱高斯模糊 
                ).permute(1, 2, 0)

            elif mode == 1:
                factor = 0.6 + torch.rand(1).item() * 0.8
                noise = torch.randn_like(patch) * self.synthetic_noise_std
                patch_aug = patch * factor + noise #对比度降低 + noise

            elif mode == 2:
                patch_aug = gaussian_blur(
                    patch.permute(2, 0, 1),
                    kernel_size=7, #强高斯模糊
                ).permute(1, 2, 0)

            elif mode == 3:
                noise = torch.randn_like(patch) * self.synthetic_noise_std
                patch_aug = patch * 0.7 + noise #半透明noise

            else:
                canvas = patch.detach().cpu().numpy()
                min_val = float(canvas.min())
                max_val = float(canvas.max())
                value_range = max_val - min_val

                if value_range > 1e-8:
                    canvas_uint8 = ((canvas - min_val) * (255.0 / value_range)).astype(np.uint8)
                else:
                    canvas_uint8 = np.zeros_like(canvas, dtype=np.uint8)

                num_lines = random.randint(1, 4)
                for _ in range(num_lines):
                    x1 = random.randint(0, patch_w - 1)
                    y1 = random.randint(0, patch_h - 1)
                    x2 = random.randint(0, patch_w - 1)
                    y2 = random.randint(0, patch_h - 1)
                    thickness = random.randint(1, 2)
                    cv2.line(
                        canvas_uint8,
                        (x1, y1),
                        (x2, y2),
                        color=(0, 0, 0),
                        thickness=thickness, #画黑色划痕
                    )

                canvas_float = canvas_uint8.astype(np.float32) / 255.0
                canvas_float = canvas_float * value_range + min_val
                patch_aug = torch.from_numpy(canvas_float).to(
                    device=patch.device,
                    dtype=patch.dtype,
                )

            original = x[dst_y1:dst_y2, dst_x1:dst_x2, :]
            blended = original * (1.0 - irregular_mask) + patch_aug * irregular_mask  #软融合（Alpha Blending）
            x[dst_y1:dst_y2, dst_x1:dst_x2, :] = blended  #把融合好的缺陷块 放回原图对应位置

            mask[dst_y1:dst_y2, dst_x1:dst_x2, 0] = torch.maximum(
                mask[dst_y1:dst_y2, dst_x1:dst_x2, 0],
                irregular_mask[..., 0],
            )

        return x.clamp(-3.0, 3.0), mask

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]

        # 加载原图 PIL
        image = self._load_rgb(sample.image_path)
        x = self.image_transform(image)          # 包含 ToTensor, Normalize, channels_last
        mask = self._load_mask(sample.mask_path, self.image_size)

        label = sample.is_anomaly
        defect_type = sample.defect_type

        if (self.use_synthetic_anomaly and sample.is_anomaly == 0
                and torch.rand(1).item() < self.synthetic_anomaly_prob):
            x, mask = self._synthesize_anomaly(x)
            label = 1
            defect_type = "synthetic"

        return {
            "image": x,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "category": sample.category,
            "defect_type": defect_type,
            "path": str(sample.image_path),
        }

class MVTecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        train_categories: Optional[List[str]] = None,
        test_categories: Optional[List[str]] = None,
        image_size: int = 256,
        channels_last: bool = True,
        normalize_imagenet: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_augment: bool = True,
        include_test_good: bool = True,
        use_synthetic_anomaly: bool = True,
        synthetic_anomaly_prob: float = 0.5,
        synthetic_min_size_ratio: float = 0.02,
        synthetic_max_size_ratio: float = 0.12,
        synthetic_max_patches: int = 3,
        synthetic_noise_std: float = 0.25,
        same_category_eval: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if train_categories is None:
            train_categories = MVTEC_CATEGORIES[:10]

        if test_categories is None:
            if same_category_eval:
                test_categories = list(train_categories)
            else:
                test_categories = _default_test_categories(train_categories)

        self.train_categories = train_categories
        self.test_categories = test_categories

        self.ds_train: Optional[MVTecDataset] = None
        self.ds_val: Optional[MVTecDataset] = None
        self.ds_test: Optional[MVTecDataset] = None

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        if self.hparams.channels_last:
            return self.hparams.image_size, self.hparams.image_size, 3
        return 3, self.hparams.image_size, self.hparams.image_size

    @property
    def num_classes(self) -> int:
        return 2

    def _collect_train_samples_for_category(self, category_dir: Path, category: str) -> List[MVTecSample]:
        samples: List[MVTecSample] = []
        good_dir = category_dir / "train" / "good"

        if not good_dir.exists():
            return samples

        for path in sorted(good_dir.rglob("*")):
            if path.is_file() and _is_image_file(path):
                samples.append(
                    MVTecSample(
                        image_path=path,
                        mask_path=None,
                        category=category,
                        defect_type="good",
                        is_anomaly=0,
                    )
                )

        return samples

    def _find_mask_path(self, category_dir: Path, defect_type: str, image_path: Path) -> Optional[Path]:
        candidate = (
            category_dir
            / "ground_truth"
            / defect_type
            / f"{image_path.stem}_mask{image_path.suffix}"
        )
        if candidate.exists():
            return candidate

        fallback = category_dir / "ground_truth" / defect_type / image_path.name
        if fallback.exists():
            return fallback

        return None

    def _collect_test_samples_for_category(self, category_dir: Path, category: str) -> List[MVTecSample]:
        samples: List[MVTecSample] = []
        test_dir = category_dir / "test"

        if not test_dir.exists():
            return samples

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            if defect_type == "good" and not self.hparams.include_test_good:
                continue

            is_anomaly = 0 if defect_type == "good" else 1

            for path in sorted(defect_dir.rglob("*")):
                if path.is_file() and _is_image_file(path):
                    mask_path = None
                    if is_anomaly == 1:
                        mask_path = self._find_mask_path(category_dir, defect_type, path)

                    samples.append(
                        MVTecSample(
                            image_path=path,
                            mask_path=mask_path,
                            category=category,
                            defect_type=defect_type,
                            is_anomaly=is_anomaly,
                        )
                    )

        return samples

    def _build_samples(self, categories: Sequence[str], split: str) -> List[MVTecSample]:
        root = Path(self.hparams.dataset_dir)
        all_samples: List[MVTecSample] = []

        for category in categories:
            category_dir = root / category
            if not category_dir.exists():
                continue

            if split == "train":
                all_samples.extend(self._collect_train_samples_for_category(category_dir, category))
            elif split in {"val", "test"}:
                all_samples.extend(self._collect_test_samples_for_category(category_dir, category))
            else:
                raise ValueError(f"Unsupported split: {split}")

        return all_samples

    def setup(self, stage: Optional[str] = None) -> None:
        train_samples = self._build_samples(self.train_categories, split="train")
        val_samples = self._build_samples(self.test_categories, split="val")
        test_samples = self._build_samples(self.test_categories, split="test")

        self.ds_train = MVTecDataset(
            train_samples,
            image_size=self.hparams.image_size,
            channels_last=self.hparams.channels_last,
            normalize_imagenet=self.hparams.normalize_imagenet,
            augment=self.hparams.train_augment,
            use_synthetic_anomaly=self.hparams.use_synthetic_anomaly,
            synthetic_anomaly_prob=self.hparams.synthetic_anomaly_prob,
            synthetic_min_size_ratio=self.hparams.synthetic_min_size_ratio,
            synthetic_max_size_ratio=self.hparams.synthetic_max_size_ratio,
            synthetic_max_patches=self.hparams.synthetic_max_patches,
            synthetic_noise_std=self.hparams.synthetic_noise_std,
        )

        self.ds_val = MVTecDataset(
            val_samples,
            image_size=self.hparams.image_size,
            channels_last=self.hparams.channels_last,
            normalize_imagenet=self.hparams.normalize_imagenet,
            augment=False,
            use_synthetic_anomaly=False,
        )

        self.ds_test = MVTecDataset(
            test_samples,
            image_size=self.hparams.image_size,
            channels_last=self.hparams.channels_last,
            normalize_imagenet=self.hparams.normalize_imagenet,
            augment=False,
            use_synthetic_anomaly=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )