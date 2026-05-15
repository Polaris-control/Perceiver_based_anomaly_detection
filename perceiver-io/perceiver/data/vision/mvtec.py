import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    #单张样本的数据结构（图片路径、mask、类别、是否异常…）
    image_path: Path
    mask_path: Optional[Path]
    category: str
    defect_type: str
    is_anomaly: int


def _default_test_categories(train_categories: Sequence[str]) -> List[str]:
    #自动把没训练的类别当作测试类别
    train_set = set(train_categories)
    return [c for c in MVTEC_CATEGORIES if c not in train_set]


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MVTecDataset(Dataset):
    """
    Returns dict:
      - image: FloatTensor, (H, W, C), normalized
      - label: LongTensor scalar, 0/1 (image-level anomaly)
      - mask: FloatTensor, (H, W, 1), 0/1 (pixel-level anomaly)
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
        use_synthetic_anomaly: bool = False,  # 新增：训练时才打开
    ):
        self.samples = samples
        self.image_size = image_size
        self.channels_last = channels_last
        self.use_synthetic_anomaly = use_synthetic_anomaly

        tfm: List[transforms.Compose] = []
        if augment:
            #图像增强
            tfm.extend(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                ]
            )
        else:
            tfm.extend(
                [
                    transforms.Resize((image_size, image_size)),
                ]
            )

        tfm.append(transforms.ToTensor())

        if normalize_imagenet:
            tfm.append(
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )

        if channels_last:
            tfm.append(channels_to_last)

        self.image_transform = transforms.Compose(tfm)

        # Mask 用 NEAREST → 保持 0 和 1 不变（最近邻插值）
        mask_tfm: List[transforms.Compose] = [
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),  # shape (1, H, W), values in [0, 1]
        ]
        self.mask_transform = transforms.Compose(mask_tfm)

    def _synthesize_anomaly(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给正常图加伪异常，同时生成伪 mask
        x: 已经 transform 完的图像shape: (H, W, C)channels_last
        返回: (augmented_x, synthetic_mask)
        """
        H, W, C = x.shape
        mask = torch.zeros((H, W, 1), dtype=torch.float32, device=x.device)

        #随机生成一个矩形异常区域（模拟划痕、凹陷、污渍）
        # 保证区域足够大，不会太小
        x1 = torch.randint(0, W // 2, (1,)).item()
        y1 = torch.randint(0, H // 2, (1,)).item()
        x2 = torch.randint(x1 + W // 8, W, (1,)).item()
        y2 = torch.randint(y1 + H // 8, H, (1,)).item()

        # 给这个区域加扰动（模拟异常纹理）
        # 因为 x 已经是 imagenet 归一化后的，所以扰动也要在这个空间
        noise = torch.randn_like(x[y1:y2, x1:x2, :]) * 0.3  # 小扰动，不破坏原图
        x[y1:y2, x1:x2, :] = x[y1:y2, x1:x2, :] + noise

        #标记 mask：异常区域=1
        mask[y1:y2, x1:x2, 0] = 1.0

        # 加一点随机噪声异常（模拟微小杂质）
        rand_noise_mask = torch.rand_like(mask) < 0.01  # 1% 像素随机异常
        rand_noise = torch.randn_like(x) * 0.1
        x[rand_noise_mask.squeeze(-1), :] = x[rand_noise_mask.squeeze(-1), :] + rand_noise[rand_noise_mask.squeeze(-1), :]
        mask = torch.logical_or(mask.bool(), rand_noise_mask).float()

        return x, mask

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
            # Keep strict binary mask
            mask = (mask > 0.5).float()

        if self.channels_last:
            mask = channels_to_last(mask)  # (H, W, 1)

        return mask

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        img = self._load_rgb(s.image_path)
        x = self.image_transform(img)  # (H, W, C) if channels_last else (C, H, W)
        m = self._load_mask(s.mask_path, self.image_size)

        # 训练时：如果开启了 synthetic anomaly，给 good 图加伪异常
        if self.use_synthetic_anomaly:
            # 只有训练集才会进来，x 是 transform 完的 good 图
            x, m = self._synthesize_anomaly(x)
            # 图像级 label 改成 1（因为我们加了异常）
            label = torch.tensor(1, dtype=torch.long)
        else:
            # 验证/测试：用原来的 label
            label = torch.tensor(s.is_anomaly, dtype=torch.long)

        return {
            "image": x,
            "label": label,
            "mask": m,
            "category": s.category,
            "defect_type": s.defect_type,
            "path": str(s.image_path),
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
    ):
        super().__init__()
        self.save_hyperparameters()

        if train_categories is None:
            # Example default split (edit as needed)
            train_categories = MVTEC_CATEGORIES[:10]

        if test_categories is None:
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
        # image-level normal/anomaly
        return 2

    def _collect_train_samples_for_category(self, category_dir: Path, category: str) -> List[MVTecSample]:
        #训练集只使用正常样本 无监督/零样本异常检测
        samples: List[MVTecSample] = []
        good_dir = category_dir / "train" / "good"
        if not good_dir.exists():
            return samples

        for p in sorted(good_dir.rglob("*")):
            if p.is_file() and _is_image_file(p):
                samples.append(
                    MVTecSample(
                        image_path=p,
                        mask_path=None,
                        category=category,
                        defect_type="good",
                        is_anomaly=0,
                    )
                )
        return samples

    def _find_mask_path(self, category_dir: Path, defect_type: str, image_path: Path) -> Optional[Path]:
        # Standard MVTec naming: image "xxx.png" -> mask "xxx_mask.png"
        stem = image_path.stem
        suffix = image_path.suffix
        candidate = category_dir / "ground_truth" / defect_type / f"{stem}_mask{suffix}"
        if candidate.exists():
            return candidate

        # Fallback: same stem if custom naming exists
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
            for p in sorted(defect_dir.rglob("*")):
                if p.is_file() and _is_image_file(p):
                    mask_path = None
                    if is_anomaly == 1:
                        mask_path = self._find_mask_path(category_dir, defect_type, p)

                    samples.append(
                        MVTecSample(
                            image_path=p,
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

        for c in categories:
            category_dir = root / c
            if not category_dir.exists():
                continue

            if split == "train":
                all_samples.extend(self._collect_train_samples_for_category(category_dir, c))
            elif split in {"val", "test"}:
                all_samples.extend(self._collect_test_samples_for_category(category_dir, c))
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
            use_synthetic_anomaly = True, #训练集 开伪异常
        )
        self.ds_val = MVTecDataset(
            val_samples,
            image_size=self.hparams.image_size,
            channels_last=self.hparams.channels_last,
            normalize_imagenet=self.hparams.normalize_imagenet,
            augment=False,
            use_synthetic_anomaly = False, #验证关， 用真实异常
        )
        self.ds_test = MVTecDataset(
            test_samples,
            image_size=self.hparams.image_size,
            channels_last=self.hparams.channels_last,
            normalize_imagenet=self.hparams.normalize_imagenet,
            augment=False,
            use_synthetic_anomaly = False, # 测试关，用真实异常
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