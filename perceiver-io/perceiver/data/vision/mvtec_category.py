"""
MVTec AD 数据集的多类别分类数据模块。
每个样本返回类别标签（0-9），用于预训练分类器。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict
import random

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

CATEGORY_TO_CLASS_ID = {
    "bottle": 0,
    "cable": 1,
    "capsule": 2,
    "carpet": 3,
    "grid": 4,
    "hazelnut": 5,
    "leather": 6,
    "metal_nut": 7,
    "pill": 8,
    "screw": 9,
    "tile": 10,
    "toothbrush": 11,
    "transistor": 12,
    "wood": 13,
    "zipper": 14,
}


@dataclass
class MVTecSample:
    image_path: Path
    category: str


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MVTecCategoryDataset(Dataset):
    """
    用于多类别分类的数据集，只使用训练集中的 good 样本。
    返回：
        "image": Tensor (H, W, C) 或 (C, H, W) 取决于 channels_last
        "label": Tensor (0-9) 类别ID
        "category": str 类别名称
    """

    def __init__(
        self,
        samples: List[MVTecSample],
        image_size: int = 224,
        channels_last: bool = True,
        normalize_imagenet: bool = True,
        augment: bool = False,
    ):
        self.samples = samples
        self.image_size = image_size
        self.channels_last = channels_last
        self.normalize_imagenet = normalize_imagenet
        self.augment = augment

        image_tfm: List = []
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        x = self.image_transform(image)

        # 关键：返回类别ID而不是异常标签
        class_id = CATEGORY_TO_CLASS_ID[sample.category]

        return {
            "image": x,
            "label": torch.tensor(class_id, dtype=torch.long),
            "category": sample.category,
        }


class MVTecCategoryDataModule(pl.LightningDataModule):
    """
    专门用于多类别分类的数据模块。
    只使用每个类别的训练集 good 图像，支持自动划分验证集。
    """

    def __init__(
        self,
        dataset_dir: str,
        categories: List[str],                  # 要使用的类别列表
        image_size: int = 224,
        channels_last: bool = True,
        normalize_imagenet: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_augment: bool = True,
        val_split: float = 0.1,                # 从训练集中划分验证集比例
    ):
        super().__init__()
        self.save_hyperparameters()
        self.categories = categories
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.channels_last = channels_last
        self.normalize_imagenet = normalize_imagenet
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_augment = train_augment
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset = None

    def _collect_samples_for_category(self, category_dir: Path, category: str) -> List[MVTecSample]:
        """收集该类别的所有训练 good 图像"""
        samples = []
        good_dir = category_dir / "train" / "good"
        if not good_dir.exists():
            return samples
        for path in sorted(good_dir.rglob("*")):
            if path.is_file() and _is_image_file(path):
                samples.append(MVTecSample(image_path=path, category=category))
        return samples

    def _build_samples(self) -> List[MVTecSample]:
        root = Path(self.hparams.dataset_dir)
        all_samples = []
        for category in self.categories:
            category_dir = root / category
            if not category_dir.exists():
                print(f"Warning: category {category} not found at {category_dir}")
                continue
            samples = self._collect_samples_for_category(category_dir, category)
            all_samples.extend(samples)
        return all_samples

    def setup(self, stage: Optional[str] = None):
        all_samples = self._build_samples()
        # 随机打乱
        random.shuffle(all_samples)
        val_size = int(len(all_samples) * self.val_split)
        train_size = len(all_samples) - val_size

        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:]

        self.train_dataset = MVTecCategoryDataset(
            train_samples,
            image_size=self.image_size,
            channels_last=self.channels_last,
            normalize_imagenet=self.normalize_imagenet,
            augment=self.train_augment,
        )
        self.val_dataset = MVTecCategoryDataset(
            val_samples,
            image_size=self.image_size,
            channels_last=self.channels_last,
            normalize_imagenet=self.normalize_imagenet,
            augment=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )