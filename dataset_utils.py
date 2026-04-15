from __future__ import annotations

import os
from typing import Literal

import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

DatasetName = Literal["cifar10", "imagenet"]

DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


def resolve_split_dir(data_root: str, split: str, explicit_dir: str = "") -> str:
    if explicit_dir:
        return explicit_dir
    return os.path.join(data_root, split)


def is_imagefolder_style_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for entry in os.scandir(path):
        if entry.is_dir():
            return True
    return False


def find_recursive_images(root: str) -> list[str]:
    image_paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if has_file_allowed_extension(filename, IMG_EXTENSIONS):
                image_paths.append(os.path.join(dirpath, filename))
    image_paths.sort()
    return image_paths


def resolve_imagenet_kaggle_split_dir(data_root: str, split: str) -> str:
    candidates = [
        os.path.join(data_root, "ILSVRC", "Data", "CLS-LOC", split),
        os.path.join(data_root, "ILSVRC", "Data", split),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return ""


def build_eval_transform(*, dataset: DatasetName, image_size: int):
    if dataset == "imagenet":
        resize_size = max(image_size, round(image_size / 0.875))
        return Compose([
            Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ])

    return Compose([
        ToTensor(),
        Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])


def build_train_transform(*, dataset: DatasetName, image_size: int):
    if dataset == "imagenet":
        return Compose([
            RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ])

    return Compose([
        ToTensor(),
        Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])


def build_supervised_dataset(
    *,
    dataset: DatasetName,
    data_root: str,
    split: str,
    transform,
    train_dir: str = "",
    val_dir: str = "",
    download: bool = False,
):
    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=data_root,
            train=(split == "train"),
            transform=transform,
            download=download,
        )

    explicit_split_dir = train_dir if split == "train" else val_dir
    split_dir = resolve_split_dir(data_root, split, explicit_split_dir)

    if explicit_split_dir:
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Explicit ImageNet {split!r} directory not found: {split_dir}."
            )
        if is_imagefolder_style_dir(split_dir):
            return torchvision.datasets.ImageFolder(split_dir, transform=transform)
        return RecursiveImageDataset(split_dir, transform=transform)

    if os.path.isdir(split_dir):
        if is_imagefolder_style_dir(split_dir):
            return torchvision.datasets.ImageFolder(split_dir, transform=transform)
        return RecursiveImageDataset(split_dir, transform=transform)

    kaggle_split_dir = resolve_imagenet_kaggle_split_dir(data_root, split)
    if kaggle_split_dir:
        if is_imagefolder_style_dir(kaggle_split_dir):
            return torchvision.datasets.ImageFolder(kaggle_split_dir, transform=transform)
        return RecursiveImageDataset(kaggle_split_dir, transform=transform)

    try:
        return torchvision.datasets.ImageNet(root=data_root, split=split, transform=transform)
    except Exception as exc:
        raise FileNotFoundError(
            f"ImageNet {split!r} is not prepared under {data_root}. "
            f"Checked {split_dir}/ and Kaggle-style paths under {data_root}/ILSVRC/Data/CLS-LOC/{split}. "
            "You can either prepare train/val directories, download the Kaggle competition data layout, "
            "or provide the official ImageNet archives in data_root."
        ) from exc


class RecursiveImageDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.samples = find_recursive_images(root)
        if not self.samples:
            raise FileNotFoundError(f"No image files found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


class SelfSupervisedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        return image, 0
