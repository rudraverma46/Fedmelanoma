"""Pixel-identical image cache and the per-hospital DataLoaders.

The source JPEGs are up to 6000x4000 px, so decoding them every epoch (about 10 images/s on
this machine) would make the run about 6x slower than the GPU. The cache stores, once, the
uint8 result of exactly the reference's per-image steps

    cv2.imread -> cv2.cvtColor(BGR2RGB) -> cv2.resize((224, 224))

and CachedMelanomaDataset applies the reference's own ToTensor + Normalize to it. Everything
else (labels, metadata vectors, row order, `df`) is inherited from the unmodified
MelanomaDataset. test_01 checks cached and uncached samples are equal tensors.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Tuple

import cv2
import numpy as np
from torch.utils.data import DataLoader

from client.dataset import MelanomaDataset

SPLITS = {"train": ("metadata_train.csv", "Train"), "test": ("metadata_test.csv", "Test")}


def split_paths(partitions_root: str, hospital: str, split: str) -> Tuple[str, str]:
    csv_name, image_sub = SPLITS[split]
    base = os.path.join(partitions_root, hospital)
    return os.path.join(base, csv_name), os.path.join(base, "images", image_sub)


def cache_paths(cache_dir: str, hospital: str, split: str) -> Tuple[str, str]:
    stem = os.path.join(cache_dir, f"{hospital}__{split}")
    return stem + ".npy", stem + ".json"


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_resized_rgb(image_path: str) -> np.ndarray:
    """client/dataset.py __getitem__ up to, not including, the ToTensor transform."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224))


def build_split_cache(partitions_root: str, cache_dir: str, hospital: str, split: str) -> None:
    csv_file, image_dir = split_paths(partitions_root, hospital, split)
    dataset = MelanomaDataset(csv_file, image_dir)
    npy, meta = cache_paths(cache_dir, hospital, split)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = npy + ".tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(dataset), 224, 224, 3))
    ids = [row["isic_id"] for row in dataset.rows]
    for i, isic_id in enumerate(ids):
        arr[i] = load_resized_rgb(os.path.join(image_dir, isic_id + ".jpg"))
    arr.flush()
    del arr
    os.replace(tmp, npy)
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"csv_sha256": _sha256(csv_file), "isic_ids": ids}, fh)


def cache_is_current(partitions_root: str, cache_dir: str, hospital: str, split: str) -> bool:
    npy, meta = cache_paths(cache_dir, hospital, split)
    if not (os.path.exists(npy) and os.path.exists(meta)):
        return False
    with open(meta, encoding="utf-8") as fh:
        return json.load(fh)["csv_sha256"] == _sha256(split_paths(partitions_root, hospital, split)[0])


class CachedMelanomaDataset(MelanomaDataset):
    """MelanomaDataset whose __getitem__ reads the resized uint8 image from the cache."""

    def __init__(self, csv_file: str, image_dir: str, npy_path: str, meta_path: str) -> None:
        super().__init__(csv_file, image_dir)
        with open(meta_path, encoding="utf-8") as fh:
            cached_ids = json.load(fh)["isic_ids"]
        if cached_ids != [row["isic_id"] for row in self.rows]:
            raise ValueError(f"cache row order does not match {csv_file}")
        self._images = np.load(npy_path, mmap_mode="r")

    def __getitem__(self, idx: int):
        image = self.transform(np.array(self._images[idx]))
        return image, self.metadata_cache[idx], self.labels[idx]


def build_cached_dataset(partitions_root: str, cache_dir: str, hospital: str, split: str) -> CachedMelanomaDataset:
    csv_file, image_dir = split_paths(partitions_root, hospital, split)
    npy, meta = cache_paths(cache_dir, hospital, split)
    return CachedMelanomaDataset(csv_file, image_dir, npy, meta)


def build_loader(dataset, batch_size: int, shuffle: bool, workers: int, pin_memory: bool) -> DataLoader:
    """Same batch size and shuffle flags as client/client.py build_hospital_dataloaders."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=pin_memory)
