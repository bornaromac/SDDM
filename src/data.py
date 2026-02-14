from __future__ import annotations

import os
import glob
import zipfile
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class DataConfig:
    root: str
    archive_zip: str
    extracted_dir: str
    image_dir_glob: List[str]
    metadata_glob: List[str]
    image_size: int
    num_workers: int
    val_ratio: float
    malignant_dx: List[str]
    benign_dx: List[str]
    num_classes: int


def _has_any_images(root_dir: str) -> bool:
    if not os.path.isdir(root_dir):
        return False
    exts = (".jpg", ".jpeg", ".png")
    for dp, _, fn in os.walk(root_dir):
        for f in fn:
            if f.lower().endswith(exts):
                return True
    return False


def maybe_extract_archive(archive_zip: str, extracted_dir: str) -> None:
    """
    Sigurna ekstrakcija:
    - ako extracted_dir ne postoji -> extract
    - ako postoji ali NEMA ni jednu sliku -> extract (ponovno)
    - ako postoji i ima slike -> ne diraj
    """
    if not os.path.isfile(archive_zip):
        raise FileNotFoundError(f"Ne nalazim archive zip: {archive_zip}. Provjeri putanju u configu.")

    os.makedirs(extracted_dir, exist_ok=True)

    if _has_any_images(extracted_dir):
        return

    with zipfile.ZipFile(archive_zip, "r") as zf:
        zf.extractall(extracted_dir)


def find_first_existing(globs_list: List[str]) -> Optional[str]:
    for g in globs_list:
        matches = glob.glob(g, recursive=True)
        if matches:
            matches = sorted(matches, key=lambda x: len(x))
            return matches[0]
    return None


def find_all_dirs(globs_list: List[str]) -> List[str]:
    dirs: List[str] = []
    for g in globs_list:
        for m in glob.glob(g, recursive=True):
            if os.path.isdir(m):
                dirs.append(m)
    seen = set()
    out = []
    for d in sorted(dirs, key=lambda x: (len(x), x)):
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(image_size + 16),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 16),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])


def load_metadata(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV nema stupce: {missing}. Nađeno: {list(df.columns)}")
    return df


def map_dx_to_binary_label(dx: str, malignant_dx: List[str], benign_dx: List[str]) -> Optional[int]:
    dx = str(dx).strip().lower()
    if dx in malignant_dx:
        return 1
    if dx in benign_dx:
        return 0
    return None


def build_image_index_from_dirs(image_dirs: List[str]) -> Dict[str, str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    idx: Dict[str, str] = {}
    for d in image_dirs:
        for e in exts:
            for p in glob.glob(os.path.join(d, e)):
                base = os.path.splitext(os.path.basename(p))[0]
                if base not in idx:
                    idx[base] = p
    return idx


def dir_has_images(d: str) -> bool:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if glob.glob(os.path.join(d, ext)):
            return True
    return False


def make_splits(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
    splits_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(splits_dir, exist_ok=True)
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")

    def regenerate_and_save() -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(df) == 0:
            raise ValueError("Ne mogu napraviti split: df je prazan (0 uzoraka).")

        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        groups = df["lesion_id"].values
        idx = np.arange(len(df))
        train_idx, val_idx = next(gss.split(idx, groups=groups))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        return train_df, val_df

    if os.path.isfile(train_csv) and os.path.isfile(val_csv):
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        if "image_id" in train_df.columns and "image_id" in val_df.columns:
            current_ids = set(df["image_id"].astype(str).tolist())
            train_ids = set(train_df["image_id"].astype(str).tolist())
            val_ids = set(val_df["image_id"].astype(str).tolist())

            ok = (
                len(train_ids) > 0 and len(val_ids) > 0 and
                train_ids.issubset(current_ids) and
                val_ids.issubset(current_ids)
            )
            if ok:
                return train_df, val_df

        return regenerate_and_save()

    return regenerate_and_save()


class HamBinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_index: Dict[str, str], tfm, num_classes: int):
        self.df = df
        self.image_index = image_index
        self.tfm = tfm
        self.num_classes = num_classes

        self.paths: List[str] = []
        self.labels: List[int] = []

        for _, r in df.iterrows():
            image_id = str(r["image_id"])
            if image_id not in image_index:
                continue
            self.paths.append(image_index[image_id])
            self.labels.append(int(r["label"]))

        if len(self.paths) == 0:
            raise RuntimeError(
                "Dataset je prazan nakon filtriranja. "
                "To obično znači: image_index je 0 ili splits cache ne odgovara datasetu."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        img = self.tfm(img)
        label = self.labels[i]
        return {"pixel_values": img, "class_labels": torch.tensor(label, dtype=torch.long)}


def build_datasets(cfg: DataConfig, seed: int):
    maybe_extract_archive(cfg.archive_zip, cfg.extracted_dir)

    metadata_path = find_first_existing(cfg.metadata_glob)
    if metadata_path is None:
        raise FileNotFoundError(f"Ne nalazim HAM10000_metadata.csv pod: {cfg.extracted_dir}")

    df = load_metadata(metadata_path)
    stats: Dict[str, Any] = {
        "metadata_rows_total": int(len(df)),
        "metadata_path": metadata_path,
    }

    df["label"] = df["dx"].apply(lambda x: map_dx_to_binary_label(x, cfg.malignant_dx, cfg.benign_dx))
    stats["rows_after_label_map"] = int(df["label"].notna().sum())
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    image_dirs = find_all_dirs(cfg.image_dir_glob)
    stats["image_dirs_found"] = image_dirs

    if not image_dirs:
        raise FileNotFoundError(
            f"Ne nalazim image direktorije po image_dir_glob. "
            f"Provjeri da extracted stvarno sadrži HAM10000_images_part_1/2. "
            f"extracted_dir='{cfg.extracted_dir}'"
        )

    part_dirs = [
        d for d in image_dirs
        if os.path.basename(d).lower() in ("ham10000_images_part_1", "ham10000_images_part_2")
    ]

    merged_dirs = [d for d in image_dirs if os.path.basename(d).lower() == "ham10000_images" and dir_has_images(d)]

    if len(merged_dirs) > 0:
        chosen_dirs = [merged_dirs[0]]
    elif len(part_dirs) > 0:
        chosen_dirs = part_dirs
    else:
        chosen_dirs = image_dirs

    stats["image_dirs_used_for_index"] = chosen_dirs

    image_index = build_image_index_from_dirs(chosen_dirs)
    stats["images_indexed"] = int(len(image_index))

    if stats["images_indexed"] == 0:
        raise RuntimeError(
            "image_index je 0 (nisam našao nijednu .jpg/.png sliku u odabranim direktorijima). "
            f"Dirs: {chosen_dirs}"
        )

    exists_mask = df["image_id"].astype(str).isin(image_index.keys())
    stats["rows_with_existing_files"] = int(exists_mask.sum())
    df = df[exists_mask].copy()

    if len(df) == 0:
        raise RuntimeError(
            "Nakon matchanja image_id -> file, df je prazan. "
            "To znači da metadata image_id ne odgovara nazivima fileova ili index nije napravljen kako treba. "
            f"images_indexed={stats['images_indexed']}"
        )

    splits_dir = os.path.join(cfg.root, "splits")
    train_df, val_df = make_splits(df, cfg.val_ratio, seed, splits_dir)

    train_tfm = build_transforms(cfg.image_size, train=True)
    val_tfm = build_transforms(cfg.image_size, train=False)

    train_ds = HamBinaryDataset(train_df, image_index, train_tfm, cfg.num_classes)
    val_ds = HamBinaryDataset(val_df, image_index, val_tfm, cfg.num_classes)

    stats["train_size"] = len(train_ds)
    stats["val_size"] = len(val_ds)

    return train_ds, val_ds, stats
