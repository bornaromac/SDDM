from __future__ import annotations
from pathlib import Path
import shutil
import random
import argparse
import pandas as pd

def find_image(image_id: str, image_dirs: list[Path]) -> Path | None:
    exts = [".jpg", ".jpeg", ".png"]
    for d in image_dirs:
        for ext in exts:
            p = d / f"{image_id}{ext}"
            if p.exists():
                return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", type=str, required=True)
    ap.add_argument("--image_dirs", type=str, nargs="+", required=True,
                    help="Jedan ili više direktorija gdje su slike (npr. HAM10000_images_part_1 HAM10000_images_part_2)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--malignant_dx", type=str, nargs="+", default=["mel","bcc","akiec"])
    ap.add_argument("--benign_dx", type=str, nargs="+", default=["nv","bkl","df","vasc"])

    ap.add_argument("--n_per_class", type=int, default=100)
    args = ap.parse_args()

    random.seed(args.seed)

    meta = pd.read_csv(args.metadata_csv)
    if "image_id" not in meta.columns or "dx" not in meta.columns:
        raise ValueError("CSV mora imati stupce: image_id, dx")

    malignant = set(args.malignant_dx)
    benign = set(args.benign_dx)

    def map_class(dx: str) -> int | None:
        if dx in malignant:
            return 1
        if dx in benign:
            return 0
        return None

    meta["class"] = meta["dx"].map(map_class)
    meta = meta.dropna(subset=["class"]).copy()
    meta["class"] = meta["class"].astype(int)

    out = Path(args.out_dir)
    out0 = out / "class_0"
    out1 = out / "class_1"
    out0.mkdir(parents=True, exist_ok=True)
    out1.mkdir(parents=True, exist_ok=True)

    image_dirs = [Path(p) for p in args.image_dirs]

    for cls, out_cls in [(0, out0), (1, out1)]:
        subset = meta[meta["class"] == cls]["image_id"].tolist()
        random.shuffle(subset)
        picked = subset[: args.n_per_class]

        copied = 0
        for image_id in picked:
            src = find_image(image_id, image_dirs)
            if src is None:
                continue
            dst = out_cls / src.name
            shutil.copy2(src, dst)
            copied += 1

        if copied < args.n_per_class:
            raise RuntimeError(f"Za class_{cls} kopirano {copied}/{args.n_per_class}. "
                               f"Provjeri putanje image_dirs i ekstenzije datoteka.")

    print(f"OK: kopirano {args.n_per_class} slika u class_0 i {args.n_per_class} u class_1 -> {out}")

if __name__ == "__main__":
    main()
