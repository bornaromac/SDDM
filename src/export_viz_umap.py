from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import umap
from tqdm import tqdm


TARGET_SIZE = 64          
THUMB_SIZE = 64          
UMAP_N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.15
UMAP_METRIC = "euclidean"
TOPK_NN = 5


@dataclass
class Item:
    id: str
    type: str       # "real" | "synthetic"
    label: int      # 0 | 1
    img: np.ndarray # float32 [64,64,3] in [0,1]
    thumb_rel: str


def _read_zip_images(zip_path: Path) -> List[Tuple[str, Image.Image]]:
    """Return (filename, PIL_image) for image files found in zip."""
    out: List[Tuple[str, Image.Image]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lname = name.lower()
            if lname.endswith((".png", ".jpg", ".jpeg", ".webp")) and not lname.endswith("/"):
                with zf.open(name) as f:
                    data = f.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                out.append((name, img))

    if not out:
        raise RuntimeError(f"No images found in {zip_path}")
    return out


def _infer_label(path_like_name: str) -> int:
    """
    Heuristika:
    - ako putanja sadrži 'class_0' -> 0
    - ako sadrži 'class_1' -> 1
    """
    s = path_like_name.replace("\\", "/").lower()
    if "class_0" in s:
        return 0
    if "class_1" in s:
        return 1
    raise ValueError(f"Ne mogu odrediti labelu iz naziva: {path_like_name}")


def _normalize_rgb_64(img: Image.Image) -> Image.Image:
    """Force RGB and resize to 64x64 so all arrays have the same shape."""
    img = img.convert("RGB")
    if img.size != (TARGET_SIZE, TARGET_SIZE):
        img = img.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.Resampling.BILINEAR)
    return img


def _img_to_array(img: Image.Image) -> np.ndarray:
    """Convert normalized RGB image to float32 array in [0,1]."""
    img = _normalize_rgb_64(img)
    arr = np.asarray(img).astype(np.float32) / 255.0
    # safety: enforce exact shape
    if arr.shape != (TARGET_SIZE, TARGET_SIZE, 3):
        raise ValueError(f"Unexpected array shape {arr.shape}, expected {(TARGET_SIZE, TARGET_SIZE, 3)}")
    return arr


def _make_thumb(img: Image.Image, size: int) -> Image.Image:
    img = _normalize_rgb_64(img)
    return img.resize((size, size), resample=Image.Resampling.NEAREST)


def _sharpness_var_laplacian(rgb01: np.ndarray) -> float:
    gray = (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    g = np.pad(gray, 1, mode="edge")
    lap = (
        k[0, 0] * g[:-2, :-2] + k[0, 1] * g[:-2, 1:-1] + k[0, 2] * g[:-2, 2:] +
        k[1, 0] * g[1:-1, :-2] + k[1, 1] * g[1:-1, 1:-1] + k[1, 2] * g[1:-1, 2:] +
        k[2, 0] * g[2:, :-2] + k[2, 1] * g[2:, 1:-1] + k[2, 2] * g[2:, 2:]
    )
    return float(lap.var())


def main(
    generated_zip: str,
    real_zip: str,
    out_dir: str = "viz_export_umap",
    seed: int = 42,
):
    generated_zip = Path(generated_zip)
    real_zip = Path(real_zip)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thumbs_dir = out / "thumbs"
    (thumbs_dir / "real").mkdir(parents=True, exist_ok=True)
    (thumbs_dir / "synthetic").mkdir(parents=True, exist_ok=True)

    real_imgs = _read_zip_images(real_zip)
    syn_imgs = _read_zip_images(generated_zip)

    items: List[Item] = []

    # real
    for i, (name, pil) in enumerate(tqdm(real_imgs, desc="Real")):
        label = _infer_label(name)
        img_arr = _img_to_array(pil)
        item_id = f"real_{i:04d}"
        thumb_rel = f"thumbs/real/{item_id}.png"
        thumb_path = out / thumb_rel
        _make_thumb(pil, THUMB_SIZE).save(thumb_path, format="PNG")
        items.append(Item(id=item_id, type="real", label=label, img=img_arr, thumb_rel=thumb_rel))

    # synthetic
    for i, (name, pil) in enumerate(tqdm(syn_imgs, desc="Synthetic")):
        label = _infer_label(name)
        img_arr = _img_to_array(pil)
        item_id = f"syn_{i:04d}"
        thumb_rel = f"thumbs/synthetic/{item_id}.png"
        thumb_path = out / thumb_rel
        _make_thumb(pil, THUMB_SIZE).save(thumb_path, format="PNG")
        items.append(Item(id=item_id, type="synthetic", label=label, img=img_arr, thumb_rel=thumb_rel))

    X = np.stack([it.img.reshape(-1) for it in items], axis=0)  # (N, 64*64*3)

    # --- UMAP 2D ---
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=seed,
    )
    emb2 = reducer.fit_transform(X)  # (N,2)

    # --- Nearest neighbor: synthetic -> real (topK) ---
    real_idx = [i for i, it in enumerate(items) if it.type == "real"]
    syn_idx = [i for i, it in enumerate(items) if it.type == "synthetic"]

    X_real = X[real_idx]
    nn = NearestNeighbors(n_neighbors=min(TOPK_NN, len(real_idx)), metric="euclidean")
    nn.fit(X_real)

    nn_map: Dict[str, List[str]] = {}
    nn_dist_map: Dict[str, float] = {}

    distances, indices = nn.kneighbors(X[syn_idx], return_distance=True)
    for local_i, global_i in enumerate(syn_idx):
        syn_id = items[global_i].id
        real_neighbors_global = [real_idx[j] for j in indices[local_i].tolist()]
        nn_ids = [items[g].id for g in real_neighbors_global]
        nn_map[syn_id] = nn_ids
        nn_dist_map[syn_id] = float(distances[local_i][0])

    sharp = []
    rgb_mean = []
    for it in items:
        sharp.append(_sharpness_var_laplacian(it.img))
        rgb_mean.append(it.img.mean(axis=(0, 1)).tolist())

    sharp_real = [s for s, it in zip(sharp, items) if it.type == "real"]
    sharp_syn = [s for s, it in zip(sharp, items) if it.type == "synthetic"]
    nn_dists = list(nn_dist_map.values())

    metrics = {
        "counts": {"real": int(len(real_idx)), "synthetic": int(len(syn_idx))},
        "resize": {"target_size": TARGET_SIZE, "thumb_size": THUMB_SIZE},
        "umap": {
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "seed": seed,
        },
        "sharpness": {
            "real_mean": float(np.mean(sharp_real)),
            "real_std": float(np.std(sharp_real)),
            "synthetic_mean": float(np.mean(sharp_syn)),
            "synthetic_std": float(np.std(sharp_syn)),
        },
        "nearest_real_distance": {
            "mean": float(np.mean(nn_dists)) if nn_dists else None,
            "std": float(np.std(nn_dists)) if nn_dists else None,
            "min": float(np.min(nn_dists)) if nn_dists else None,
            "max": float(np.max(nn_dists)) if nn_dists else None,
        },
    }

    points = []
    for i, it in enumerate(items):
        p = {
            "id": it.id,
            "type": it.type,
            "label": int(it.label),
            "x": float(emb2[i, 0]),
            "y": float(emb2[i, 1]),
            "thumb": it.thumb_rel,
            "sharpness": float(sharp[i]),
            "rgb_mean": rgb_mean[i],
        }
        if it.type == "synthetic":
            p["nn_real_id"] = nn_map[it.id][0] if nn_map.get(it.id) else None
            p["nn_real_dist"] = nn_dist_map.get(it.id)
        points.append(p)

    (out / "points.json").write_text(json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "nn_map.json").write_text(json.dumps(nn_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK -> {out.resolve()}")
    print(f"- points.json: {len(points)} items")
    print(f"- nn_map.json: {len(nn_map)} synthetic entries")
    print(f"- thumbs: {thumbs_dir.resolve()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_zip", required=True)
    ap.add_argument("--real_zip", required=True)
    ap.add_argument("--out_dir", default="viz_export_umap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.generated_zip, args.real_zip, args.out_dir, args.seed)
