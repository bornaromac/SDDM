from __future__ import annotations

import os
import re
import yaml
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler

from src.data import DataConfig
from src.utils import ensure_dir, seed_everything


def ckpt_tag(ckpt_path: str) -> str:
    base = os.path.basename(ckpt_path).lower()
    m = re.search(r"ckpt_(\d+)", base)
    if m:
        return m.group(1).zfill(7)  
    if "best" in base:
        return "best"
    if "final" in base:
        return "final"
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return stem[:32]


def build_unet(cfg: dict, data_cfg: DataConfig, device: torch.device) -> UNet2DModel:
    m = cfg["model"]
    base = int(m["base_channels"])
    channel_mults = m["channel_mults"]
    block_out_channels = tuple(base * int(x) for x in channel_mults)
    layers_per_block = int(m["layers_per_block"])
    dropout = float(m.get("dropout", 0.0))
    add_attn = bool(m.get("add_attention", True))
    num_classes = int(cfg["data"]["num_classes"])

    n_blocks = len(block_out_channels)
    if add_attn:
        mid = n_blocks // 2
        down_types = ["DownBlock2D"] * n_blocks
        up_types = ["UpBlock2D"] * n_blocks
        down_types[mid] = "AttnDownBlock2D"
        up_types[mid] = "AttnUpBlock2D"
    else:
        down_types = ["DownBlock2D"] * n_blocks
        up_types = ["UpBlock2D"] * n_blocks

    unet = UNet2DModel(
        sample_size=int(data_cfg.image_size),
        in_channels=3,
        out_channels=3,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_types,
        up_block_types=up_types,
        dropout=dropout,
        class_embed_type="timestep",
        num_class_embeds=num_classes,
    ).to(device)

    return unet


def build_scheduler(cfg: dict, sampler: str):
    sampler = sampler.lower().strip()
    common = dict(
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        prediction_type=str(cfg["diffusion"]["prediction_type"]),
        clip_sample=True,
    )

    if sampler == "ddim":
        return DDIMScheduler(**common)
    if sampler == "ddpm":
        return DDPMScheduler(**common)
    raise ValueError("sampler mora biti 'ddim' ili 'ddpm'")


@torch.no_grad()
def main(
    config_path: str,
    ckpt_path: str,
    sampler: str = "ddim",
    steps: int = 50,
    n_per_class: int = 8,
    save_individual: bool = False,
    seed: int | None = None,
    device_str: str | None = None,
    fp16: bool = False,
    out_dir_override: str | None = None,
):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    if seed is None:
        seed = int(cfg.get("seed", 42))
    seed_everything(int(seed))

    data_cfg = DataConfig(**cfg["data"])

    if out_dir_override is not None:
        out_root = out_dir_override
    else:
        out_root = cfg.get("output", {}).get("save_images_dir")
        if out_root is None:
            base_out = cfg.get("output", {}).get("out_dir", "runs")
            out_root = os.path.join(base_out, "samples")

    tag = ckpt_tag(ckpt_path)
    run_dir = os.path.join(out_root, f"ckpt_{tag}", f"{sampler}_steps{int(steps)}_n{int(n_per_class)}")
    ensure_dir(run_dir)

    # device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # UNet
    unet = build_unet(cfg, data_cfg, device)

    state = torch.load(ckpt_path, map_location="cpu")
    if "unet" not in state:
        raise KeyError(f"Checkpoint nema 'unet'. Ključevi: {list(state.keys())}")

    unet.load_state_dict(state["unet"], strict=True)
    unet.eval()

    # Scheduler
    sched = build_scheduler(cfg, sampler)
    sched.set_timesteps(int(steps), device=device)

    image_size = int(data_cfg.image_size)
    all_imgs = []

    use_amp = bool(fp16 and device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(device_type="cpu", enabled=False)

    for cls in [0, 1]:
        b = int(n_per_class)
        x = torch.randn((b, 3, image_size, image_size), device=device)
        labels = torch.full((b,), cls, dtype=torch.long, device=device)

        with autocast_ctx:
            for t in sched.timesteps:
                noise_pred = unet(x, t, class_labels=labels).sample
                x = sched.step(noise_pred, t, x).prev_sample

        imgs = ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)
        all_imgs.append(imgs)

        if save_individual:
            cls_dir = os.path.join(run_dir, f"cls{cls}")
            ensure_dir(cls_dir)
            for i in range(imgs.shape[0]):
                save_image(imgs[i], os.path.join(cls_dir, f"{tag}_{sampler}_s{steps}_cls{cls}_{i:03d}.png"))

    grid = make_grid(torch.cat(all_imgs, dim=0), nrow=int(n_per_class)).clamp(0, 1)
    grid_img = (grid * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    out_path = os.path.join(run_dir, f"grid_{tag}_{sampler}_steps{steps}_n{n_per_class}.png")
    Image.fromarray(grid_img).save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n_per_class", type=int, default=8)
    p.add_argument("--save_individual", action="store_true")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="npr. cuda, cpu, cuda:0")
    p.add_argument("--fp16", action="store_true", help="koristi autocast fp16 na CUDA")
    p.add_argument("--out_dir", type=str, default=None, help="override output.save_images_dir iz yaml-a")

    args = p.parse_args()

    main(
        args.config,
        args.ckpt,
        sampler=args.sampler,
        steps=args.steps,
        n_per_class=args.n_per_class,
        save_individual=args.save_individual,
        seed=args.seed,
        device_str=args.device,
        fp16=args.fp16,
        out_dir_override=args.out_dir,
    )
