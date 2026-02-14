from __future__ import annotations

import os
import math
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image

from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel

from src.data import DataConfig, build_datasets
from src.utils import seed_everything, ensure_dir


@dataclass
class TrainConfig:
    batch_size: int
    grad_accum_steps: int
    lr: float
    weight_decay: float
    epochs: int
    max_train_steps: Optional[int]
    grad_clip_norm: float
    mixed_precision: str
    save_every_steps: int
    log_every_steps: int


def _get_eval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    e = cfg.get("eval", {}) or {}
    return {
        "val_every_steps": int(e.get("val_every_steps", 500)),
        "max_val_batches": int(e.get("max_val_batches", 50)),
        "sample_every_steps": int(e.get("sample_every_steps", 1000)),
        "sample_steps": int(e.get("sample_steps", 50)),
        "sample_n_per_class": int(e.get("sample_n_per_class", 16)),
    }


def _build_unet(cfg: Dict[str, Any], image_size: int, num_classes: int) -> UNet2DModel:
    m = cfg["model"]
    base = int(m["base_channels"])
    channel_mults = m["channel_mults"]
    block_out = tuple(base * int(x) for x in channel_mults)
    layers = int(m["layers_per_block"])
    dropout = float(m.get("dropout", 0.0))
    add_attn = bool(m.get("add_attention", True))

    n_blocks = len(block_out)
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
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=layers,
        block_out_channels=block_out,
        down_block_types=down_types,
        up_block_types=up_types,
        dropout=dropout,
        class_embed_type="timestep",
        num_class_embeds=num_classes,
    )
    return unet


@torch.no_grad()
def _ddim_sample_grid(
    accelerator: Accelerator,
    unet: UNet2DModel,
    cfg: Dict[str, Any],
    image_size: int,
    out_path: str,
    steps: int,
    n_per_class: int,
) -> None:
    device = accelerator.device

    ddim = DDIMScheduler(
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        prediction_type=str(cfg["diffusion"]["prediction_type"]),
        clip_sample=True,
    )
    ddim.set_timesteps(int(steps), device=device)

    all_imgs = []
    for cls in [0, 1]:
        b = int(n_per_class)
        x = torch.randn((b, 3, image_size, image_size), device=device)
        labels = torch.full((b,), cls, dtype=torch.long, device=device)

        for t in ddim.timesteps:
            noise_pred = unet(x, t, class_labels=labels).sample
            x = ddim.step(noise_pred, t, x).prev_sample

        imgs = ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)
        all_imgs.append(imgs)

    grid = make_grid(torch.cat(all_imgs, dim=0), nrow=int(n_per_class)).clamp(0, 1)
    grid_img = (grid * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(grid_img).save(out_path)


@torch.no_grad()
def _validate_loss(
    accelerator: Accelerator,
    unet: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    val_loader: DataLoader,
    max_batches: int,
) -> float:
    unet.eval()
    total = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        x = batch["pixel_values"]
        y = batch["class_labels"].long()
        b = x.shape[0]

        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (b,), device=x.device
        ).long()
        noisy = noise_scheduler.add_noise(x, noise, timesteps)

        pred = unet(noisy, timesteps, class_labels=y).sample
        loss = F.mse_loss(pred, noise, reduction="mean")

        loss_g = accelerator.gather_for_metrics(loss.detach())
        total += loss_g.mean().item()
        count += 1

    unet.train()
    return total / max(count, 1)


def main(config_path: str, resume_path: Optional[str] = None) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    eval_cfg = _get_eval_cfg(cfg)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    data_cfg = DataConfig(**cfg["data"])
    train_cfg = TrainConfig(**cfg["train"])

    out_dir = cfg["output"]["out_dir"]
    samples_dir = cfg["output"]["save_images_dir"]
    ensure_dir(out_dir)
    ensure_dir(samples_dir)

    accelerator = Accelerator(
        mixed_precision=None if train_cfg.mixed_precision == "no" else train_cfg.mixed_precision,
        gradient_accumulation_steps=int(train_cfg.grad_accum_steps),
    )

    # data
    train_ds, val_ds, stats = build_datasets(data_cfg, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(data_cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(data_cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # model
    unet = _build_unet(cfg, image_size=int(data_cfg.image_size), num_classes=int(data_cfg.num_classes))

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        prediction_type=str(cfg["diffusion"]["prediction_type"]),
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )

    # EMA
    use_ema = bool(cfg.get("ema", {}).get("use_ema", True))
    ema_decay = float(cfg.get("ema", {}).get("decay", 0.9999))
    ema: Optional[EMAModel] = None

    # prepare
    unet, optimizer, train_loader, val_loader = accelerator.prepare(unet, optimizer, train_loader, val_loader)

    if use_ema and accelerator.is_main_process:
        ema = EMAModel(accelerator.unwrap_model(unet).parameters(), decay=ema_decay)

    steps_per_epoch = len(train_loader)
    total_steps = int(train_cfg.max_train_steps) if train_cfg.max_train_steps else int(train_cfg.epochs) * steps_per_epoch
    epochs = int(train_cfg.epochs) if train_cfg.max_train_steps is None else math.ceil(total_steps / steps_per_epoch)

    start_step = 0
    best_val = float("inf")

    if resume_path is None:
        candidate = os.path.join(out_dir, "ckpt_last.pt")
        if os.path.isfile(candidate):
            resume_path = candidate

    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        accelerator.unwrap_model(unet).load_state_dict(ckpt["unet"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("val_loss", best_val))

        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])

        if accelerator.is_main_process:
            accelerator.print(f"Resumed from {resume_path} at step={start_step}, best_val={best_val}")

    if accelerator.is_main_process:
        accelerator.print("=== DATA STATS ===")
        accelerator.print(stats)
        accelerator.print(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={epochs}")

    global_step = start_step
    unet.train()

    def save_checkpoint(tag: str, val_loss: Optional[float] = None, also_save_last: bool = True):
        if not accelerator.is_main_process:
            return
        ckpt_path = os.path.join(out_dir, f"{tag}.pt")
        payload = {
            "step": global_step,
            "unet": accelerator.unwrap_model(unet).state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "config_path": config_path,
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()
        torch.save(payload, ckpt_path)
        accelerator.print(f"saved: {ckpt_path}")

        if also_save_last:
            last_path = os.path.join(out_dir, "ckpt_last.pt")
            torch.save(payload, last_path)

    for epoch in range(epochs):
        for batch in train_loader:
            with accelerator.accumulate(unet):
                x = batch["pixel_values"]          # [-1,1]
                y = batch["class_labels"].long()   # 0/1
                b = x.shape[0]

                noise = torch.randn_like(x)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (b,), device=x.device
                ).long()

                noisy = noise_scheduler.add_noise(x, noise, timesteps)
                pred = unet(noisy, timesteps, class_labels=y).sample
                loss = F.mse_loss(pred, noise, reduction="mean")

                accelerator.backward(loss)

                if float(train_cfg.grad_clip_norm) and float(train_cfg.grad_clip_norm) > 0:
                    accelerator.clip_grad_norm_(unet.parameters(), float(train_cfg.grad_clip_norm))

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.step(accelerator.unwrap_model(unet).parameters())

            global_step += 1

            if accelerator.is_main_process and (global_step % int(train_cfg.log_every_steps) == 0):
                accelerator.print(f"step {global_step}/{total_steps} loss={loss.item():.4f}")

            # validation
            if global_step % int(eval_cfg["val_every_steps"]) == 0:
                val_loss = _validate_loss(
                    accelerator, unet, noise_scheduler, val_loader, int(eval_cfg["max_val_batches"])
                )
                if accelerator.is_main_process:
                    accelerator.print(f"[val] step {global_step} val_loss={val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint("ckpt_best", val_loss=val_loss, also_save_last=False)

            # sampling (EMA)
            if global_step % int(eval_cfg["sample_every_steps"]) == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(unet)
                backup = None

                if ema is not None:
                    backup = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
                    ema.copy_to(unwrapped.parameters())

                out_path = os.path.join(samples_dir, f"sample_step_{global_step:07d}.png")
                _ddim_sample_grid(
                    accelerator,
                    unwrapped,
                    cfg,
                    image_size=int(data_cfg.image_size),
                    out_path=out_path,
                    steps=int(eval_cfg["sample_steps"]),
                    n_per_class=int(eval_cfg["sample_n_per_class"]),
                )
                accelerator.print(f"sampled: {out_path}")

                if ema is not None and backup is not None:
                    unwrapped.load_state_dict(backup)

            # checkpoint
            if global_step % int(train_cfg.save_every_steps) == 0:
                save_checkpoint(f"ckpt_{global_step:07d}")

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
            break

    if accelerator.is_main_process:
        save_checkpoint("ckpt_final", val_loss=best_val, also_save_last=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="Putanja do checkpointa (.pt). Ako nije dano, traži out_dir/ckpt_last.pt")
    args = p.parse_args()
    main(args.config, resume_path=args.resume)
