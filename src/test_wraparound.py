import os
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms


def main(image_path: str, out_dir: str, image_size: int = 64):
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    img.save(os.path.join(out_dir, "original.png"))

    tfm = transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                         # [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3),        # -> [-1,1]
    ])

    x = tfm(img)  # tensor [3,H,W] u [-1,1]

    x01 = ((x + 1) / 2).clamp(0, 1)

    out = (x01 * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(out).save(os.path.join(out_dir, "roundtrip_new.png"))

    out_old = (x01 * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(out_old).save(os.path.join(out_dir, "roundtrip_old.png"))

    print("Tensor stats after denorm [0,1]:",
          "min=", float(x01.min()), "max=", float(x01.max()))

    print("Saved to:", out_dir)
    print(" - original.png")
    print(" - roundtrip_new.png (correct save)")
    print(" - roundtrip_old.png (byte() save)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help=r"C:\Users\Borna\Desktop\SDDM\data\extracted\HAM10000_images_part_1\ISIC_0024306.jpg")
    p.add_argument("--out_dir", default="debug_roundtrip", help="Output folder")
    p.add_argument("--image_size", type=int, default=64)
    args = p.parse_args()

    main(args.image, args.out_dir, args.image_size)
