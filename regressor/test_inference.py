from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import ColorJitter

# Import helper functions from the inference script in the same directory
from inference import (
    load_config,
    build_network,
    preprocess_image,
    infer_betas,
    DEFAULT_CHECKPOINT,
)


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    cfg = load_config(args.cfg)
    cfg.use_cuda = device.type == "cuda"

    model = build_network(cfg, device, args.checkpoint)

    img = Image.open(args.image).convert("RGB")
    aug = ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)
    img_aug = aug(img)

    crop_size = cfg.datasets.pose.transforms.crop_size
    mean = cfg.datasets.pose.transforms.mean
    std = cfg.datasets.pose.transforms.std

    img_t = preprocess_image(img, crop_size, mean, std)
    img_aug_t = preprocess_image(img_aug, crop_size, mean, std)

    betas = infer_betas(model, img_t, device)
    betas_aug = infer_betas(model, img_aug_t, device)

    print("Original betas:", betas.numpy())
    print("Augmented betas:", betas_aug.numpy())
    max_diff = (betas - betas_aug).abs().max().item()
    print("Max absolute difference:", max_diff)
    print("Close:", torch.allclose(betas, betas_aug, atol=args.tolerance))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(img_aug)
    ax[1].set_title("Augmented")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single-image inference")
    parser.add_argument("--cfg", type=Path, required=True, help="Experiment YAML")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=("Checkpoint directory. Defaults to SHAPY_A inside data/trained_models"),
    )
    parser.add_argument("--image", type=Path, required=True, help="Aligned image")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Tolerance for betas difference")
    main(parser.parse_args())
