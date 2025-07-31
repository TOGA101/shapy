import argparse
import os
import os.path as osp
from typing import Sequence

from PIL import Image
from loguru import logger
import torch
from torchvision import transforms
from omegaconf import OmegaConf

from human_shape.config.defaults import conf as default_conf
from human_shape.models.build import build_model
from human_shape.utils.checkpointer import Checkpointer
from human_shape.data.structures import AbstractStructure, StructureList


@torch.no_grad()
def load_model(cfg_path: str, device: torch.device, *, output_folder: str) -> torch.nn.Module:
    cfg = default_conf.copy()
    cfg.merge_with(OmegaConf.load(cfg_path))
    cfg.is_training = False
    cfg.use_cuda = device.type == "cuda"
    cfg.output_folder = output_folder

    model_dict = build_model(cfg)
    model = model_dict["network"].to(device)

    checkpoint_folder = osp.join(osp.expandvars(cfg.output_folder), cfg.checkpoint_folder)
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder, pretrained=cfg.pretrained)
    checkpointer.load_checkpoint()
    model.eval()
    return model, cfg


def preprocess_image(img_path: str, mean: Sequence[float], std: Sequence[float], crop_size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)


@torch.no_grad()
def infer_betas(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    target = AbstractStructure()
    target.add_field("gender", "n")
    targets = StructureList([target])
    output = model(image.to(device), targets=targets, compute_losses=False)
    stage_key = output.get("stage_keys", ["stage_02"])[-1]
    betas = output[stage_key]["betas"]
    return betas.squeeze(0).cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAPY single image inference")
    parser.add_argument("--image-path", required=True, help="Path to an aligned full-body image")
    parser.add_argument("--exp-cfg", default="configs/b2a_expose_hrnet_demo.yaml", help="Path to experiment config")
    parser.add_argument(
        "--output-folder",
        default="../data/trained_models/shapy/SHAPY_A",
        help="Folder containing trained checkpoints",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, cfg = load_model(args.exp_cfg, device, output_folder=args.output_folder)

    part_key = cfg.get("part_key", "pose")
    transf_cfg = cfg.datasets.get(part_key, {}).get("transforms", {})
    crop_size = transf_cfg.get("crop_size", 256)
    mean = transf_cfg.get("mean", (0.485, 0.456, 0.406))
    std = transf_cfg.get("std", (0.229, 0.224, 0.225))

    image = preprocess_image(args.image_path, mean, std, crop_size)
    betas = infer_betas(model, image, device)
    logger.info(f"Betas: {betas.tolist()}")
