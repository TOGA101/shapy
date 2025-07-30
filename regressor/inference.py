import argparse
from pathlib import Path

import torch
import cv2
from torchvision.transforms import functional as F

from omegaconf import OmegaConf

from human_shape.config.defaults import conf as default_conf
from human_shape.models.build import build_model
from human_shape.utils import Checkpointer
from human_shape.data.structures import BoundingBox, StructureList


def load_cfg(cfg_path: Path, output_folder: Path) -> "OmegaConf":
    cfg = default_conf.copy()
    cfg.merge_with(OmegaConf.load(str(cfg_path)))
    cfg.output_folder = str(output_folder)
    cfg.is_training = False
    return cfg


def build_network(cfg, device: torch.device):
    model_dict = build_model(cfg)
    model = model_dict["network"].to(device)
    ckpt_dir = Path(cfg.output_folder) / cfg.checkpoint_folder
    checkpointer = Checkpointer(model, save_dir=str(ckpt_dir),
                                pretrained=cfg.pretrained)
    checkpointer.load_checkpoint()
    model.eval()
    return model


def preprocess_image(img_path: Path, crop_size: int, mean, std, device: torch.device):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    tensor = F.to_tensor(img)
    tensor = F.normalize(tensor, mean=list(mean), std=list(std))
    tensor = tensor.unsqueeze(0).to(device)

    target = BoundingBox(torch.tensor([0, 0, crop_size, crop_size], dtype=torch.float32),
                         size=(crop_size, crop_size, 3))
    target.add_field('fname', img_path.name)
    target.add_field('gender', 'neutral')
    target.to_tensor()
    target = target.to(device)
    return tensor, [target]


def run_inference(model, image: torch.Tensor, targets: StructureList):
    with torch.no_grad():
        out = model(image, targets, compute_losses=False)
    last_stage = out['stage_keys'][-1]
    betas = out[last_stage]['betas'][0].detach().cpu().numpy()
    return betas


def main():
    parser = argparse.ArgumentParser(description="Predict SMPL betas from a single image")
    parser.add_argument('image_path', type=Path, help='Aligned person image path')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'b2a_expose_hrnet_demo.yaml',
        help='Model config file',
    )
    parser.add_argument(
        '--model-folder',
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / 'data'
        / 'trained_models'
        / 'shapy'
        / 'SHAPY_A',
        help='Folder with pre-trained checkpoints',
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='Device to run the network on',
    )
    parser.add_argument('--output', type=Path, default='betas.npy', help='File to save predicted betas')

    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    cfg = load_cfg(args.config, args.model_folder)
    model = build_network(cfg, device)

    crop_size = cfg.datasets.pose.transforms.crop_size
    mean = cfg.datasets.pose.transforms.mean
    std = cfg.datasets.pose.transforms.std

    img_tensor, targets = preprocess_image(args.image_path, crop_size, mean, std, device)
    betas = run_inference(model, img_tensor, targets)

    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(args.output, betas)
    print(f"Saved betas to {args.output}")


if __name__ == '__main__':
    main()
