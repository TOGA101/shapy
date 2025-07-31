import argparse
import os
import os.path as osp

import torch
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf

from loguru import logger

from human_shape.config.defaults import conf as default_conf
from human_shape.models.build import build_model
from human_shape.utils import Checkpointer


def load_model(exp_cfg, device):
    model_dict = build_model(exp_cfg)
    model = model_dict['network'].to(device)
    checkpoint_folder = osp.join(exp_cfg.output_folder, exp_cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder,
                                pretrained=exp_cfg.pretrained)
    checkpointer.load_checkpoint()
    model.eval()
    return model


def preprocess_image(path, crop_size, mean, std):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


@torch.no_grad()
def infer_single_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    stage_key = output['stage_keys'][-1]
    betas = output[stage_key]['betas'][0].detach().cpu().numpy()
    return betas


def main():
    parser = argparse.ArgumentParser(description='Estimate SMPL betas from an image')
    parser.add_argument('image', type=str, help='Path to aligned full-body image')
    parser.add_argument('--exp-cfg', default='configs/b2a_expose_hrnet_demo.yaml',
                        help='Configuration file path')
    parser.add_argument('--model-folder', default='../data/trained_models/shapy/SHAPY_A',
                        help='Folder with trained model checkpoints')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                        help='Computation device')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    cfg = default_conf.copy()
    cfg.merge_with(OmegaConf.load(args.exp_cfg))
    cfg.output_folder = args.model_folder
    cfg.is_training = False
    if 'smplx' in cfg.network:
        cfg.network.smplx.use_b2a = False
        cfg.network.smplx.use_a2b = False

    model = load_model(cfg, device)

    crop_size = cfg.datasets.pose.transforms.crop_size
    mean = cfg.datasets.pose.transforms.mean
    std = cfg.datasets.pose.transforms.std
    image_tensor = preprocess_image(args.image, crop_size, mean, std)

    betas = infer_single_image(model, image_tensor, device)
    print(betas)


if __name__ == '__main__':
    main()
