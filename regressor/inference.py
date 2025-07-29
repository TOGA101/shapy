"""Run SHAPY on an aligned full-body image to regress SMPL-X betas.

This script expects a configuration YAML and a checkpoint directory as used by
``regressor/demo.py``. It performs no alignment, detection or rendering.

Example:
    python regressor/inference.py --cfg regressor/configs/b2a_expose_hrnet_demo.yaml \
        --checkpoint output/SHAPY_A --image path/to/image.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple
import sys

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from omegaconf import OmegaConf
import os

# Support running from the repository root by adding required submodules to
# ``sys.path``. This lets the script resolve the local ``attributes`` package
# and the optional ``body_measurements`` utilities without requiring
# ``PYTHONPATH`` to be set.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mesh-mesh-intersection"))
sys.path.insert(0, str(ROOT / "attributes"))

# When the optional CUDA extension for mesh-mesh-intersection is not compiled,
# importing `body_measurements` fails. Create lightweight stub modules so that
# the rest of the code can run without the extension.
import types
if "mesh_mesh_intersect_cuda" not in sys.modules:
    stub = types.ModuleType("mesh_mesh_intersect_cuda")

    def _missing(*_args, **_kwargs):  # pragma: no cover - runtime fallback
        raise ImportError("mesh_mesh_intersect_cuda extension is unavailable")

    stub.mesh_to_mesh_forward = _missing
    sys.modules["mesh_mesh_intersect_cuda"] = stub

from human_shape.config.defaults import conf as default_conf
from human_shape.models.build import build_model
from human_shape.utils.checkpointer import Checkpointer


def load_config(cfg_path: Path | None) -> OmegaConf:
    """Load an experiment configuration."""
    cfg = default_conf.copy()
    base_dir = Path('.')
    if cfg_path is not None:
        cfg.merge_with(OmegaConf.load(str(cfg_path)))
        base_dir = cfg_path.parent
    cfg.is_training = False
    # Body measurement computation relies on the optional CUDA extension and is
    # unnecessary during inference. Disable it to avoid import errors.
    for key in ("smpl", "smplh", "smplx", "expose"):
        path = f"network.{key}.compute_measurements"
        if OmegaConf.select(cfg, path) is not None:
            OmegaConf.update(cfg, path, False, merge=False)

    def _resolve_paths(obj):
        if isinstance(obj, (dict, OmegaConf)):
            for k, v in obj.items():
                if isinstance(v, str) and v and not os.path.isabs(v) and (
                        '/' in v or '\\' in v):
                    obj[k] = str((base_dir / v).resolve())
                else:
                    _resolve_paths(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str) and v and not os.path.isabs(v) and (
                        '/' in v or '\\' in v):
                    obj[i] = str((base_dir / v).resolve())
                else:
                    _resolve_paths(v)

    _resolve_paths(cfg)

    return cfg


def build_network(cfg, device: torch.device, checkpoint_dir: Path | None):
    """Construct the network and load weights."""
    model_dict = build_model(cfg)
    model = model_dict["network"].to(device)

    ckpt_dir = Path(cfg.output_folder) / cfg.checkpoint_folder
    if checkpoint_dir is not None:
        ckpt_dir = checkpoint_dir
        cfg.pretrained = str(checkpoint_dir)
    ckpt_dir = ckpt_dir if ckpt_dir.is_dir() else Path(cfg.output_folder)
    checkpointer = Checkpointer(model, save_dir=str(ckpt_dir),
                                pretrained=cfg.pretrained)
    checkpointer.load_checkpoint()
    model.eval()
    return model


def preprocess_image(img: Image.Image, size: int,
                      mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    """Resize and normalize an image for the regressor."""
    transform = Compose([
        # ``antialias`` is only supported in newer torchvision versions.
        # TODO: switch to ``Resize(..., antialias=True)`` once the minimum
        # supported torchvision version includes it.
        Resize((size, size), interpolation=Image.BILINEAR),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    return transform(img).unsqueeze(0)


def infer_betas(model, image_tensor: torch.Tensor,
                device: torch.device) -> torch.Tensor:
    """Run the network and return the betas vector."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        out = model(image_tensor, targets=None, compute_losses=False)
    stage_keys: Tuple[str, ...] = tuple(out.get("stage_keys", []))
    if not stage_keys:
        raise RuntimeError("Model output missing stage keys")
    last_stage = out[stage_keys[-1]]
    betas = last_stage["betas"].squeeze(0)
    return betas.cpu()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg = load_config(args.cfg)
    cfg.use_cuda = device.type == "cuda"

    model = build_network(cfg, device, args.checkpoint)

    img = Image.open(args.image).convert("RGB")
    crop_size = cfg.datasets.pose.transforms.crop_size
    mean = cfg.datasets.pose.transforms.mean
    std = cfg.datasets.pose.transforms.std
    img_tensor = preprocess_image(img, crop_size, mean, std)

    betas = infer_betas(model, img_tensor, device)
    print("Estimated betas:", betas.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAPY single-image inference")
    parser.add_argument("--cfg", type=Path, required=True,
                        help="Path to experiment YAML configuration")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to checkpoint directory (unused, kept for backwards compatibility)")
    parser.add_argument("--image", type=Path, required=True,
                        help="Path to an aligned RGB image")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Computation device")
    main(parser.parse_args())
