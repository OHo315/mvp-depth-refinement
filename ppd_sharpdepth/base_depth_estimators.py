from enum import Enum
import torch

from diffusers import UNet2DConditionModel
from unidepth.models import UniDepthV1
from .sharpdepth.data.datasets_and_samplers import get_dataset
from .sharpdepth.data.datasets_and_samplers.base_depth_dataset import (
    BaseDepthDataset,
    DatasetMode,
    DepthFileNameMode,
    get_pred_name,
)

from .depth_anything.dpt import DepthAnything
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2

from huggingface_hub import hf_hub_download
from ppd_sharpdepth.ppd.models.ppd import PixelPerfectDepth

import torch.nn.functional as F

from typing import Callable

def get_base_depth_estimator_fn(base_model: str, device: torch.device, dtype: torch.dtype) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if base_model == "unidepth":
        unidepth = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        unidepth = unidepth.to(device, dtype=dtype)
        def base_depth_estimator_fn(marigold_preprocessed_image_1chw, _raw_image_1chw):
            ret_11hw = unidepth.infer((marigold_preprocessed_image_1chw*255).squeeze().int())['depth']
            return ret_11hw

            # sanity check, for reference:

            # print(f"Input to unidepth. shape: {marigold_preprocessed_image.shape}, std: {marigold_preprocessed_image.std()}, mean: {marigold_preprocessed_image.mean()}, dtype: {marigold_preprocessed_image.dtype}")
            # print(f"Output from unidepth. shape: {ret_11hw.shape}, std: {ret_11hw.std()}, mean: {ret_11hw.mean()}, dtype: {ret_11hw.dtype}")
            # raise ValueError("Stop here")

            # Input to unidepth. shape: torch.Size([1, 3, 728, 768]), std: 0.32351335883140564, mean: 0.34626907110214233, dtype: torch.float32
            # Output from unidepth. shape: torch.Size([1, 1, 728, 768]), std: 0.360894113779068, mean: 1.302354335784912, dtype: torch.float32

    elif base_model in ["depth_anything_small", "depth_anything_large"]:

        encoder_kind = "vits" if base_model == "depth_anything_small" else "vitl"

        depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder_kind)).to(device).eval()

        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        def base_depth_estimator_fn(marigold_preprocessed_image_1chw, raw_image_1chw):

            image_1hwc = transform({'image': (raw_image_1chw/255.).permute(0, 2, 3, 1).squeeze(0).float().cpu().numpy()})['image'][None]
            image_1hwc = torch.from_numpy(image_1hwc).to(device)

            disparity_raw_1hw = depth_anything(image_1hwc)
            depth_raw_1hw = disparity_raw_1hw.max() - disparity_raw_1hw

            depth_resized_11hw = F.interpolate(depth_raw_1hw[None], (marigold_preprocessed_image_1chw.shape[-2], marigold_preprocessed_image_1chw.shape[-1]), mode='bilinear', align_corners=False)

            return depth_resized_11hw

            # sanity check (Good! Matches the stats in submodules/Depth-Anything/run.py):
            # filename:  submodules/SharpDepth/assets/in-the-wild_example/00.jpg
            # Input to depth_anything. shape: torch.Size([1, 3, 518, 546]), std: 1.4334324598312378, mean: -0.45399439334869385, dtype: torch.float32
            # Output from depth_anything. shape: torch.Size([1, 1, 728, 768]), std: 7.16904878616333, mean: 10.526022911071777, dtype: torch.float32

            # print(f"Input to depth_anything. shape: {image_1hwc.shape}, std: {image_1hwc.std()}, mean: {image_1hwc.mean()}, dtype: {image_1hwc.dtype}")
            # print(f"Resized output from depth_anything. shape: {disparity_raw_1hw.shape}, std: {disparity_raw_1hw.std()}, mean: {disparity_raw_1hw.mean()}, dtype: {disparity_raw_1hw.dtype}")
            # raise NotImplementedError("Depth Anything is not implemented yet")
        
    elif base_model == "pixel_perfect_depth":

        ckpt_path = hf_hub_download(repo_id="gangweix/pixel-perfect-depth", filename="ppd.pth")
        semantics_path = hf_hub_download(repo_id="depth-anything/Depth-Anything-V2-Large", filename="depth_anything_v2_vitl.pth")

        model = PixelPerfectDepth(semantics_pth=semantics_path, sampling_steps=4)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model = model.to(device).eval()

        def base_depth_estimator_fn(marigold_preprocessed_image_1chw, raw_image_1chw):
            H, W = marigold_preprocessed_image_1chw.squeeze(0).shape[1:3]
            raw_image_hwc = raw_image_1chw.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            depth_raw_11hw, _ = model.infer_image(raw_image_hwc)
            depth_11hw = F.interpolate(depth_raw_11hw, size=(H, W), mode='bilinear', align_corners=False)
            return depth_11hw

            # sanity check (passes!)
            # Input to pixel perfect depth. shape: (804, 848, 3), std: 82.61926369403086, mean: 88.29934251697487, dtype: uint8
            # Output from pixel perfect depth. shape: (1, 1, 3, 804), std: 0.3026374280452728, mean: 0.4216078519821167, dtype: float32

            # print(f"Input to pixel perfect depth. shape: {raw_image_hwc.shape}, std: {raw_image_hwc.std()}, mean: {raw_image_hwc.mean()}, dtype: {raw_image_hwc.dtype}")
            # print(f"Resized output from pixel perfect depth. shape: {depth_11hw.shape}, std: {depth_11hw.std()}, mean: {depth_11hw.mean()}, dtype: {depth_11hw.dtype}")
            # raise NotImplementedError("Pixel Perfect Depth is not implemented yet")

    
    else:
        raise ValueError(f"Invalid base model: {base_model}")
    
    return base_depth_estimator_fn