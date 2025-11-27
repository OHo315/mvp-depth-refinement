# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from unidepth.models import UniDepthV1

from .sharpdepth.data.datasets_and_samplers import get_dataset
from .sharpdepth.data.datasets_and_samplers.base_depth_dataset import (
    BaseDepthDataset,
    DatasetMode,
    DepthFileNameMode,
    get_pred_name,
)
from .sharpdepth.pipeline.pipeline import SharpDepthPipeline

from .depth_anything.dpt import DepthAnything
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2

import torch.nn.functional as F

import debugpy

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using SharpDepth."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

    # inference setting
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument("--base_model", type=str, default="unidepth", help="Base model to use for depth estimation. Options: unidepth, depth_anything_small, depth_anything_large")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached")


    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    input_dir = args.input_dir
    half_precision = args.half_precision

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(f"Inference settings: checkpoint = `{checkpoint_path}`, ")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(f"Running with half precision ({dtype}), might lead to suboptimal result.")
    else:
        dtype = torch.float32
        variant = None

    pipeline = SharpDepthPipeline.from_pretrained(checkpoint_path)

    pipeline = pipeline.to(device, dtype=dtype)


    if args.base_model == "unidepth":
        unidepth = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        unidepth = unidepth.to(device, dtype=dtype)
        def base_estimator_fn(marigold_preprocessed_image, raw_image):
            ret = unidepth.infer((marigold_preprocessed_image*255).squeeze().int())['depth']
            # print(f"Input to unidepth. shape: {x.shape}, std: {x.std()}, mean: {x.mean()}, dtype: {x.dtype}")
            # print(f"Output from unidepth. shape: {ret.shape}, std: {ret.std()}, mean: {ret.mean()}, dtype: {ret.dtype}")
            # raise ValueError("Stop here")

            # Input to unidepth. shape: torch.Size([1, 3, 728, 768]), std: 0.32351335883140564, mean: 0.34626907110214233, dtype: torch.float32
            # Output from unidepth. shape: torch.Size([1, 1, 728, 768]), std: 0.360894113779068, mean: 1.302354335784912, dtype: torch.float32

            return ret
    elif args.base_model in ["depth_anything_small", "depth_anything_large"]:

        encoder_kind = "vits" if args.base_model == "depth_anything_small" else "vitl"

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

        def base_estimator_fn(marigold_preprocessed_image, raw_image):

            image = transform({'image': (raw_image/255.).permute(0, 2, 3, 1).squeeze().numpy()})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(device)


            # still sorta mismatches, TODO fix this
            print(f"Input to depth_anything. shape: {image.shape}, std: {image.std()}, mean: {image.mean()}, dtype: {image.dtype}")

            depth_raw = depth_anything(image)
            print(f"Output from depth_anything. shape: {depth_raw.shape}, std: {depth_raw.std()}, mean: {depth_raw.mean()}, dtype: {depth_raw.dtype}")

            depth_resized = F.interpolate(depth_raw[None], (marigold_preprocessed_image.shape[-2], marigold_preprocessed_image.shape[-1]), mode='bilinear', align_corners=False)

            return depth_resized.unsqueeze(0)


            # for reference, depth_anything has this interface:
            # Input to depth_anything. shape: torch.Size([1, 3, 518, 686]), std: 0.13804294168949127, mean: -1.7824370861053467, dtype: torch.float32
            # Output from depth_anything. shape: torch.Size([1, 518, 686]), std: 2.3302955627441406, mean: 5.578943729400635, dtype: torch.float32

            raise NotImplementedError("Depth Anything is not implemented yet")
    else:
        raise ValueError(f"Invalid base model: {args.base_model}")

    imgs = sorted(os.listdir(input_dir))
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(imgs):
            # Read input image
            rgb = Image.open(os.path.join(input_dir, batch))
            out = pipeline(rgb, base_estimator_fn, processing_res=768, denoising_steps=1)
            depth_colored = out.depth_colored
            unidepth_colored = out.depth_base_colored

            depth_colored.save(os.path.join(output_dir, batch))
            unidepth_colored.save(os.path.join(output_dir, batch.split(".")[0] + "_uni.jpg"))
