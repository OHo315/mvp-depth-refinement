# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .sharpdepth.pipeline.pipeline import SharpDepthPipeline

import debugpy

from .base_depth_estimators import get_base_depth_estimator_fn

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

    parser.add_argument("--base_model", type=str, default="unidepth", help="Base model to use for depth estimation. Options: unidepth, depth_anything_small, depth_anything_large, pixel_perfect_depth")
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

    base_depth_estimator_fn = get_base_depth_estimator_fn(args.base_model, device, dtype)

    imgs = sorted(os.listdir(input_dir))
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(imgs):
            # Read input image
            rgb = Image.open(os.path.join(input_dir, batch))
            if args.debug: print("filename: ", os.path.join(input_dir, batch))
            out = pipeline(rgb, base_depth_estimator_fn, processing_res=768, denoising_steps=1)

            out.depth_base_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}.jpg"))
            out.depth_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}_sharpdepth.png"))

