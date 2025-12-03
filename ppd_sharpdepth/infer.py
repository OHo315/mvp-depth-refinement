# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import logging
import os
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image 
from torchvision.transforms.functional import pil_to_tensor, resize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import debugpy

from .depth_estimators import get_depth_estimator_fn, ModelArchitecture
from .preprocessors import MarigoldPreProcessor

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs."
    ) 

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of model.")

    # NOTE: There are two modes for inferences: txt and dir mode. If txt is specified, it is in txt mode, otherwise dir mode.
    parser.add_argument("--input_dir", type=str, required=True, help="Input image dataset directory")
    parser.add_argument("--input_txt", type=str, help="Txt filepath containing paths of images to infer and labels") 

    parser.add_argument("--output_dir", type=str, required=True, help="Output depth dataset directory.") 

    # inference setting
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument("--model_architecture", type=str, default="unidepth", help=f"Model to use for depth estimation. Options: {[e.value for e in ModelArchitecture]}")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached")


    checkpoint_path = args.checkpoint
    model_architecture = ModelArchitecture(args.model_architecture)
    output_dir = str(Path(os.environ["BASE_PREDS_DIR"]) / args.output_dir / model_architecture.value)
    input_dir = str(Path(os.environ["BASE_DATA_DIR"]) / args.input_dir)
    input_txt = args.input_txt
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

    # Determine img paths depending on mode.
    if not input_txt:
        imgs = list(sorted(os.listdir(input_dir)))
    else:
        with open(input_txt, "r") as f:
            # EXPECTED FORMAT PER LINE: '{rgb_path} {depth_label_path}'
            lines = [line.strip() for line in f.readlines()]
            imgs = [line.split(" ")[0] for line in lines]
            depths = [line.split(" ")[1] for line in lines]

    model_infer_fn = get_depth_estimator_fn(model_architecture, device, dtype, checkpoint_path)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i, img in tqdm(list(enumerate(imgs))):
            img_path = os.path.join(input_dir, img)
            # Read input image
            input_image = Image.open(img_path).convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb_int_1chw = pil_to_tensor(input_image)
            rgb_int_1chw = rgb_int_1chw.unsqueeze(0)  # [1, rgb, H, W], dtype int
            rgb_int_1chw = rgb_int_1chw.to(torch.int32)

            if args.debug: print("filename: ", img)

            depth_np_11hw = model_infer_fn(rgb_int_1chw, MarigoldPreProcessor)
            depth_np_11hw = depth_np_11hw.cpu().numpy()
            save_path = os.path.join(output_dir, img)
            os.makedirs(Path(save_path).parent, exist_ok=True)
            np.save(save_path, depth_np_11hw)

            #out.depth_base_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}.jpg"))
            #out.depth_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}_sharpdepth.png"))

    
    print(f"successfully saved outputs to {output_dir}.")


