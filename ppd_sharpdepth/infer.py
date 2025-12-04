# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import logging
import os
from pathlib import Path
from omegaconf import OmegaConf

os.environ["XFORMERS_DISABLED"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image 
from torchvision.transforms.functional import pil_to_tensor, resize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ppd_sharpdepth.sharpdepth.util.image_util import colorize_depth_maps, chw2hwc

import debugpy

from .depth_estimators import get_depth_estimator_fn, ModelArchitecture
from .preprocessors import MarigoldPreProcessor

from src.dataset import get_dataset, DatasetMode

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs."
    ) 

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of model.")
    parser.add_argument("--dataset_config_path", type=str, required=True, help="Path of the dataset config yaml file.")
    parser.add_argument("--model_architecture", type=str, required=True, help="Model.") 
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    #parser.add_argument("--input_dir", type=str, required=True, help="Input image dataset directory")
    #parser.add_argument("--output_dir", type=str, required=True, help="Output depth dataset directory.") 

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dataset_config_path = args.dataset_config_path
    model_architecture = ModelArchitecture(args.model_architecture)
    half_precision = args.half_precision
    #output_dir = str(Path(os.environ["BASE_PREDS_DIR"]) / args.output_dir / model_architecture.value)
    #input_dir = str(Path(os.environ["BASE_DATA_DIR"]) / args.input_dir)

    #os.makedirs(output_dir, exist_ok=True)

    cfg_data = OmegaConf.load(dataset_config_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(f"Running with half precision ({dtype}), might lead to suboptimal result.")
    else:
        dtype = torch.float32
        variant = None

    with open(cfg_data.filenames, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        imgs = [line.split(" ")[0] for line in lines]
        depths = [line.split(" ")[1] for line in lines]

    model_infer_fn = get_depth_estimator_fn(model_architecture, device, dtype, checkpoint_path)
    
    BASE_DATA_DIR = Path(os.environ["BASE_DATA_DIR"])
    BASE_PREDS_DIR = Path(os.environ["BASE_PREDS_DIR"])

    input_dir = BASE_DATA_DIR / cfg_data.dir
    output_dir = BASE_PREDS_DIR / cfg_data.dir / model_architecture.value

    color_map = "inferno_r"

    dataset = get_dataset(
        cfg_data, base_data_dir=BASE_DATA_DIR, mode=DatasetMode.EVAL
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    for data in tqdm(dataloader, desc="Inferring"):
        # GT data
        rgb_raw_11chw = data["rgb_int"]
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        depth_np_11hw = model_infer_fn(rgb_raw_11chw, MarigoldPreProcessor)
        depth_np_11hw = depth_np_11hw.cpu().numpy()
        save_path = output_dir / rgb_name
        os.makedirs(save_path.parent, exist_ok=True)
        np.save(save_path, depth_np_11hw)

        # Clip to dataset min max
        depth_np_11hw = np.clip(
            depth_np_11hw, a_min=dataset.min_depth, a_max=dataset.max_depth
        )
        # clip to d > 0 for evaluation
        depth_np_11hw = np.clip(depth_np_11hw, a_min=1e-6, a_max=None)

        depth_np_11hw = np.squeeze(depth_np_11hw)
        depth_np_11hw *= valid_mask 

        # Generate color maps

        depth_colored = colorize_depth_maps(depth_raw, 0, depth_raw.max(), cmap=color_map).squeeze()

        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_label_colored_img = Image.fromarray(depth_colored_hwc)

        depth_colored = colorize_depth_maps(depth_np_11hw, 0, depth_np_11hw.max(), cmap=color_map).squeeze()

        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_pred_colored_img = Image.fromarray(depth_colored_hwc) 

        depth_label_colored_img.save(f"{save_path.parent}/{save_path.stem}_depth_label.png")
        depth_pred_colored_img.save(f"{save_path.parent}/{save_path.stem}_depth_pred.png")
    
    print(f"successfully saved outputs.")


