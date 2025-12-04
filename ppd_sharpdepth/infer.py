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

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i, img in tqdm(list(enumerate(imgs))):
            img_path = input_dir/ img
            # Read input image
            input_image = Image.open(img_path).convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb_int_1chw = pil_to_tensor(input_image)
            rgb_int_1chw = rgb_int_1chw.unsqueeze(0)  # [1, rgb, H, W], dtype int
            rgb_int_1chw = rgb_int_1chw.to(torch.int32)

            depth_np_11hw = model_infer_fn(rgb_int_1chw, MarigoldPreProcessor)
            depth_np_11hw = depth_np_11hw.cpu().numpy()
            save_path = output_dir / img
            os.makedirs(save_path.parent, exist_ok=True)
            np.save(save_path, depth_np_11hw)

            #out.depth_base_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}.jpg"))
            #out.depth_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}_sharpdepth.png"))

    print(f"successfully saved outputs to {output_dir}.")


