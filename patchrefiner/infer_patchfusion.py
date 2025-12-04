import argparse
import logging
import os

#import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

#from models.patchrefiner import PatchRefiner
from models.patchfusion import PatchFusion

from utils.color import colorize

import debugpy

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using PatchRefiner."
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

    parser.add_argument("--base_model", type=str, default="zoedepth", help="Base model to use for PatchFusion. Options: zoedepth, depth_anything_small, depth_anything_big, depth_anything_large")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached")

    output_dir = args.output_dir
    input_dir = args.input_dir
    half_precision = args.half_precision

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
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

    match args.base_model.lower():
        case "zoedepth":
            model_name = 'Zhyever/patchfusion_zoedepth'
        case "depth_anything_small":
            model_name = 'Zhyever/patchfusion_depth_anything_vits14'
        case "depth_anything_big":
            model_name = 'Zhyever/patchfusion_depth_anything_vitb14'
        case "depth_anything" | "depth_anything_large":
            model_name = 'Zhyever/patchfusion_depth_anything_vitl14'
        case _:
            logging.warning(f"Invalid model name. Defaulting to zoedepth as base model...")
            model_name = 'Zhyever/patchfusion_zoedepth'

    pipeline = PatchFusion.from_pretrained(model_name)#.to(device).eval()
    assert pipeline.default_processing_resolution == 768, f"default_processing_resolution = {pipeline.default_processing_resolution}, expected 768"
    assert pipeline.default_denoising_steps == 1, f"default_denoising_steps = {pipeline.default_denoising_steps}, expected 1"
    
    pipeline = pipeline.to(device, dtype=dtype).eval()

    image_raw_shape = pipeline.tile_cfg['image_raw_shape']
    image_resizer = pipeline.resizer

    imgs = sorted(os.listdir(input_dir))
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(imgs):
            rgb = Image.open(os.path.join(input_dir, batch))
            if args.debug: print("filename: ", os.path.join(input_dir, batch))

            #image = cv2.imread('./examples/example_1.jpeg')
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = transforms.ToTensor()(np.asarray(rgb)) # raw image

            image_lr = image_resizer(image.unsqueeze(dim=0)).float().to(device)
            image_hr = F.interpolate(image.unsqueeze(dim=0), image_raw_shape, mode='bicubic', align_corners=True).float().to(device)

            mode = 'r128' # inference mode
            process_num = 4 # batch process size
            depth_prediction, _ = pipeline(mode='infer', cai_mode=mode, process_num=process_num, image_lr=image_lr, image_hr=image_hr)
            depth_prediction = F.interpolate(depth_prediction, image.shape[-2:])[0, 0].detach().cpu().numpy() # depth shape would be (h, w), similar to the input image
            color_prediction = colorize(depth_prediction)
            depth_map = transforms.toPILImage(depth_prediction)
            color_map = transforms.toPILImage(color_prediction)
            depth_map.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}_depth.npy"))
            color_map.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}_color.rgb"))