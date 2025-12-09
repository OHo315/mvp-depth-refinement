from argparse import ArgumentParser
from src.dataset import get_dataset, DatasetMode
from pydantic import BaseModel
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from pprint import pprint
import subprocess
import math
from PIL import Image

from ppd_sharpdepth.depth_estimators import ModelArchitecture 
from script.evaluation.metrics import abs_rel, rmse, dbe_completeness, ppd_metric

def save_numpy_bitmap_as_png(bitmap_array: np.ndarray, file_path: str):
    """
    Converts a 2D NumPy array (bitmap) into a PNG image file.

    The function assumes the following:
    1. The input array is 2-dimensional (Height x Width).
    2. The values represent a binary image (e.g., 0 for black, 1 for white).
    3. The output image will be saved as a monochrome (1-bit) PNG.

    Args:
        bitmap_array: The 2D NumPy array representing the bitmap.
                      It's best if the array is of dtype bool or uint8.
        file_path: The path and filename for the output PNG (e.g., 'output.png').
    """
    
    # --- 1. Validate Input ---
    if bitmap_array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional. Found {bitmap_array.ndim} dimensions.")

    # --- 2. Data Preparation ---
    # Ensure the data is in the correct format (uint8) and normalize if necessary.
    # If the array is boolean (True/False), convert to 0/255.
    if bitmap_array.dtype == bool:
        # Convert boolean to 0 (False) or 255 (True) for visualization
        processed_array = bitmap_array.astype(np.uint8) * 255
        
    # If the array contains 0s and 1s, convert 1s to 255 for visibility
    elif np.max(bitmap_array) <= 1 and np.min(bitmap_array) >= 0:
        processed_array = (bitmap_array.astype(np.uint8) * 255)
        
    else:
        # Use as is, assuming it's already an 8-bit grayscale image (0-255)
        processed_array = bitmap_array.astype(np.uint8)


    # --- 3. Create PIL Image Object ---
    # Mode 'L' is for 8-bit grayscale.
    # If you specifically need a 1-bit image (mode '1'), you must ensure the
    # array is exactly 0 or 255 before casting to mode '1'.
    img = Image.fromarray(processed_array, mode='L')

    # --- 4. Save to PNG ---
    try:
        img.save(file_path, 'PNG')
        #print(f"Successfully saved bitmap to '{file_path}' (Grayscale).")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    parser = ArgumentParser("Script to evaluate outputs of models against ground truth labels.")

    parser.add_argument("--dataset_config_path", type=str, required=True, help="Path of the dataset config yaml file.")
    parser.add_argument("--model_architecture", type=str, required=True, help="Model.")

    args = parser.parse_args()

    dataset_config_path = args.dataset_config_path
    model_architecture = ModelArchitecture(args.model_architecture)
    run_message = input("Give a message for this eval run: ")
    
    BASE_DATA_DIR = Path(os.environ["BASE_DATA_DIR"])
    BASE_PREDS_DIR = Path(os.environ["BASE_PREDS_DIR"])
    results_filepath = Path("results.csv")

    cfg_data = OmegaConf.load(dataset_config_path)
    
    dataset = get_dataset(
        cfg_data, base_data_dir=BASE_DATA_DIR, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    model_architecture = ModelArchitecture(model_architecture) 
 
    total_abs_rel = 0.0
    total_rmse = 0.0
    total_ppd = 0.0
    total_valid_examples = 0
    total_valid_ppd_examples= 0

    # computing edge metric requires intrinsic data and high quality edges (from synthetic data), and hypersim is the only supported dataset which meets these requirements.
    with_edge_metric = cfg_data.name == "hypersim_depth"

    for data in tqdm(dataloader, desc="Evaluating"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        pred_path = BASE_PREDS_DIR / cfg_data.dir / model_architecture.value / (rgb_name + ".npy")
        depth_pred = np.load(str(pred_path)).astype(np.float32)
        
        depth_pred = np.squeeze(depth_pred)
        
        total_abs_rel += abs_rel(depth_pred, depth_raw, valid_mask)
        total_rmse += rmse(depth_pred, depth_raw, valid_mask)
        total_valid_examples += 1
        
        if with_edge_metric:
            intrinsics_ts = data["intrinsics"].squeeze()
            intrinsics = intrinsics_ts.numpy()
            ppd_score = ppd_metric(depth_pred, depth_raw, intrinsics)
            if ppd_score > 0:
                total_ppd += ppd_metric(depth_pred, depth_raw, intrinsics)
                total_valid_ppd_examples += 1
        
        # DEBUG NORMAL MAPS VIA VISUALISATION.
        #predicted_edge_path = BASE_PREDS_DIR / cfg_data.dir / model_architecture.value / (rgb_name + "_pred_edges.png")
        #ground_edge_path = BASE_PREDS_DIR / cfg_data.dir / model_architecture.value / (rgb_name + "_ground_edges.png")

        #save_numpy_bitmap_as_png(ground_edges, str(ground_edge_path))
        #save_numpy_bitmap_as_png(predicted_edges, str(predicted_edge_path))


    gitname = subprocess.check_output(["git", "config", "user.name"]).decode().strip()
    
    # Update tracker.
    df = pd.read_csv(results_filepath)
    metrics = {
        "id": len(df[df["gitname"] == gitname]),
        "gitname": gitname,
        "model_architecture": model_architecture.value,
        "preds_dataset": cfg_data.dir,
        "run_message": run_message,
        "abs_rel": total_abs_rel / total_valid_examples,
        "rmse": total_rmse / total_valid_examples,
        "ppd": total_ppd / total_valid_ppd_examples ,
    }
    df = pd.concat([df, pd.DataFrame([metrics])])
    df.to_csv(results_filepath, index=False)

    pprint(metrics)






        
        
   
