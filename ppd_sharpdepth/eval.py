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

from ppd_sharpdepth.depth_estimators import ModelArchitecture 
from script.evaluation.metrics import abs_rel, rmse, dbe_completeness

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
    total_dbe = 0.0

    for data in tqdm(dataloader, desc="Evaluating"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        pred_path = BASE_PREDS_DIR / cfg_data.dir / model_architecture.value / (rgb_name + ".npy")
        depth_pred = np.load(str(pred_path)).astype(np.float32)

        # Clip to dataset min max
        depth_pred = np.clip(
            depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        
        depth_pred = np.squeeze(depth_pred)

        total_abs_rel += abs_rel(depth_pred, depth_raw, valid_mask)   
        total_rmse += rmse(depth_pred, depth_raw, valid_mask)
        total_dbe += dbe_completeness(depth_pred, depth_raw, valid_mask) 

    gitname = subprocess.check_output(["git", "config", "user.name"]).decode().strip()
    
    # Update tracker.
    df = pd.read_csv(results_filepath)
    metrics = {
        "id": len(df[df["gitname"] == gitname]),
        "gitname": gitname,
        "model_architecture": model_architecture.value,
        "preds_dataset": cfg_data.dir,
        "run_message": run_message,
        "abs_rel": total_abs_rel / len(dataloader),
        "rmse": total_rmse / len(dataloader),
        "dbe": total_dbe / len(dataloader),
    }
    df = pd.concat([df, pd.DataFrame([metrics])])
    df.to_csv(results_filepath)

    pprint(metrics)






        
        
   
