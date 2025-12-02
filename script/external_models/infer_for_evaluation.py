import argparse
import logging
import os
import sys

import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

import debugpy

# Import model pipelines
from sharpdepth.pipeline.pipeline import SharpDepthPipeline   # SharpDepth
from base_depth_estimators import get_base_depth_estimator_fn # Estimator Model for Base Model of SharpDepth
from ppd.models.ppd import PixelPerfectDepth                  # Pixel Perfect
from depth_anything.dpt import DepthAnything                  # DepthAnything

# SharpDepth default arguments
sd_checkpoint_path = "prs-eth/marigold-v1-0"
sd_kind = "lotus"
dtype = torch.float32 # Defaulting to full precision
base_model = "unidepth" # Defaulting base model to UniDepth

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # ARGUMENTS
    parser = argparse.ArgumentParser(
        description="Run inference using a specified model on a specified dataset."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name of model to be used for inference."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="./dataset.yaml",
        help="The path to a yaml file containing the dataset configuration."
    )
    
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    args = parser.parse_args()

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # Dataset
    dataset = {}
    try:
        with open('r', args.dataset) as file:
            dataset = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file '{args.dataset}' was not found.")
        sys.exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit()
    
    # Dataset Name and Images
    data_name = dataset["name"]
    imgs = sorted(os.listdir(dataset["dir"]))

    # Model
    output_dir = f"./inference/unknown_{data_name}" # Path to the directory it should output to
    model = None # Eventually holds the name of the model
    pipeline = None # Eventually holds the pipeline of the model
    match args.model.lower():
        # SharpDepth
        case "sharp-depth" | "sharpdepth" | "sd":
            model = "sharp_depth"
            output_dir = f"./inference/sharpdepth_{data_name}"
            pipeline = SharpDepthPipeline.from_pretrained(sd_checkpoint_path, sharpdepth_kind=sd_kind, default_processing_resolution=768, default_denoising_steps=1)
            pipeline = pipeline.to(device, dtype=dtype)

            # NOTE TO SELF: Figure out whether other models require this base depth estimator
            base_depth_estimator_fn = get_base_depth_estimator_fn(base_model, device, dtype)

            # May be able to generalise this and take it out of match case.
            # Inference
            with torch.no_grad():
                for batch in tqdm(imgs):
                    # Read input image
                    rgb = Image.open(os.path.join(dataset["dir"], batch))
                    if args.debug: 
                        print("filename: ", os.path.join(dataset["dir"], batch))
                    out = pipeline(rgb, base_depth_estimator_fn)

                    out.depth_base_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}.jpg"))
                    out.depth_colored.save(os.path.join(output_dir, batch.split(".")[0] + f"_{args.base_model}_{model}.png"))
                    # Somehow need to output .npy (depth map) and .rgb (colour visual) images

        # Pixel Perfect
        case "pixel-perfect" | "pixelperfect" | "ppd":
            pipeline = PixelPerfectDepth.from_pretrained("andrew-healey/sharpdepth", subfolder="ppd")
            pipeline = PixelPerfectDepth.from_pretrained("andrew-healey/pixel-perfect-depth")
            for batch in tqdm(imgs):
                rgb = Image.open(os.path.join(dataset["dir"], batch))
                if args.debug: 
                    print("filename: ", os.path.join(dataset["dir"], batch))
                depth, resize_image = pipeline.infer_image(rgb)
                
            pass
        
        # Depth Anything
        case "depth-anything" | "depthanything" | "da":
            pass
        
        # Invalid Model
        case _:
            raise ValueError("Invalid model specified. Here are the valid models:\n\t- SharpDepth [sd]\n\t- PixelPerfect [ppd]\n\t- DepthAnything [da]\n\n")