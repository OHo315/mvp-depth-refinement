import os
import re
from pathlib import Path
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
from pydantic import BaseModel

BASE_DATA_DIR = os.environ["BASE_DATA_DIR"]
RAW_DIR = BASE_DATA_DIR + "middlebury_raw/data"
PROCESSED_DIR = BASE_DATA_DIR + "middlebury_processed/test"


class Calibration(BaseModel):
    doffs: float
    baseline: float
    focal_length: float


def parse_calibration_file(filepath: Path) -> Calibration:
    with open(filepath, "r") as f:
        lines = f.readlines()
    kv_pairs = {k: v for k, v in map(lambda line: line.split("="), lines)}
    doffs = float(kv_pairs["doffs"])
    baseline = float(kv_pairs["baseline"])
    focal_length = float(kv_pairs["cam0"].split(";")[0][1:].split(" ")[0])
    calibration = Calibration(doffs=doffs, baseline=baseline, focal_length=focal_length)

    return calibration


def read_pfm(file_path: Path):
    with open(file_path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(f.readline().decode("utf-8").rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:  # big-endian
            endian = ">"

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data


def disparity_to_depth(disparity_map, baseline_mm, focal_length_pixels, doffs_pixels):
    """
    Converts a disparity map (in pixels) to a depth map (in millimeters)
    using the standard stereo vision formula.

    Z = (baseline * f) / (disparity + doffs)
    """
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)

    # Disparity values of 0 or inf are invalid
    valid_mask = (disparity_map > 0) & np.isfinite(disparity_map)

    denominator = disparity_map[valid_mask] + doffs_pixels
    numerator = baseline_mm * focal_length_pixels

    depth_map[valid_mask] = numerator / denominator

    return depth_map


def process_middlebury_raw() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_dirpath = Path(PROCESSED_DIR)
    data_split = []
    scene_folders = list(Path(RAW_DIR).iterdir())
    for scene_folder in tqdm(
        scene_folders, desc="moving and converting middlebury files..."
    ):
        # Note that we have stereo images, so take the (rgb, depth) which contain 0 in the name.

        # Rename images, move them to preprocessed dir and convert depth map to grayscale png.
        scene_rgb_file = scene_folder / "im0.png"
        scene_depth_file = scene_folder / "disp0.pfm"

        moved_scene_rgb_filepath = (
            processed_dirpath / f"rgb_{scene_folder.name}.png"
        ).resolve()
        moved_scene_depth_filepath = Path(
            processed_dirpath / f"depth_{scene_folder.name}.png"
        )
        print("from",scene_rgb_file) 
        print("to", moved_scene_rgb_filepath)
        print("scene_folder", scene_folder)
        shutil.copy(scene_rgb_file, moved_scene_rgb_filepath)

        img = read_pfm(scene_depth_file)

        calibration_filepath = scene_folder / "calib.txt"
        calibration = parse_calibration_file(calibration_filepath)

        # Note that depths are in mm, so when we cast to ints for saving as a grayscale image, we truncate to mm level precision.
        depth_map = disparity_to_depth(
            img,
            calibration.baseline,
            calibration.focal_length,
            calibration.doffs,
        )

        depth_map = depth_map.astype(np.uint16)

        Image.fromarray(depth_map).save(moved_scene_depth_filepath)

        # Generate the train, val and test txt splits.
        data_split.append(
            f"test/{moved_scene_rgb_filepath.name} test/{moved_scene_depth_filepath.name}"
        )

    data_split_filepath = Path(
        BASE_DATA_DIR + "../data_split/middlebury_depth/filename_list_test.txt"
    )
    data_split_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(data_split_filepath, "w") as f:
        f.write("\n".join(data_split))
    print("created test data split txt...")


if __name__ == "__main__":
    process_middlebury_raw()
