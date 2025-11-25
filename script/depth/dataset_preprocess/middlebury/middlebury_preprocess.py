import os
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm

BASE_DATA_DIR = os.environ["BASE_DATA_DIR"]
RAW_DIR = BASE_DATA_DIR + "middlebury_raw"
PROCESSED_DIR = BASE_DATA_DIR + "middlebury_processed/test"


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
            processed_dirpath / f"{scene_folder.name}_rgb.png"
        ).resolve()
        moved_scene_depth_filepath = Path(
            processed_dirpath / f"{scene_folder.name}_depth.png"
        )
        shutil.copy(scene_rgb_file, moved_scene_rgb_filepath)
        scene_rgb_file = moved_scene_rgb_filepath

        subprocess.run(
            [
                "convert",
                scene_depth_file,
                moved_scene_depth_filepath,
            ]
        )

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
