import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

# remove entries where the rgb entry is solid black

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_file", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    args = parser.parse_args()

    txt_file = args.txt_file
    datadir = args.datadir

    with open(txt_file, "r") as f:
        lines = f.readlines()
    
    # lines = [line for line in lines if "ai_003_001/rgb_cam_00_fr0085.png" in line]
    
    new_lines = []
    for line in tqdm(lines):
        rgb_path, depth_path = line.split()
        rgb = cv2.imread(os.path.join(datadir, rgb_path))
        if np.all(rgb == 0):
            print(f"Removing {line} because the rgb is solid black")
        else:
          new_lines.append(line)

    with open(txt_file, "w") as f:
        f.writelines(new_lines)
