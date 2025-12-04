import random

INPUT_FILE = "data_split/kitti_depth/eigen_test_files_with_gt_filtered.txt"
OUTPUT_FILE = "data_split/kitti_depth/eigen_test_files_with_gt_filtered_smaller.txt"
NUM_LINES = 150

SEED = 42
random.seed(SEED)

# Read all lines
with open(INPUT_FILE, "r") as f:
    lines = f.readlines()

# Sample 150 lines without replacement
sampled_lines = random.sample(lines, min(NUM_LINES, len(lines)))

# Write sampled lines to a new file
with open(OUTPUT_FILE, "w") as f:
    f.writelines(sampled_lines)

