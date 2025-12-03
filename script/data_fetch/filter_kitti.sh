#!/bin/bash

# Remove bad lines, like ones containing None.

src_file="data_split/kitti_depth/eigen_test_files_with_gt.txt"
dst_file="data_split/kitti_depth/eigen_test_files_with_gt_filtered.txt"

# Delete lines containing "None" in place
sed -i '/None/d' "$src_file" > "$dst_file"

