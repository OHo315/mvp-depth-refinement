#!/bin/bash

DATA_DIR="data"

if [ -d "$DATA_DIR" ]; then
    echo "Data directory exists. Exiting."
    exit 1
fi

read -p "Do you want a smaller hypersim dataset (y/n) " ANSWER

if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
    SMALL=true
else
    SMALL=false
fi

mkdir -p $DATA_DIR/hypersim

# Fetch hypersim
if SMALL; then
  # Fetch hypersim
  python3 script/depth/dataset_preprocess/hypersim/dataset_download_images.py \
    --downloads_dir $DATA_DIR/hypersim \
    --decompress_dir $DATA_DIR/hypersim \
    --delete_archive_after_decompress \
    --small

  # Unzip all zip files
  for f in $DATA_DIR/hypersim/*.zip; do
        unzip -o "$f" -d "${f%.zip}"
  done

  # Filter down dataset for small
  python3 script/depth/dataset_preprocess/hypersim/filter_hypersim_split.py \
    --input_csv data_split/hypersim_depth/metadata_images_split_scene_v1.csv \
    --output_csv data_split/hypersim_depth/metadata_images_split_scene_v1_small.csv \
    --scenes ai_001_001 ai_003_010 ai_001_010

  # Preprocess hypersim
  python3 script/depth/dataset_preprocess/hypersim/preprocess_hypersim.py \
    --split_csv data_split/hypersim_depth/metadata_images_split_scene_v1_small.csv \
    --dataset_dir $DATA_DIR/hypersim \
    --output_dir $DATA_DIR/hypersim_processed
 
else
  # Fetch hypersim
  python3 script/depth/dataset_preprocess/hypersim/dataset_download_images.py \
    --downloads_dir $DATA_DIR/hypersim \
    --decompress_dir $DATA_DIR/hypersim \
    --delete_archive_after_decompress

  # Unzip all zip files
  for f in $DATA_DIR/hypersim/*.zip; do
        unzip -o "$f" -d "${f%.zip}"
  done

  # Preprocess hypersim
  python3 script/depth/dataset_preprocess/hypersim/preprocess_hypersim.py \
    --split_csv data_split/hypersim_depth/metadata_images_split_scene_v1.csv \
    --dataset_dir $DATA_DIR/hypersim \
    --output_dir $DATA/hypersim_processed
fi

# Fetch nyu v2 raw dataset 
mkdir $DATA_DIR/nyu_v2_raw
wget -O $DATA_DIR/nyu_v2_raw "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

# Preprocess nyu v2
python3 script/depth/dataset_preprocess/nyu/nyu_preprocess.py
