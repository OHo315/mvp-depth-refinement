#!/bin/bash

cd Depth-Anything

BASE_HYPERSIM_DATASET_DIR=../data/hypersim_processed

for SPLIT in train val test; do
    SPLIT_DIR="$BASE_HYPERSIM_DATASET_DIR/$SPLIT"

    for dir in "$SPLIT_DIR"/*/; do
        echo "$dir"
        echo "Running depth anything on $SPLIT split..."
        python3 run.py --encoder vits --img-path "$dir" --outdir "$dir" --pred-only
    done
done

echo "Zipping depth_anything processed hypersim..."
zip -r "$BASE_DATA_DIR/depth_anything_hypersim.zip" "$BASE_HYPERSIM_DATASET_DIR"

