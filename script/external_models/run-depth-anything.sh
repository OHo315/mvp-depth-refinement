#!/bin/bash

pushd submodules/Depth-Anything
source env/bin/activate

######################

BASE_HYPERSIM_DATASET_DIR=../data/hypersim_processed

for SPLIT in train val test; do
    SPLIT_DIR="$BASE_HYPERSIM_DATASET_DIR/$SPLIT"
    echo "Running depth anything on hypersim $SPLIT split..."

    for dir in "$SPLIT_DIR"/*/; do
        LABELS_DIR="${dir%/}_labels"
        mkdir $LABELS_DIR 
        mv "$dir"depth*  $LABELS_DIR
        python3 run.py --encoder vits --img-path "$dir" --outdir "$dir" --pred-only
        mv "$LABELS_DIR"/depth* $dir
        rm -rf $LABELS_DIR
    done
done

echo "Symlinking depth_anything processed hypersim..."
ln -sf "$BASE_HYPERSIM_DATASET_DIR/" "$BASE_DATA_DIR/depth_anything_hypersim"

######################

BASE_NYU_DATASET_DIR=../data/nyu_v2_processed

for SPLIT in train test; do
    SPLIT_DIR="$BASE_NYU_DATASET_DIR/$SPLIT"
    echo "Running depth anything on NYU_V2 $SPLIT split..."

    for dir in "$SPLIT_DIR"/*/; do
        LABELS_DIR="${dir%/}_labels"
        mkdir $LABELS_DIR 
        mv "$dir"depth*  $LABELS_DIR
        python3 run.py --encoder vits --img-path "$dir" --outdir "$dir" --pred-only
        mv "$LABELS_DIR"/depth* $dir
        rm -rf $LABELS_DIR
    done
done

echo "Zipping depth_anything processed NYU_V2..."
zip -r "$BASE_DATA_DIR/depth_anything_nyu_v2.zip" "$BASE_NYU_DATASET_DIR"


######################

BASE_MIDDLEBURY_DATASET_DIR=../data/middlebury_processed
SPLIT_DIR="$BASE_MIDDLEBURY_DATASET_DIR/test"
LABELS_DIR="${SPLIT_DIR%/}_labels"

mkdir $LABELS_DIR 
mv "$SPLIT_DIR/"depth*  $LABELS_DIR
echo "Running depth anything on middlebury $SPLIT split..."
python3 run.py --encoder vits --img-path "$SPLIT_DIR" --outdir "$SPLIT_DIR" --pred-only
mv "$LABELS_DIR"/depth* $SPLIT_DIR
rm -rf $LABELS_DIR

echo "Zipping depth_anything processed middlebury..."
zip -r "$BASE_DATA_DIR/depth_anything_middlebury.zip" "$BASE_MIDDLEBURY_DATASET_DIR"
