#!/bin/bash

pushd submodules/SharpDepth
source env/bin/activate

# Define some dir vars 
BASE_DATASET_DIR=../../data/hypersim_processed

TRAIN_DIR=$BASE_DATASET_DIR/train
TRAIN_INPUT_DIR=$BASE_DATASET_DIR/sharpdepth/train_in
TRAIN_OUTPUT_DIR=$BASE_DATASET_DIR/sharpdepth/train_out

VAL_DIR=$BASE_DATASET_DIR/val
VAL_INPUT_DIR=$BASE_DATASET_DIR/sharpdepth/val_in
VAL_OUTPUT_DIR=$BASE_DATASET_DIR/sharpdepth/val_out

TEST_DIR=$BASE_DATASET_DIR/test
TEST_INPUT_DIR=$BASE_DATASET_DIR/sharpdepth/test_in
TEST_OUTPUT_DIR=$BASE_DATASET_DIR/sharpdepth/test_out

# Create respective input/ouput dirs
mkdir -p $TRAIN_INPUT_DIR
mkdir -p $TRAIN_OUTPUT_DIR

mkdir -p $VAL_INPUT_DIR
mkdir -p $VAL_OUTPUT_DIR

mkdir -p $TEST_INPUT_DIR
mkdir -p $TEST_OUTPUT_DIR

# Isolate the rgb files from the train, val and test dirs as input for pixel perfect depth
# cp $TRAIN_DIR/ai*/rgb* $TRAIN_INPUT_DIR


echo "Resizing images (takes a few minutes)..."
mogrify -path "$TRAIN_INPUT_DIR" -resize 640x480! "$TRAIN_DIR"/ai*/rgb*
rm $TRAIN_INPUT_DIR/*_depth.png
mogrify -path "$VAL_INPUT_DIR" -resize 640x480! "$VAL_DIR"/ai*/rgb*
rm $VAL_INPUT_DIR/*_depth.png
mogrify -path "$TEST_INPUT_DIR" -resize 640x480! "$TEST_DIR"/ai*/rgb*
rm $TEST_INPUT_DIR/*_depth.png

export PYTHONPATH=$(pwd)/src/sharpdepth/evaluation:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

echo "Running sharpdepth on validation split..."
python3 app.py --input_dir $VAL_INPUT_DIR --output_dir $VAL_OUTPUT_DIR
echo "Running sharpdepth on training split..."
python3 app.py --input_dir $TRAIN_INPUT_DIR --output_dir $TRAIN_OUTPUT_DIR
echo "Running sharpdepth on testing split..."
python3 app.py --input_dir $TEST_INPUT_DIR --output_dir $TEST_OUTPUT_DIR

popd

