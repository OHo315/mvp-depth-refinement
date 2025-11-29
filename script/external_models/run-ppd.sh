##!/bin/bash

#pushd submodules/pixel-perfect-depth 
#source env/bin/activate

## Define some dir vars 
#BASE_DATASET_DIR=../../data/hypersim_processed

#TRAIN_DIR=$BASE_DATASET_DIR/train
#TRAIN_INPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/train_in
#TRAIN_OUTPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/train_out

#VAL_DIR=$BASE_DATASET_DIR/val
#VAL_INPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/val_in
#VAL_OUTPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/val_out

#TEST_DIR=$BASE_DATASET_DIR/test
#TEST_INPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/test_in
#TEST_OUTPUT_DIR=$BASE_DATASET_DIR/pixel_perfect_depth/test_out

## Create respective input/ouput dirs
#mkdir -p $TRAIN_INPUT_DIR
#mkdir -p $TRAIN_OUTPUT_DIR

#mkdir -p $VAL_INPUT_DIR
#mkdir -p $VAL_OUTPUT_DIR

#mkdir -p $TEST_INPUT_DIR
#mkdir -p $TEST_OUTPUT_DIR

## Isolate the rgb files from the train, val and test dirs as input for pixel perfect depth
#cp $TRAIN_DIR/ai*/rgb* $TRAIN_INPUT_DIR
#cp $VAL_DIR/ai*/rgb* $VAL_INPUT_DIR
#cp $TEST_DIR/ai*/rgb* $TEST_INPUT_DIR

#echo "Running pixel perfect depth on training split..."
#python3 run.py --img_path $TRAIN_INPUT_DIR --outdir $TRAIN_OUTPUT_DIR --pred_only
#echo "Running pixel perfect depth on validation split..."
#python3 run.py --img_path $VAL_INPUT_DIR --outdir $VAL_OUTPUT_DIR --pred_only
#echo "Running pixel perfect depth on testing split..."
#python3 run.py --img_path $TEST_INPUT_DIR --outdir $TEST_OUTPUT_DIR --pred_only

## Zip train, val and test datasets for download
#BASE_ZIPPED_OUTPUT_DIR=../data/pixel_perfect_depth/hypersim
#mkdir -p $BASE_ZIPPED_OUTPUT_DIR

#echo "Zipping pixel_perfect_depth datasplits..."
#ln -sf $TRAIN_OUTPUT_DIR $BASE_ZIPPED_OUTPUT_DIR/train
#ln -sf $VAL_OUTPUT_DIR $BASE_ZIPPED_OUTPUT_DIR/val
#ln -sf $TEST_OUTPUT_DIR $BASE_ZIPPED_OUTPUT_DIR/test

#popd

##!/bin/bash

#pushd submodules/Depth-Anything
#source env/bin/activate

#######################

#BASE_HYPERSIM_DATASET_DIR=../../data/hypersim_processed

#for SPLIT in train val test; do
#    SPLIT_DIR="$BASE_HYPERSIM_DATASET_DIR/$SPLIT"
#    echo "Running depth anything on hypersim $SPLIT split..."

#    for dir in "$SPLIT_DIR"/*/; do
#        LABELS_DIR="${dir%/}_labels"
#        mkdir $LABELS_DIR 
#        mv "$dir"depth*  $LABELS_DIR
#        python3 run.py --encoder vits --img-path "$dir" --outdir "$dir" --pred-only
#        mv "$LABELS_DIR"/depth* $dir
#        rm -rf $LABELS_DIR
#    done
#done

#echo "Symlinking depth_anything processed hypersim..."
#ln -sf "$BASE_HYPERSIM_DATASET_DIR/" "$BASE_DATA_DIR/depth_anything_hypersim"

######################


pushd submodules/pixel-perfect-depth 
source env/bin/activate

######################

BASE_NYU_DATASET_DIR=../../data/nyu_v2_processed

for SPLIT in train test; do
    SPLIT_DIR="$BASE_NYU_DATASET_DIR/$SPLIT"
    echo "Running ppd on NYU_V2 $SPLIT split..."

    for dir in "$SPLIT_DIR"/*/; do
        LABELS_DIR="${dir%/}_labels"
        mkdir $LABELS_DIR 
        mv "$dir"depth*  $LABELS_DIR
        mv "$dir"filled*  $LABELS_DIR
        mv "$dir"*depth.png  $LABELS_DIR
        python3 run.py --img_path "$dir" --outdir "$dir" --pred_only
        mv "$LABELS_DIR"/depth* $dir
        mv "$LABELS_DIR"/filled* $dir
        mv "$LABELS_DIR"/*depth.png $dir
        rm -rf $LABELS_DIR
    done
done

# echo "Zipping ppd processed NYU_V2..."
# zip -r "$BASE_DATA_DIR/depth_anything_nyu_v2.zip" "$BASE_NYU_DATASET_DIR"


######################

# BASE_MIDDLEBURY_DATASET_DIR=../data/middlebury_processed
# SPLIT_DIR="$BASE_MIDDLEBURY_DATASET_DIR/test"
# LABELS_DIR="${SPLIT_DIR%/}_labels"

# mkdir $LABELS_DIR 
# mv "$SPLIT_DIR/"depth*  $LABELS_DIR
# echo "Running depth anything on middlebury $SPLIT split..."
# python3 run.py --encoder vits --img-path "$SPLIT_DIR" --outdir "$SPLIT_DIR" --pred-only
# mv "$LABELS_DIR"/depth* $SPLIT_DIR
# rm -rf $LABELS_DIR

# echo "Zipping ppd processed middlebury..."
# zip -r "$BASE_DATA_DIR/ppd_middlebury.zip" "$BASE_MIDDLEBURY_DATASET_DIR"

popd
