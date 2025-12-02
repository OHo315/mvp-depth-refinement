#!/bin/sh

HYPERSIM_TRAINING_DIR="$BASE_DATA_DIR/hypersim_processed"
mkdir -p $HYPERSIM_TRAINING_DIR
hf download bambezius/hypersim --repo-type=dataset --local-dir $HYPERSIM_TRAINING_DIR
tar -xzf $HYPERSIM_TRAINING_DIR/train.tar.gz -C $HYPERSIM_TRAINING_DIR
rm $HYPERSIM_TRAINING_DIR/train.tar.gz

