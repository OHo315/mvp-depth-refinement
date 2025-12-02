#!/bin/sh

ARKITSCENES_TRAINING_DIR_COMPRESSED="$BASE_DATA_DIR/arkitscenes_processed/upsampling/TrainingCompressed"
ARKITSCENES_TRAINING_DIR="$BASE_DATA_DIR/arkitscenes_processed/upsampling/Training"
mkdir -p $ARKITSCENES_TRAINING_DIR_COMPRESSED
mkdir -p $ARKITSCENES_TRAINING_DIR
hf download bambezius/arkitscenes --repo-type=dataset --local-dir $ARKITSCENES_TRAINING_DIR_COMPRESSED
for file in $ARKITSCENES_TRAINING_DIR_COMPRESSED/*; do
	echo "Decompressing $file"
	TMP=$(mktemp -d)
	tar -xzf $file -C $TMP
	for scene_dir in $TMP/$(basename $file .tar.gz)/*; do
		mv $scene_dir $ARKITSCENES_TRAINING_DIR
	done
	rm -rf $TMP
done
rm -rf $ARKITSCENES_TRAINING_DIR_COMPRESSED
