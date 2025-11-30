echo "Fetching ARKitScenes dataset..."
mkdir -p $BASE_DATA_DIR/arkitscenes_processed
python3 submodules/ARKitScenes/download_data.py upsampling \
  --split Training \
  --video_id_csv submodules/ARKitScenes/depth_upsampling/upsampling_train_val_splits.csv \
  --download_dir $BASE_DATA_DIR/arkitscenes_processed
