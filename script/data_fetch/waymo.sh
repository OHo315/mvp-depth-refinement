echo "Fetching Waymo dataset..."
mkdir -p $BASE_DATA_DIR
hf download "bambezius/waymo_train" --repo-type dataset --local-dir $BASE_DATA_DIR

echo "Preprocessing Waymo dataset..."
python3 script/depth/dataset_preprocess/waymo/waymo_preprocess.py --parquet_dir $BASE_DATA_DIR/waymo_raw --outdir $BASE_DATA_DIR/waymo_preprocess
