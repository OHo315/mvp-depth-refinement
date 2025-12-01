echo "Fetching Diode Indoor dataset..."
TARGET_DIR=$BASE_DATA_DIR/diode/indoor/train
mkdir -p $TARGET_DIR 
hf download "bambezius/diode_indoor_train" --repo-type dataset --local-dir $TARGET_DIR
