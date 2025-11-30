echo "Fetching Diode Indoor dataset..."
mkdir -p $BASE_DATA_DIR
hf download "bambezius/diode_indoor_train" --repo-type dataset --local-dir $BASE_DATA_DIR
