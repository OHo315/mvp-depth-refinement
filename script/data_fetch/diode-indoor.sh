echo "Fetching Diode Indoor dataset..."
TARGET_DIR=$BASE_DATA_DIR/diode/indoor
mkdir -p $TARGET_DIR 
hf download "bambezius/diode_indoor_train" --repo-type dataset --local-dir $TARGET_DIR
gunzip $TARGET_DIR/diode.tar.gz
tar -xf $TARGET_DIR/diode.tar -C $TARGET_DIR
mv $TARGET_DIR/diode $TARGET_DIR/train
rm $TARGET_DIR/diode.tar
