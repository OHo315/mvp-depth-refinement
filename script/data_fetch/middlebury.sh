echo "Fetching middlebury dataset..."
mkdir $BASE_DATA_DIR/middlebury_raw
wget -O $BASE_DATA_DIR/middlebury_raw/all.zip "https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip"
cd $BASE_DATA_DIR/middlebury_raw
unzip -q all.zip && rm all.zip

echo "Preprocessing middelbury dataset..."
python3 script/depth/dataset_preprocess/nyu/nyu_preprocess.py
