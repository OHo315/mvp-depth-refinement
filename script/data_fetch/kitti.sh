echo "Fetching kitti dataset..."
mkdir $BASE_DATA_DIR/kitti_raw
wget -O $BASE_DATA_DIR/kitti_raw/kitti_eigen_split_test.tar https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/kitti/kitti_eigen_split_test.tar
tar -xvf $BASE_DATA_DIR/kitti_raw/kitti_eigen_split_test.tar -C $BASE_DATA_DIR/kitti_raw