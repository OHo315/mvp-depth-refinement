#!/bin/sh

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_kitti_eigen_test.yaml \
	--model_architecture sharpdepth_lotus_unidepth

