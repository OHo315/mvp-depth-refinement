#!/bin/sh

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
	--model_architecture unidepth

