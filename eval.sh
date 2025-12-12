#!/bin/sh

########################################
########## MIDDLEBURY SECTION ##########
########################################

#python -m ppd_sharpdepth.eval \
#	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
#	--model_architecture unidepth

#python -m ppd_sharpdepth.eval \
#	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
#	--model_architecture patchrefiner

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
	--model_architecture sharpdepth_ppd_zoedepth

########################################
########## NYU SECTION ##########
########################################

#python -m ppd_sharpdepth.eval \
#	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
#	--model_architecture pixelperfectdepth

#python -m ppd_sharpdepth.eval \
#	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
#	--model_architecture patchrefiner

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
	--model_architecture sharpdepth_ppd_zoedepth

#########################################
########## HYPERSIM SECTION ##########
########################################

#python -m ppd_sharpdepth.eval \
#	--dataset_config_path config/dataset_depth/data_hypersim_test.yaml \
#	--model_architecture pixelperfectdepth

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_hypersim_test.yaml \
	--model_architecture sharpdepth_ppd_zoedepth