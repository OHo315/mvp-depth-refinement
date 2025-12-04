#python -m ppd_sharpdepth.infer \
#	--checkpoint submodules/SharpDepth/checkpoints/sharpdepth \
#	--model_architecture sharpdepth_lotus_unidepth \
#	--input_txt data_split/kitti_depth/eigen_test_files_with_gt_filtered.txt \
#	--input_dir kitti \
#	--output_dir kitti \

#python -m ppd_sharpdepth.infer \
#	--checkpoint submodules/SharpDepth/checkpoints/sharpdepth \
#	--model_architecture sharpdepth_ppd_unidepth \
#	--input_txt data_split/kitti_depth/eigen_test_files_with_gt_filtered.txt \
#	--input_dir kitti \
#	--output_dir kitti \

#python -m ppd_sharpdepth.infer \
#	--checkpoint submodules/SharpDepth/checkpoints/sharpdepth \
#	--dataset_config_path config/dataset_depth/data_kitti_eigen_test.yaml \
#	--model_architecture sharpdepth_ppd_unidepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "lpiccinelli/unidepth-v1-vitl14" \
#	--dataset_config_path config/dataset_depth/data_kitti_eigen_test.yaml \
#	--model_architecture unidepth

python -m ppd_sharpdepth.infer \
	--checkpoint "andrew-healey/sharpdepth" \
	--dataset_config_path config/dataset_depth/data_kitti_eigen_test.yaml \
	--model_architecture pixelperfectdepth



