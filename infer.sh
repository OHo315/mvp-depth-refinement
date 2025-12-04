python -m ppd_sharpdepth.infer \
	--checkpoint submodules/SharpDepth/checkpoints/sharpdepth \
	--model_architecture sharpdepth_lotus_unidepth \
	--input_txt data_split/kitti_depth/eigen_test_files_with_gt_filtered.txt \
	--input_dir kitti \
	--output_dir kitti \
