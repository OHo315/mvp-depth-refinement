#export WORKSPACE_DIR="$(dirname $0)/../.."
export WORKSPACE_DIR=$BASE_DATA_DIR../
export PYTHONPATH="$WORKSPACE_DIR":$PYTHONPATH

echo $WORKSPACE_DIR
echo $PYTHONPATH

accelerate launch ppd_sharpdepth/training/train.py \
    --depth_weight 0.4 \
    --base_ckpt_dir andrew-healey/sharpdepth \
    --student_ckpt_dir andrew-healey/sharpdepth \
    --add_datetime_prefix \
    --report_to wandb \
    --mixed_precision bf16 \
    --seed 42 \
    --allow_tf32 \
    --learning_rate 1e-6 \
    --scale_lr \
    --lr_scheduler cosine \
    --lr_warmup_steps 200 \
    --tracker_project_name ppd_sharpdepth_train \
    --set_grads_to_none \
    --checkpointing_steps 5000 \
    --validation_steps 100 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 6 \
    --use_ema \
    --base_data_dir "$WORKSPACE_DIR/data/" \
    --config "$WORKSPACE_DIR/config/train_marigold_depth.yaml" \
    --output_dir "$WORKSPACE_DIR/train_output/" \
    --base_model unidepth \
    --denoiser pixel_perfect_depth
