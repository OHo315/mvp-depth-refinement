#export WORKSPACE_DIR="$(dirname $0)/../.."
export WORKSPACE_DIR=$BASE_DATA_DIR../
export PYTHONPATH="$WORKSPACE_DIR":$PYTHONPATH

echo $WORKSPACE_DIR
echo $PYTHONPATH


accelerate launch --num_processes 2 --gpu-ids 0,1 ppd_sharpdepth/training/train.py \
    --sds_loss_weight 10.0 \
    --depth_weight 0.4 \
    --base_ckpt_dir andrew-healey/sharpdepth \
    --student_ckpt_dir andrew-healey/sharpdepth \
    --add_datetime_prefix \
    --report_to wandb \
    --mixed_precision bf16 \
    --seed 42 \
    --allow_tf32 \
    --learning_rate 1e-5 \
    --wandb_name "lr_1e-5" \
    --lr_scheduler cosine \
    --lr_warmup_steps 150 \
    --tracker_project_name ppd_sharpdepth_train \
    --set_grads_to_none \
    --checkpointing_steps 5000 \
    --validation_steps 100 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --use_ema \
    --base_data_dir "$WORKSPACE_DIR/data/" \
    --config "$WORKSPACE_DIR/config/train_marigold_depth.yaml" \
    --output_dir "$WORKSPACE_DIR/train_output/" \
    --base_model unidepth \
    --denoiser pixel_perfect_depth \
    --use_conditioning_probability 0.8 \
    --dit_patch_encoder_lr_multiplier 0.01 \
    "$@"
