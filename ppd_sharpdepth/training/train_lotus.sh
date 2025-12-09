#export WORKSPACE_DIR="$(dirname $0)/../.."
export WORKSPACE_DIR=$BASE_DATA_DIR../
export PYTHONPATH="$WORKSPACE_DIR":$PYTHONPATH

echo $WORKSPACE_DIR
echo $PYTHONPATH

num_gpus=$NUM_GPUS
macrobatch_size=8

gradient_accumulation_steps=$((macrobatch_size / num_gpus))

accelerate launch --num_processes $num_gpus ppd_sharpdepth/training/train.py \
    --sds_loss_weight 1.0 \
    --depth_weight 0.4 \
    --base_ckpt_dir andrew-healey/sharpdepth \
    --student_ckpt_dir andrew-healey/sharpdepth \
    --add_datetime_prefix \
    --report_to wandb \
    --mixed_precision bf16 \
    --seed 42 \
    --allow_tf32 \
    --learning_rate 5e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --tracker_project_name lotus_train_andrew \
    --wandb_name "lotus_lr_5e-5" \
    --set_grads_to_none \
    --checkpointing_steps 500 \
    --validation_steps 100 \
    --train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 8 \
    --use_ema \
    --base_data_dir "$WORKSPACE_DIR/data/" \
    --config "$WORKSPACE_DIR/config/train_marigold_depth.yaml" \
    --output_dir "$WORKSPACE_DIR/train_output/" \
    --base_model unidepth \
    --denoiser lotus \
    --use_conditioning_probability 0.8 \
    --align_depth_least_square \
    "$@"
