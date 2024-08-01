set -x

# 设置环境变量
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/output_internvl2_2b_pretrained/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full_pairwise'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# 设置多机多卡训练参数
NNODES=2  # 机器数量
NODE_RANK=${NODE_RANK:-0}  # 当前机器的rank，0表示主节点，1表示从节点
MASTER_ADDR="29.35.175.37"  # 主节点的IP地址

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/InternVL2-2B" \
  --conv_style "internlm2-chat-pairwise" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_pair/person_train_mllm_swift_pairwise.jsonl" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_custom_trainer True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/zero_stage3_config.json" \
  --report_to "none" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"