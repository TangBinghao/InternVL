#!/bin/bash

# 定义要使用的CUDA设备
export NPROC_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_LAUNCH_BLOCKING=1

# 定义checkpoint的基目录
# CKPT_BASE_DIR="output/minicpm-v-v2-chat/v0-20240621-192023"
CKPT_BASE_DIR="/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/output_internvl2_2b/10w_pair_dataset_internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full_pairwise/checkpoint-800"
# CKPT_BASE_DIR="/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_pair/output/internvl2-2b/v0-20240711-230026"

# 定义其他参数
CUSTOM_VAL_DATASET_PATH="/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift-main/person_test_mllm_swift_v2.jsonl"
VERBOSE="false"
DO_SAMPLE="true"
VAL_DATASET_SAMPLE="-1"
SHOW_DATASET_SAMPLE="-1"
LOAD_DATASET_CONFIG="true"
MAX_NEW_TOKENS=500


CKPT_DIR="${CKPT_BASE_DIR}"
echo "Running inference with checkpoint: ${CKPT_DIR}"

swift infer --ckpt_dir "${CKPT_DIR}" \
            --model_type internvl2-2b \
            --custom_val_dataset_path "${CUSTOM_VAL_DATASET_PATH}" \
            --verbose "${VERBOSE}" \
            --do_sample "${DO_SAMPLE}" \
            --val_dataset_sample "${VAL_DATASET_SAMPLE}" \
            --show_dataset_sample "${SHOW_DATASET_SAMPLE}" \
            --load_dataset_config "${LOAD_DATASET_CONFIG}" \
            --max_new_tokens="${MAX_NEW_TOKENS}"
