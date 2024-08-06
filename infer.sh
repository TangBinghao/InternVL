torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=12345 \
    /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/internvl_pairwise/internvl_chat/eval/evaluate_search.py \
    --checkpoint /mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/code/swift_v1/output_align_epoch1_sft/internvl2-2b/v0-20240720-145652/checkpoint-1386 \
