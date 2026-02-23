#!/bin/bash

# Default parameters for MVU Evaluation Distributed Run

# Set your available GPU count per node
# e.g., if you have 8 GPUs on this machine, set NUM_GPUS=8
NUM_GPUS=1

# Configuration file
CONFIG_PATH="eval_mvueval/config_mvueval.yaml"

# Multi-node settings (Default is single node 1/1)
# Total number of nodes involved in the evaluation
NUM_NODES=1
# The rank of the current node (0, 1, 2...)
NODE_RANK=0

# Generate GPU list string (e.g., "0,1,2,3" if NUM_GPUS=4)
GPUS=$(seq -s, 0 $((NUM_GPUS-1))) # 默认使用所有gpu, 如果需要指定gpu, 可以修改为 GPUS="0,1" 这样的格式

echo "--------------------------------------------------------"
echo "Starting MVU Evaluation (Distributed Mode)"
echo "--------------------------------------------------------"
echo "Config: ${CONFIG_PATH}"
echo "Node:   ${NODE_RANK} / ${NUM_NODES}"
echo "GPUs:   ${GPUS}"
echo "--------------------------------------------------------"

# Run the python script with the distributed parameters
# PYTHONPATH=. ensures that the script can import modules from the project root
PYTHONPATH=. python eval_mvueval/run_mvu_eval.py \
    --config "${CONFIG_PATH}" \
    --gpus "${GPUS}" \
    --num_nodes "${NUM_NODES}" \
    --node_rank "${NODE_RANK}"

if [ $? -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "MVU Evaluation completed successfully on Node ${NODE_RANK}."
    echo "--------------------------------------------------------"
else
    echo "--------------------------------------------------------"
    echo "MVU Evaluation FAILED on Node ${NODE_RANK}."
    echo "--------------------------------------------------------"
fi
