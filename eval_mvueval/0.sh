#!/bin/bash

# Default parameters for MVU Evaluation Distributed Run

# Set your available GPU count per node
# e.g., if you have 8 GPUs on this machine, set NUM_GPUS=8
NUM_GPUS=1

# Configuration file
CONFIG_PATH="MVA/eval_mvueval/0.yaml"

# Multi-node settings (Default is single node 1/1)
# Total number of nodes involved in the evaluation
NUM_NODES=1
# The rank of the current node (0, 1, 2...)
NODE_RANK=0

# Generate GPU list string (e.g., "0,1,2,3" if NUM_GPUS=4)
GPUS=$(seq -s, 0 $((NUM_GPUS-1))) # 默认使用所有gpu, 如果需要指定gpu, 可以修改为 GPUS="0,1" 这样的格式

echo "--------------------------------------------------------"
echo "Starting MVU Evaluation"
echo "--------------------------------------------------------"
echo "Config: ${CONFIG_PATH}"
echo "Node:   ${NODE_RANK} / ${NUM_NODES}"
echo "GPUs:   ${GPUS}"
echo "--------------------------------------------------------"

# Run the python script with the distributed parameters
# PYTHONPATH=. ensures that the script can import modules from the project root
# Create temp file to capture output for extraction
OUTPUT_LOG="mvu_run_output.log"

PYTHONPATH=. python MVA/eval_mvueval/run_mvu_eval.py \
    --config "${CONFIG_PATH}" \
    --gpus "${GPUS}" \
    --num_nodes "${NUM_NODES}" \
    --node_rank "${NODE_RANK}" 2>&1 | tee "${OUTPUT_LOG}"

# Capture the exit status of the python command (first command in pipe)
RUN_STATUS=${PIPESTATUS[0]}

if [ $RUN_STATUS -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "MVU Evaluation completed successfully on Node ${NODE_RANK}."
    
    # Extract output file path
    # Look for line "OUTPUT_FILE: /path/to/file.json"
    PREDICT_FILE=$(grep "OUTPUT_FILE: " "${OUTPUT_LOG}" | tail -n 1 | sed 's/OUTPUT_FILE: //g' | tr -d '\r')
    
    if [ -n "${PREDICT_FILE}" ]; then
        echo "Found output file: ${PREDICT_FILE}"
        echo "--------------------------------------------------------"
        echo "Starting Evaluation..."
        # Run evaluation
        python MVA/eval_mvueval/evaluate_mvu_eval.py --predict "${PREDICT_FILE}"
    else
        echo "Warning: Could not find output file path in logs. Evaluation skipped."
    fi
    echo "--------------------------------------------------------"
else
    echo "--------------------------------------------------------"
    echo "MVU Evaluation FAILED on Node ${NODE_RANK}."
    echo "--------------------------------------------------------"
fi

# Clean up temporary log file
rm "${OUTPUT_LOG}"
