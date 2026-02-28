#!/bin/bash

# Default parameters for MVU Evaluation Distributed Run

# Set your available GPU count per node
# e.g., if you have 8 GPUs on this machine, set NUM_GPUS=8
NUM_GPUS=1

# Configuration file
CONFIG_PATH="MVA/eval_vidic/config_vidic.yaml"

# Control printing of model I/O. Set to "true" to enable printing of prompt, image paths, and model output.
# You can override by passing a first argument to the script, or by exporting PRINT_MODEL_IO in the environment.
PRINT_MODEL_IO=false
if [ -n "$1" ]; then
    PRINT_MODEL_IO=$1
fi
export PRINT_MODEL_IO

# Optional: number of tasks (can be set via env or second positional arg)
# Example: ./0.sh true 10  -> sets PRINT_MODEL_IO=true and NUM_TASK=10
NUM_TASK=1
if [ -n "$2" ]; then
    NUM_TASK=$2
fi
export NUM_TASK

# Multi-node settings (Default is single node 1/1)
# Total number of nodes involved in the evaluation
NUM_NODES=1
# The rank of the current node (0, 1, 2...)
NODE_RANK=0

# Generate GPU list string (e.g., "0,1,2,3" if NUM_GPUS=4)
GPUS=$(seq -s, 0 $((NUM_GPUS-1))) # 默认使用所有gpu, 如果需要指定gpu, 可以修改为 GPUS="0,1" 这样的格式

echo "--------------------------------------------------------"
echo "Starting ViDiC-1K Evaluation"
echo "--------------------------------------------------------"
echo "Config: ${CONFIG_PATH}"
echo "Node:   ${NODE_RANK}"
echo "GPUs:   ${GPUS}"
echo "NUM_TASK: ${NUM_TASK}"
echo "--------------------------------------------------------"

# Run the python script with the distributed parameters
# PYTHONPATH=. ensures that the script can import modules from the project root
# Create temp file to capture output for extraction
OUTPUT_LOG="vidic_run_output.log"

PYTHONPATH=. python MVA/eval_vidic/run_vidic.py \
    --config "${CONFIG_PATH}" \
    --gpus "${GPUS}" \
    --num_nodes "${NUM_NODES}" \
    --node_rank "${NODE_RANK}" \
    --num_task "${NUM_TASK}" 2>&1 | tee "${OUTPUT_LOG}"

# Capture the exit status of the python command (first command in pipe)
RUN_STATUS=${PIPESTATUS[0]}

if [ $RUN_STATUS -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "ViDiC-1K completed successfully on Node ${NODE_RANK}."
    
    # Extract output file path
    # Look for line "OUTPUT_FILE: /path/to/file.json"
    PREDICT_FILE=$(grep "OUTPUT_FILE: " "${OUTPUT_LOG}" | tail -n 1 | sed 's/OUTPUT_FILE: //g' | tr -d '\r')
    
    if [ -n "${PREDICT_FILE}" ]; then
        echo "Found output file: ${PREDICT_FILE}"
        echo "--------------------------------------------------------"
        echo "Starting Evaluation..."
        # Run evaluation
        python MVA/eval_vidic/evaluate_vidic.py --predict "${PREDICT_FILE}"
    else
        echo "Warning: Could not find output file path in logs. Evaluation skipped."
    fi
    echo "--------------------------------------------------------"
else
    echo "--------------------------------------------------------"
    echo "ViDiC-1K FAILED on Node ${NODE_RANK}."
    echo "--------------------------------------------------------"
fi

# Clean up temporary log file
rm "${OUTPUT_LOG}"
