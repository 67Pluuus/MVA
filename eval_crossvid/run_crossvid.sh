#!/bin/bash

# Default parameters for CrossVid Evaluation
NUM_GPUS=8

# Tasks to evaluate
tasks=("BU" "NC" "CC" "PEA")

# Multi-node settings
NUM_NODES=1
NODE_RANK=0

GPUS=$(seq -s, 0 $((NUM_GPUS-1)))

echo "--------------------------------------------------------"
echo "Starting CrossVid Evaluation Batch"
echo "Tasks: ${tasks[*]}"
echo "--------------------------------------------------------"

for task in "${tasks[@]}"; do
    CONFIG_PATH="eval_crossvid/${task}.yaml"
    OUTPUT_LOG="crossvid_${task}_run_output.log"
    
    echo "Processing Task: ${task}"
    echo "Config: ${CONFIG_PATH}"
    
    # Run the python script
    PYTHONPATH=. python eval_crossvid/run_crossvid.py \
        --config "${CONFIG_PATH}" \
        --gpus "${GPUS}" \
        --num_nodes "${NUM_NODES}" \
        --node_rank "${NODE_RANK}" 2>&1 | tee "${OUTPUT_LOG}"
    
    RUN_STATUS=${PIPESTATUS[0]}
    
    if [ $RUN_STATUS -eq 0 ]; then
        echo "Run completed successfully for ${task}. Starting Evaluation..."
        
        # Extract output file path
        PREDICT_FILE=$(grep "OUTPUT_FILE: " "${OUTPUT_LOG}" | tail -n 1 | sed 's/OUTPUT_FILE: //g' | tr -d '\r')
        
        # Extract GT file path from config (Simple grep)
        # Assumes format: json_file: "path/to/file.json"
        GT_FILE=$(grep "json_file:" "${CONFIG_PATH}" | head -n 1 | sed 's/.*json_file: //g' | tr -d '"' | tr -d "'" | tr -d '\r' | xargs)
        
        if [ -n "${PREDICT_FILE}" ]; then
             if [ -n "${GT_FILE}" ]; then
                echo "Predict File: ${PREDICT_FILE}"
                echo "GT File: ${GT_FILE}"
                
                # Check if evaluate script exists
                EVAL_SCRIPT="eval_crossvid/evaluate_${task}.py"
                if [ -f "${EVAL_SCRIPT}" ]; then
                    python "${EVAL_SCRIPT}" --predict "${PREDICT_FILE}" --gt "${GT_FILE}"
                else
                    echo "Error: Evaluation script not found: ${EVAL_SCRIPT}"
                fi
             else
                echo "Warning: Could not extract GT file path from config. Trying default..."
                python "eval_crossvid/evaluate_${task}.py" --predict "${PREDICT_FILE}"
             fi
        else
            echo "Warning: Could not find output file path in logs. Evaluation skipped for ${task}."
        fi
    else
        echo "Run FAILED for ${task}."
    fi
    
    echo "--------------------------------------------------------"
    # Clean up log
    rm "${OUTPUT_LOG}"
done

echo "Batch Processing Complete."
