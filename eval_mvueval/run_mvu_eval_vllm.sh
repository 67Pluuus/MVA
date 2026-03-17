#!/bin/bash

# ==========================================
# 分布式评测脚本 (接入 vLLM 架构版本)
# ==========================================

# --- 1. vLLM 服务端参数 ---
# NUM_GPUS 控制分配给 vLLM 后端的物理显卡总数 (对应 --tensor-parallel-size)
NUM_GPUS=1
# vLLM 服务运行的端口
VLLM_PORT=8007
# 你的模型绝对路径或 HuggingFace 模型库名称
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct" # ⚠️ 注意：请修改为你实际的模型存放路径

# --- 2. 评测并发客户端参数 ---
# NUM_WORKERS 控制同时向 vLLM 发送请求的 Python 进程数（并发线程/进程）
NUM_WORKERS=2

# --- 3. 评测配置与任务参数 ---
CONFIG_PATH="MVA/eval_mvueval/config_mvueval.yaml"
NUM_NODES=1
NODE_RANK=0

# Extract sub_task from config file
SUB_TASK=$(grep "sub_task:" "${CONFIG_PATH}" | head -n 1 | sed 's/.*sub_task: //g' | tr -d '"' | tr -d "'" | tr -d '\r' | xargs)
if [ -z "${SUB_TASK}" ]; then
    SUB_TASK="default"
fi

echo "--------------------------------------------------------"
echo "【步骤 1】正在后台启动 vLLM API 服务器..."
echo "模型: ${MODEL_PATH}"
echo "分配显卡数 (TP): ${NUM_GPUS}"
echo "监听端口: ${VLLM_PORT}"
echo "--------------------------------------------------------"

# 启动 vLLM 服务器放入后台运行 (&)，保存进程 PID 以便结束时关闭
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --api-key sk-abc123 \
    --served-model-name "${MODEL_PATH}" \
    --tensor-parallel-size ${NUM_GPUS} \
    ----pipeline-parallel-size 1 \
    --port ${VLLM_PORT} \
    --trust-remote-code \
    --host localhost \
    --max-model-len 80000 \
    --allowed-local-media-path "$(pwd)" > vllm_server.log 2>&1 &
    
VLLM_PID=$!

echo "等待 vLLM 服务器初始化权重并准备就绪 (可能需要1-3分钟，请耐心等待)..."
# 循环轮询 vLLM 的 health 端点，直到返回成功才继续往下走
while ! curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null; do
    sleep 5
done
echo "✅ vLLM 服务器已就绪！可以开始并发评测。"

echo "--------------------------------------------------------"
echo "【步骤 2】启动 MVU 评测并发客户端"
echo "配置:   ${CONFIG_PATH}"
echo "子任务: ${SUB_TASK}"
echo "并发数: ${NUM_WORKERS} 个并发 Worker"
echo "--------------------------------------------------------"

# 生成 Worker ID 队列，例如 "0,1,2,3...15" (对应原版代码中的 GPUS 变量)
WORKERS=$(seq -s, 0 $((NUM_WORKERS-1)))

OUTPUT_LOG="mvu_run_output_${SUB_TASK}_${NODE_RANK}.log"

# 运行 Python 并发评测客户端
# 注意：底层的 run_mvu_eval.py 依然使用 --gpus 参数接收，但它现在指的是 Workers 列表
PYTHONPATH=. python MVA/eval_mvueval/run_mvu_eval.py \
    --config "${CONFIG_PATH}" \
    --gpus "${WORKERS}" \
    --num_nodes "${NUM_NODES}" \
    --node_rank "${NODE_RANK}" \
    --model_path "${MODEL_PATH}" 2>&1 | tee "${OUTPUT_LOG}"

RUN_STATUS=${PIPESTATUS[0]}

echo "--------------------------------------------------------"
echo "【步骤 3】评测客户端执行完毕，正在关闭后端的 vLLM 服务器..."
kill ${VLLM_PID}
wait ${VLLM_PID} 2>/dev/null
echo "vLLM 服务器已关闭。"

echo "--------------------------------------------------------"
if [ $RUN_STATUS -eq 0 ]; then
    echo "MVU Evaluation completed successfully on Node ${NODE_RANK}."
    
    # Extract output file path
    PREDICT_FILE=$(grep "OUTPUT_FILE: " "${OUTPUT_LOG}" | tail -n 1 | sed 's/OUTPUT_FILE: //g' | tr -d '\r')
    
    if [ -n "${PREDICT_FILE}" ]; then
        echo "找到输出文件: ${PREDICT_FILE}"
        echo "--------------------------------------------------------"
        echo "正在运行指标评估脚本 (Evaluation)..."
        python MVA/eval_mvueval/evaluate_mvu_eval.py --predict "${PREDICT_FILE}"
    else
        echo "Warning: Could not find output file path in logs. Evaluation skipped."
    fi
else
    echo "❌ MVU Evaluation FAILED on Node ${NODE_RANK}."
fi
echo "--------------------------------------------------------"

# Clean up temporary log file
rm "${OUTPUT_LOG}"