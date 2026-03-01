import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen3-VL-2B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 请根据实际情况修改文件名，这里假设预测结果是 1.json，真值文件是 2.json
predict = ""
gt = r"Benchmark/CrossVid/QA/CCQA.json"

with open(predict, 'r', encoding='utf-8') as f:
    pred_js = json.load(f)

with open(gt, 'r', encoding='utf-8') as f:
    gt_js = json.load(f)

def extract_score(text):
    """从包含 <score> 标签的文本中提取并解析 JSON"""
    match = re.search(r'<score>(.*?)</score>', text, re.DOTALL)
    if not match:
        return None
    
    json_str = match.group(1).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

SCORE = """
You are asked to score the output of a model, given the following information:
- Question: {QUESTION}
- Standard Answer: {ANSWER}
- Scoring Points: {POINTS}
- Model's Output: {OUTPUT}

Please perform the following two-part scoring:
Part 1: Coverage of Scoring Points
- For each scoring point, determine whether it is covered by the Model's Output.
- Mark as covered (true) **only if** the scoring point is addressed **explicitly, independently, and clearly**.
- If the mention is vague, partial, or ambiguous, consider it **not covered**.

Part 2: Accuracy of Details
- For each covered scoring point, compare the details in the Model's Output to the Standard Answer.
- Mark as correct (true) **only if** the details are **fully accurate and consistent** with the Standard Answer, without any error, omission, or ambiguity.
- If the answer is partially correct, too broad/narrow, or not strictly consistent, mark it as **not correct** (false).
- For scoring points not covered, mark as incorrect.

Format your answer in a json format as follows:
{{
    "coverage": [true, false, true, ...],
    "correctness": [true, false, false, ...]
}}
The length of 'coverage' and 'correctness' lists should match the number of scoring points.
Wrap the json output within <score></score> tags.

Your answer:
"""

# 用于存储各分类的统计数据
# 结构: {'task_name': {'score_sum': 0, 'max_score': 0}}
task_stats = {}

total_score_sum = 0
total_max_score = 0

for item in pred_js:
    key = str(item['key']) # 确保 key 是字符串，因为 json 的键通常是字符串
    
    if key not in gt_js:
        print(f"Warning: Key {key} not found in ground truth.")
        continue

    gt_item = gt_js[key]
    
    # 获取任务类型和真值
    task = gt_item.get('task', 'Unknown')
    
    question = item.get('question', '')
    gt_ans = gt_item.get('standard_answer', '').strip()
    pred_ans = item['predicted_answer'].strip()
    scoring_points_list = gt_item.get('scoring_points', [])
    scoring_points = '\n'.join(scoring_points_list)

    prompt = SCORE.replace("{QUESTION}", question).replace("{ANSWER}", gt_ans).replace("{POINTS}", scoring_points).replace("{OUTPUT}", pred_ans)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
    )

    output_ids = generated_ids[0][len(model.inputs_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"--- Scoring for key: {key} ---")
    print("Model Output:", pred_ans)
    print("Scorer Response:", content)

    # 初始化该任务的统计
    if task not in task_stats:
        task_stats[task] = {'score_sum': 0, 'max_score': 0}
    
    # 计分
    max_score_for_item = len(scoring_points_list)
    task_stats[task]['max_score'] += max_score_for_item
    total_max_score += max_score_for_item

    score_data = extract_score(content)
    if score_data and 'correctness' in score_data and isinstance(score_data['correctness'], list):
        # 得分是 'correctness' 列表中 true 的数量
        current_score = sum(1 for v in score_data['correctness'] if v is True)
        task_stats[task]['score_sum'] += current_score
        total_score_sum += current_score
        print(f"Score: {current_score}/{max_score_for_item}")
    else:
        print(f"Score: 0/{max_score_for_item} (Could not parse scorer response)")
    print("-" * (20 + len(key)))


# 输出结果
sorted_tasks = sorted(task_stats.keys())

# 计算总体准确率
overall_acc = total_score_sum / total_max_score if total_max_score > 0 else 0

# 构造表头 (第一行)
header = ["Overall"] + sorted_tasks
print(", ".join(header))

# 构造准确率行 (第二行)
accuracies = [f"{overall_acc:.2%}"]
for task in sorted_tasks:
    stats = task_stats[task]
    acc = stats['score_sum'] / stats['max_score'] if stats['max_score'] > 0 else 0
    accuracies.append(f"{acc:.2%}")

print(", ".join(accuracies))
