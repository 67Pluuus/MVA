import json

# 请根据实际情况修改文件名，这里假设预测结果是 1.json，真值文件是 2.json
predict = r"D:\Desktop\毕设\Agent\eval_mvueval\result.json"
gt = r"D:\Desktop\毕设\Agent\Benchmark\MVU-Eval_Data\MVU_Eval_QAs.json"

with open(predict, 'r', encoding='utf-8') as f:
    pred_js = json.load(f)

with open(gt, 'r', encoding='utf-8') as f:
    gt_js = json.load(f)

# 用于存储各分类的统计数据
# 结构: {'task_name': {'correct': 0, 'total': 0}}
task_stats = {}

total_correct = 0
total_count = 0

for item in pred_js:
    key = str(item['key']) # 确保 key 是字符串，因为 json 的键通常是字符串
    
    if key not in gt_js:
        print(f"Warning: Key {key} not found in ground truth.")
        continue

    gt_item = gt_js[key]
    
    # 获取任务类型和真值
    task = gt_item.get('task', 'Unknown')
    gt_choice = gt_item['ground_truth']
    pred_choice = item['predicted_answer']
    
    # 初始化该任务的统计
    if task not in task_stats:
        task_stats[task] = {'correct': 0, 'total': 0}
    
    #哪怕预测没有结果，也要计入总数
    task_stats[task]['total'] += 1
    total_count += 1
    
    # 判断是否正确 (多选，精确匹配)
    # 清理空白字符，并将字符排序后比较
    pred_sorted = "".join(sorted(list(pred_choice.strip().upper())))
    gt_sorted = "".join(sorted(list(gt_choice.strip().upper())))

    if pred_sorted and gt_sorted and pred_sorted == gt_sorted:
        task_stats[task]['correct'] += 1
        total_correct += 1

# 输出结果
sorted_tasks = sorted(task_stats.keys())

# 计算总体准确率
overall_acc = total_correct / total_count if total_count > 0 else 0

# 构造表头 (第一行)
header = ["Overall"] + sorted_tasks
print(", ".join(header))

# 构造准确率行 (第二行)
accuracies = [f"{overall_acc:.2%}"]
for task in sorted_tasks:
    stats = task_stats[task]
    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    accuracies.append(f"{acc:.2%}")

print(", ".join(accuracies))
