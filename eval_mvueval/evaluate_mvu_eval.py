import json

# 请根据实际情况修改文件名，这里假设预测结果是 1.json，真值文件是 2.json
predict = r"D:\Desktop\毕设\Agent\eval_mvueval\node_0_final_results(1).json"
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
    gt_choice = gt_item.get('ground_truth')
    pred_choice = item.get('predicted_answer')
    
    # 初始化该任务的统计
    if task not in task_stats:
        task_stats[task] = {'correct': 0, 'total': 0}
    
    #哪怕预测没有结果，也要计入总数
    task_stats[task]['total'] += 1
    total_count += 1
    
    # 判断是否正确 (假设由单个字母组成，如 "A", "B")
    # 清理一下可能的空白字符，并统一转大写比较
    if pred_choice and gt_choice and str(pred_choice).strip().upper().startswith(str(gt_choice).strip().upper()):
        task_stats[task]['correct'] += 1
        total_correct += 1

# --- 输出结果 ---

def print_as_table(headers, data_rows):
    """手动打印表格，不使用tabulate"""
    if not data_rows:
        print("No data to display.")
        return

    # 计算每列的最大宽度
    num_columns = len(headers)
    col_widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            if len(str(cell)) > col_widths[i]:
                col_widths[i] = len(str(cell))

    # 打印表头
    header_line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(num_columns))
    print(header_line)

    # 打印分隔线
    separator_line = "-+-".join("-" * col_widths[i] for i in range(num_columns))
    print(separator_line)

    # 打印数据行
    for row in data_rows:
        data_line = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(num_columns))
        print(data_line)

# 准备表格数据
table_data = []
sorted_tasks = sorted(task_stats.keys())

for task in sorted_tasks:
    stats = task_stats[task]
    correct = stats['correct']
    total = stats['total']
    accuracy = correct / total if total > 0 else 0
    table_data.append([task, correct, total, f"{accuracy:.2%}"])

# 打印按 Task 分类的结果表格
print("--- Accuracy by Task ---")
print_as_table(["Task", "Correct", "Total", "Accuracy"], table_data)
print("\n")

# 打印总体正确率
overall_acc = total_correct / total_count if total_count > 0 else 0
print("--- Overall Accuracy ---")
print(f"{overall_acc:.2%}")
