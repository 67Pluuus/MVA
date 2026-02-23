import json

# 请根据实际情况修改文件名，这里假设预测结果是 1.json，真值文件是 2.json
predict = r"D:\Desktop\毕设\Agent\eval_vidic\node_0_final_results.json"
gt = r"D:\Desktop\毕设\Agent\Benchmark\ViDiC-1K\test\metadata.json"

with open(predict, 'r', encoding='utf-8') as f:
    pred_js = json.load(f)

with open(gt, 'r', encoding='utf-8') as f:
    gt_js = json.load(f)

# 用于存储各分类的统计数据
# 结构: {'task_name': {'correct': 0, 'total': 0}}
task_stats = {}
# 新增 clas_stats 用于按 clas 分类统计
clas_stats = {}

total_correct = 0
total_count = 0

for item in pred_js:
    if item.get('success') == False:
        continue
    key = str(item['key']) # 确保 key 是字符串，因为 json 的键通常是字符串
    
    if key not in gt_js:
        print(f"Warning: Key {key} not found in ground truth.")
        continue

    gt_item = gt_js[key]
    
    # 获取任务类型和类别
    try:
        task, clas = gt_item.get('task', 'Unknown-Unknown').split('-', 1)
    except ValueError:
        task = gt_item.get('task', 'Unknown')
        clas = 'Unknown'

    gt_choice = gt_item.get('ground_truth')
    pred_choice = item.get('predicted_answer')
    
    # 初始化该任务的统计
    if task not in task_stats:
        task_stats[task] = {'correct': 0, 'total': 0}
    
    # 初始化该类别的统计
    if clas not in clas_stats:
        clas_stats[clas] = {'correct': 0, 'total': 0}
    
    # 哪怕预测没有结果，也要计入总数
    task_stats[task]['total'] += 1
    clas_stats[clas]['total'] += 1
    total_count += 1
    
    # 判断是否正确 (假设由单个字母组成，如 "A", "B")
    # 清理一下可能的空白字符，并统一转大写比较
    if pred_choice and gt_choice and str(pred_choice).strip().upper().startswith(str(gt_choice).strip().upper()):
        task_stats[task]['correct'] += 1
        clas_stats[clas]['correct'] += 1
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

# 1. 输出总体正确率
overall_acc = total_correct / total_count if total_count > 0 else 0
print("--- Overall Accuracy ---")
print(f"{overall_acc:.2%}\n")


# 2. 输出按 Task 分类的结果
print("--- Accuracy by Task ---")
sorted_tasks = sorted(task_stats.keys())
task_table_data = []
for task in sorted_tasks:
    stats = task_stats[task]
    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    task_table_data.append([task, stats['correct'], stats['total'], f"{acc:.2%}"])
print_as_table(["Task", "Correct", "Total", "Accuracy"], task_table_data)
print("\n")


# 3. 输出按 Clas 分类的结果
print("--- Accuracy by Clas ---")
sorted_clas = sorted(clas_stats.keys())
clas_table_data = []
for c in sorted_clas:
    stats = clas_stats[c]
    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    clas_table_data.append([c, stats['correct'], stats['total'], f"{acc:.2%}"])
print_as_table(["Clas", "Correct", "Total", "Accuracy"], clas_table_data)
