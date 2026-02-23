import json

# 请根据实际情况修改文件名
predict = r"D:\Desktop\毕设\Agent\eval_mvueval\result.json"
gt = r"D:\Desktop\毕设\Agent\Benchmark\MVU-Eval_Data\MVU_Eval_QAs.json"

with open(predict, 'r', encoding='utf-8') as f:
    pred_js = json.load(f)

with open(gt, 'r', encoding='utf-8') as f:
    gt_js = json.load(f)

def parse_interval(interval_str):
    """将 'start-end' 格式的字符串解析为 (float, float) 元组"""
    if not isinstance(interval_str, str):
        return None
    parts = interval_str.split('-')
    if len(parts) == 2:
        try:
            start = float(parts[0])
            end = float(parts[1])
            return min(start, end), max(start, end)
        except (ValueError, TypeError):
            return None
    return None

def calculate_iou(interval1, interval2):
    """计算两个一维区间的 IoU"""
    start1, end1 = interval1
    start2, end2 = interval2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_length = max(0, intersection_end - intersection_start)
    if intersection_length == 0:
        return 0.0
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_length = union_end - union_start
    if union_length == 0:
        return 0.0
    return intersection_length / union_length

# 用于存储各分类的统计数据
# 结构: {'task_name': {'ious': [], 'total': 0}}
task_stats = {}
total_ious = []

for item in pred_js:
    key = str(item['key'])
    
    if key not in gt_js:
        print(f"Warning: Key {key} not found in ground truth.")
        continue

    gt_item = gt_js[key]
    
    task = gt_item.get('task', 'Unknown')
    gt_choice = gt_item.get('ground_truth')
    pred_choice = item.get('predicted_answer')
    
    if task not in task_stats:
        task_stats[task] = {'ious': [], 'total': 0}
    
    task_stats[task]['total'] += 1
    
    gt_interval = parse_interval(gt_choice)
    pred_interval = parse_interval(pred_choice)
    
    current_iou = 0.0
    if gt_interval and pred_interval:
        current_iou = calculate_iou(gt_interval, pred_interval)
    
    task_stats[task]['ious'].append(current_iou)
    total_ious.append(current_iou)

# --- 输出结果 ---

def print_as_table(headers, data_rows, add_separator_before_last=False):
    """手动打印表格，不使用tabulate"""
    if not data_rows:
        print("No data to display.")
        return
    num_columns = len(headers)
    col_widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            if len(str(cell)) > col_widths[i]:
                col_widths[i] = len(str(cell))
    header_line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(num_columns))
    print(header_line)
    separator_line = "-+-".join("-" * col_widths[i] for i in range(num_columns))
    print(separator_line)
    for i, row in enumerate(data_rows):
        if add_separator_before_last and i == len(data_rows) - 1:
            print(separator_line)
        data_line = " | ".join(str(row[j]).ljust(col_widths[j]) for j in range(num_columns))
        print(data_line)

# 准备表格数据
table_data = []
sorted_tasks = sorted(task_stats.keys())

for task in sorted_tasks:
    stats = task_stats[task]
    total = stats['total']
    avg_iou = sum(stats['ious']) / total if total > 0 else 0
    table_data.append([task, total, f"{avg_iou:.4f}"])

# 计算总体平均IoU并添加到表格数据中
overall_avg_iou = sum(total_ious) / len(total_ious) if total_ious else 0
table_data.append(["Overall", len(total_ious), f"{overall_avg_iou:.4f}"])

# 打印结果表格
print("--- Average IoU by Task ---")
print_as_table(["Task", "Total", "Avg IoU"], table_data, add_separator_before_last=True)
