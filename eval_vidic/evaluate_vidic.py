import json
import argparse
import sys
import os

# 默认路径
DEFAULT_PREDICT = "eval_vidic/node_0_final_results.json"
DEFAULT_GT = "Benchmark/ViDiC-1K/test/metadata.json"

def main():
    parser = argparse.ArgumentParser(description="Evaluate ViDiC results")
    parser.add_argument("--predict", type=str, default=DEFAULT_PREDICT, help="Path to prediction JSON file")
    parser.add_argument("--gt", type=str, default=DEFAULT_GT, help="Path to ground truth JSON file")
    
    args = parser.parse_args()
    
    predict = args.predict
    gt = args.gt
    
    print(f"Loading predictions from: {predict}")
    print(f"Loading ground truth from: {gt}")

    if not os.path.exists(predict):
         print(f"Error: Prediction file not found: {predict}")
         return

    if not os.path.exists(gt):
         print(f"Error: Ground truth file not found: {gt}")
         return

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

    # --- 输出结果 (Format matching the image) ---

    # Calculate metrics
    avg_acc = total_correct / total_count if total_count > 0 else 0

    # Task metrics: Difference, Similarity
    # Note: In the file keys are usually "Difference-..." or "Similarity-..."
    # We split them into `task` and `clas` above.
    # Expecting task keys: "Difference", "Similarity"
    
    diff_stats = task_stats.get("Difference", {'correct': 0, 'total': 0})
    diff_acc = diff_stats['correct'] / diff_stats['total'] if diff_stats['total'] > 0 else 0
    
    sim_stats = task_stats.get("Similarity", {'correct': 0, 'total': 0})
    sim_acc = sim_stats['correct'] / sim_stats['total'] if sim_stats['total'] > 0 else 0

    # Class metrics mapping
    # Image headers: Subject | Motion | Pos. | Backgr. | Cam. | Style | Tech.
    # JSON keys:    subject | motion | position | background | camera | style | playback technique
    
    class_mapping = {
        "Subject": "subject",
        "Motion": "motion",
        "Pos.": "position",
        "Backgr.": "background",
        "Cam.": "camera",
        "Style": "style",
        "Tech.": "playback technique"
    }
    
    class_accs = {}
    for header, key in class_mapping.items():
        stats = clas_stats.get(key, {'correct': 0, 'total': 0})
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        class_accs[header] = acc

    # Construct the table
    # Columns: Model | Avg. | Diff. | Sim. | Subject | Motion | Pos. | Backgr. | Cam. | Style | Tech.
    
    headers = ["Model", "Avg.", "Diff.", "Sim."] + list(class_mapping.keys())
    
    # We don't have a model name effectively, providing a placeholder or empty string
    model_name = "Model" 
    
    # Format percentages
    def fmt(val):
        return f"{val*100:.1f}"

    row_val = [
        model_name,
        fmt(avg_acc),
        fmt(diff_acc),
        fmt(sim_acc),
    ]
    for header in class_mapping.keys():
        row_val.append(fmt(class_accs[header]))

    # Print nicely formatted pipe-separated table
    
    # Calculate widths for alignment (optional but nice)
    # Just printing standard Markdown/Pipe table format
    
    # Calculate column widths
    col_widths = []
    
    # Combine headers and values for width calculation
    # Only one row of data plus header
    
    all_rows = [headers, row_val]
    
    num_cols = len(headers)
    
    for i in range(num_cols):
        max_w = 0
        for row in all_rows:
            w = len(str(row[i]))
            if w > max_w:
                max_w = w
        col_widths.append(max_w)
    
    # Function to format a row
    def format_row(row, widths):
        parts = []
        for i, val in enumerate(row):
            parts.append(str(val).ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"
        
    print("\n" + "="*80)
    print("--- ViDiC Evaluation Results ---")
    print("="*80)
    
    header_line = format_row(headers, col_widths)
    separator_line = "| " + " | ".join(["-" * w for w in col_widths]) + " |"
    data_line = format_row(row_val, col_widths)
    
    print("-" * len(header_line))
    print(header_line)
    print(separator_line)
    print(data_line)
    print("-" * len(header_line))
    print("\n")


if __name__ == "__main__":
    main()
