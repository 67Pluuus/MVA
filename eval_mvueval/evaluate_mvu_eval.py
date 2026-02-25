import json
import argparse
import sys
import os

# 默认路径
DEFAULT_PREDICT = "eval_mvueval/node_0_final_results(1).json"
DEFAULT_GT = "Benchmark/MVU-Eval_Data/MVU_Eval_QAs.json"

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate MVU Eval results")
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
         # Attempt absolute path relative to workspace root if needed, but let's assume correct path passed
         return

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

    # Prepare accuracy values
    overall_acc = total_correct / total_count if total_count > 0 else 0
    
    # Task specific accuracies
    # The image has: Overall | Perception (OR, SU, Counting, Comparison) | Reasoning (KIR, ICL, RAG, TR)
    # Mapping task names in JSON to these headers.
    # Assuming JSON task names match these abbreviations or full names.
    # If JSON has "Object Recognition", mapping it to "OR".
    # Since I don't know the exact JSON task strings for all of them, I will try to match loosely or use the keys if they match.
    # The user's previous prompt showed "Counting" as a task.
    # I will assume "Counting" -> "Counting".
    # "Comparison" -> "Comparison".
    # For others, I'll check if they exist in stats directly.
    
    # Heuristic mapping based on common dataset naming or just direct usage
    # If the JSON uses "Counting", "Comparison", etc. directly.
    
    # Define the headers we want
    headers_perception = ["OR", "SU", "Counting", "Comparison"]
    headers_reasoning = ["KIR", "ICL", "RAG", "TR"]
    
    # Function to safe get accuracy
    def get_acc(key_list):
        # key_list can be a list of potential keys for one header, e.g. ["OR", "Object Recognition"]
        total_c = 0
        total_t = 0
        found = False
        for key in key_list:
            if key in task_stats:
                total_c += task_stats[key]['correct']
                total_t += task_stats[key]['total']
                found = True
        
        if not found:
            return "N/A"
            
        return total_c / total_t if total_t > 0 else 0

    # Mapping based on likely names (adjust if needed specific to dataset)
    # If the task names in JSON are exactly "OR", "SU", etc., this works.
    # If "Counting" is "Counting", it works.
    
    # Values for the row
    row_vals = []
    
    # Overall
    row_vals.append(f"{overall_acc*100:.1f}")
    
    # Perception
    # OR
    row_vals.append(f"{get_acc(['OR'])*100:.1f}" if get_acc(['OR']) != "N/A" else "-")
    # SU
    row_vals.append(f"{get_acc(['SU'])*100:.1f}" if get_acc(['SU']) != "N/A" else "-")
    # Counting
    row_vals.append(f"{get_acc(['Counting'])*100:.1f}" if get_acc(['Counting']) != "N/A" else "-")
    # Comparison
    row_vals.append(f"{get_acc(['Comparison'])*100:.1f}" if get_acc(['Comparison']) != "N/A" else "-")
    
    # Reasoning
    # KIR
    row_vals.append(f"{get_acc(['KIR'])*100:.1f}" if get_acc(['KIR']) != "N/A" else "-")
    # ICL
    row_vals.append(f"{get_acc(['ICL'])*100:.1f}" if get_acc(['ICL']) != "N/A" else "-")
    # RAG
    row_vals.append(f"{get_acc(['RAG'])*100:.1f}" if get_acc(['RAG']) != "N/A" else "-")
    # TR (Temporal Reasoning?)
    row_vals.append(f"{get_acc(['TR'])*100:.1f}" if get_acc(['TR']) != "N/A" else "-")
    
    # Formatting the table
    # Columns: Overall | OR | SU | Counting | Comparison | KIR | ICL | RAG | TR
    # Groups:          |         Perception              |        Reasoning
    
    # Widths
    w_data = 10
    
    def pad_center(s, w):
        return str(s).center(w)
        
    # Top Header
    # "Overall" spans 1 col (index 0)
    # "Perception" spans 4 cols
    # "Reasoning" spans 4 cols
    
    # Top Header Structure
    # | Overall  |              Perception             |              Reasoning              |
    
    top_str = "|" + pad_center("", w_data) + "|" + pad_center("Perception", w_data*4 + 3) + "|" + pad_center("Reasoning", w_data*4 + 3) + "|"
    
    # Second Header
    sub_headers = ["Overall"] + headers_perception + headers_reasoning
    sub_str = "|" + "|".join([pad_center(h, w_data) for h in sub_headers]) + "|"
    
    # Separator
    sep_str = "-" * len(sub_str)
    
    # Data Row
    data_str = "|" + "|".join([pad_center(v, w_data) for v in row_vals]) + "|"
    
    print("\n" + "="*len(sep_str))
    print(top_str)
    print(sep_str)
    print(sub_str)
    print(sep_str)
    print(data_str)
    print("-" * len(sep_str) + "\n")

if __name__ == "__main__":
    main()
