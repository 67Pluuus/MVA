import json
import argparse
import sys
import os

# 默认路径
DEFAULT_PREDICT = r"eval_crossvid\node_0_final_results.json"
DEFAULT_GT = r"EBenchmark\CrossVid\QA\BU_1.json"

def main():
    parser = argparse.ArgumentParser(description="Evaluate BU results")
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
         # Try fallback to BU_1.json if BU.json fails, just in case
         gt_alt = gt.replace("BU.json", "BU_1.json")
         if os.path.exists(gt_alt):
             gt = gt_alt
             print(f"Ground truth file found at alternative path: {gt}")
         else:
             print(f"Error: Ground truth file not found: {gt}")
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
        gt_choice = gt_item['ground_truth']
        pred_choice = item.get('predicted_answer') # Use .get() to avoid KeyError

        
        # 初始化该任务的统计
        if task not in task_stats:
            task_stats[task] = {'correct': 0, 'total': 0}
        
        #哪怕预测没有结果，也要计入总数
        task_stats[task]['total'] += 1
        total_count += 1
        
        if not pred_choice:
             continue

        # 判断是否正确 (多选，精确匹配)
        # 清理空白字符，并将字符排序后比较
        pred_sorted = "".join(sorted(list(str(pred_choice).strip().upper())))
        gt_sorted = "".join(sorted(list(str(gt_choice).strip().upper())))

        if pred_sorted and gt_sorted and pred_sorted == gt_sorted:
            task_stats[task]['correct'] += 1
            total_correct += 1
            
    # --- 输出结果 ---
    print(f"--- BU Evaluation Results ---")
    print(f"Total Count: {total_count}")
    print(f"Total Correct: {total_correct}")
    
    # 计算总体准确率
    overall_acc = total_correct / total_count if total_count > 0 else 0
    if total_count > 0:
        print(f"Overall Accuracy: {overall_acc:.2%}")
    else:
        print(f"Overall Accuracy: 0.00%")

    sorted_tasks = sorted(task_stats.keys())

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

if __name__ == "__main__":
    main()
