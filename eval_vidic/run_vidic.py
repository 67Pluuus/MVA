import json
import os
import shutil
import argparse
import sys
import multiprocessing as mp
import math
from tqdm import tqdm

# Add root directory to path to import agent_runner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_runner import AgentRunner
import yaml

# Configuration
CONFIG_PATH = 'eval_vidic/config_vidic.yaml'

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def process_chunk(args):
    """
    Process a chunk of tasks on a specific GPU
    """
    gpu_id, tasks, config_path, node_rank, run_id = args
    
    # Initialize runner for this process/GPU with run_id
    runner = AgentRunner(config_path, device_id=gpu_id, node_rank=node_rank, run_id=run_id)
    video_base_dir = runner.config['paths']['video_base_dir']
    
    results = []
    for key, item in tqdm(tasks, desc=f"GPU {gpu_id}", position=gpu_id, leave=False):
        try:
            # Pass key info to runner
            item_with_key = item.copy()
            item_with_key['key'] = key
            
            # Run runner
            result_data = runner.run_on_sample(item_with_key, video_base_dir)
            results.append(result_data)
            
        except Exception as e:
            # print(f"Error processing sample {key}: {e}")
            err_entry = {"key": key, "error": str(e), "success": False}
            results.append(err_entry)
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_PATH)
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")')
    parser.add_argument('--num_nodes', type=int, default=1, help='Total number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of current node')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Create Run ID and Log Folder
    import textwrap
    import datetime
    
    # Get params
    benchmark_name = config.get('paths', {}).get('Benchmark_name', 'ViDiC')
    type_watermark = config.get('parameters', {}).get('type_watermark', 'default').replace(' ', '_')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Format: {Benchmark_name}_{date}_{time}_{type_watermark}
    run_id = f"{benchmark_name}_{timestamp}_{type_watermark}"
    
    # Base log directory
    log_dir = os.path.join("log", run_id)
    os.makedirs(log_dir, exist_ok=True)
    
    # Copy config file to log directory
    shutil.copy(args.config, log_dir)
    
    # Parse GPU list
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    num_gpus = len(gpu_list)
    
    print(f"--- ViDiC Eval ---")
    print(f"Run ID: {run_id}")
    print(f"Log Directory: {log_dir}")
    print(f"Node: {args.node_rank}/{args.num_nodes - 1}")
    print(f"GPUs available: {gpu_list}")
    
    # output_dir is overridden by log_dir logic in AgentRunner, but main needs it for final save
    final_output_dir = log_dir
    
    json_file = config['paths']['json_file']
    # output_dir = config['paths']['output_dir']
    
    # Load dataset
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Get all keys sorted
    all_keys = list(data.keys())
    # Sort to ensure consistent order across nodes
    all_keys.sort(key=lambda x: int(x) if x.isdigit() else x) 

    # Apply num_tasks limit if set
    num_tasks = config.get('parameters', {}).get('num_tasks', -1)
    if num_tasks != -1 and num_tasks < len(all_keys):
        print(f"Limiting tasks to first {num_tasks} (from config)")
        all_keys = all_keys[:num_tasks]
    
    # --- Split Logic ---
    # 1. Split for Nodes
    tasks_total = len(all_keys)
    chunk_size_node = math.ceil(tasks_total / args.num_nodes)
    start_node = args.node_rank * chunk_size_node
    end_node = min((args.node_rank + 1) * chunk_size_node, tasks_total)
    
    node_keys_all = all_keys[start_node:end_node]
    print(f"Total tasks: {tasks_total}")
    print(f"Tasks assigned to this node (Rank {args.node_rank}): {len(node_keys_all)} (Indices {start_node}-{end_node})")

    if not node_keys_all:
        print("No tasks assigned to this node.")
        return

    # 2. Split for GPUs within this node
    tasks_per_gpu = {gid: [] for gid in gpu_list}
    
    for i, key in enumerate(node_keys_all):
        # Assign round-robin to available GPUs
        target_gpu = gpu_list[i % num_gpus]
        tasks_per_gpu[target_gpu].append((key, data[key]))
        
    # Prepare mp args
    pool_args = []
    for g in gpu_list:
        if tasks_per_gpu[g]:
            pool_args.append((g, tasks_per_gpu[g], args.config, args.node_rank, run_id))
            
    # Run Pool
    print(f"Starting {len(pool_args)} processes...")
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    final_node_results = []
    with mp.Pool(processes=len(pool_args)) as pool:
        gpu_results_list = pool.map(process_chunk, pool_args)
        
        # Flatten results
        for res_list in gpu_results_list:
            final_node_results.extend(res_list)
        
    print("Node processing complete.")

    # Save final results to JSON inside log directory
    final_output_file = os.path.join(final_output_dir, f"node_{args.node_rank}_final_results.json")
    print(f"Saving {len(final_node_results)} results to {final_output_file}...")
    
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_node_results, f, ensure_ascii=False, indent=2)
    
    # Print the final output file path for external scripts to capture
    print(f"OUTPUT_FILE: {os.path.abspath(final_output_file)}")

if __name__ == "__main__":
    main()