import os
import sys
import yaml
import json
import time
from typing import Dict, Any, List
import math

# Adjust path to import from root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils import init, answer
from agent import ToolAgent, DescAgent
import cv2
from PIL import Image
import numpy as np


# --- User Customization Section ---
def calculate_priority_score(current_score, acceleration, config):
    """
    Calculate a priority score for a video to determine if it should be selected for optimization.
    Higher priority score means higher chance of being selected.
    
    Args:
        current_score (float): The current score of the video.
        acceleration (float): The improvement rate of the last agent internal step.
                              (Typically from the previous run).
        config (dict): Configuration dictionary.
        
    Returns:
        float: The priority score.
    """
    # ---------------------------------------------------------
    # TODO: User can modify this formula manually.
    # Goal: Balanace between "needs improvement" (low score) and "has potential" (high acceleration).
    
    k = config['parameters'].get('priority_score_k', 1.0) # Weight for acceleration
    t = config['parameters'].get('priority_score_t', 1.0) # Weight for current score

    priority = (math.e**(k*acceleration))*((1-current_score)**t) # Base priority on how much improvement potential there is (low score) and how promising the acceleration is.
        
    return priority
    # ---------------------------------------------------------

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class AgentRunner:
    def __init__(self, config_path, device_id=0, node_rank=0, run_id=None):
        self.config = load_config(config_path)
        self.device_id = device_id
        self.node_rank = node_rank
        self.run_id = run_id
        
        # If run_id is provided, override output paths to save everything in log/run_id folder
        if self.run_id:
            # We assume the script is running from root, so 'log' will be in root
            base_log_dir = os.path.join("log", self.run_id)
            
            # Update paths to be inside the log folder
            self.config['paths']['output_dir'] = base_log_dir
            self.config['paths']['temp_frames_dir'] = os.path.join(base_log_dir, "temp_frames")
            self.config['paths']['video_frames_dir'] = os.path.join(base_log_dir, "video_frames")
            self.config['paths']['key_frames_dir'] = os.path.join(base_log_dir, "key_frames")
        
        # Prepare directories
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
        # Also create subdirectories if they are used directly (some scripts might rely on them existing)
        os.makedirs(self.config['paths']['temp_frames_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['video_frames_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['key_frames_dir'], exist_ok=True)
        
        # Initialize model
        self.model, self.processor = init(
            model_path=self.config['models']['main_model_path'], 
            device_id=device_id
        )

    def run_on_sample(self, sample: Dict[str, Any], video_base_dir: str, prompt_override: str = None):
        """
        Run the agent on a single sample and return full details.
        
        sample: {
            "key": str/int (optional, task key)
            "question": str,
            "options": List[str],
            "video_paths": List[str] (filenames or relative paths),
            "answer": str (optional, ground truth),
            "task": str (optional)
        }
        
        Returns:
            dict: The complete result data structure.
        """
        paths = self.config['paths']
        params = self.config['parameters']
        prompts = self.config['prompts']
        
        # Extract basic info
        key = sample.get('key', 'unknown')
        question = sample.get('question', '')
        # Prioritize 'ground_truth', fallback to 'answer', default to empty string
        ground_truth = sample.get('ground_truth', sample.get('answer', ''))
            
        task_name = sample.get('task', '')
        
        # Adapt to different dataset keys for video list
        if 'video_paths' in sample:
            video_filenames = sample.get('video_paths', [])
        elif 'videos' in sample:
            video_filenames = sample.get('videos', [])
        else:
            video_filenames = []

        options_raw = sample.get('options', [])
        if isinstance(options_raw, list):
            options_text = "\n".join(options_raw)
        else:
            options_text = str(options_raw)
        
        full_video_paths = [os.path.join(video_base_dir, v) for v in video_filenames]
        
        # Initialize timing and tracking
        timings = {}
        total_start = time.time()
        
        # --- Validate Videos ---
        valid_videos = []
        
        for i, v_path in enumerate(full_video_paths, 1):
            if not os.path.exists(v_path):
                print(f"Warning: Video not found {v_path}")
                continue
            valid_videos.append((i, v_path))

        if not valid_videos:
            return {'error': "Error: No valid videos", 'success': False, 'key': key}

        # --- Initialize TextBank and Agents ---
        TextBank = {
            'question': question,
            'videos': {}
        }
        
        tool_agent = ToolAgent(self.model, self.processor, self.config, self.device_id)
        desc_agent = DescAgent(self.model, self.processor, self.config, self.device_id)
        
        path_prefix = f"eval_node{self.node_rank}_gpu{self.device_id}" 
        # Ensure q_id is unique across distributed nodes if key is unknown or shared
        if key != 'unknown':
            q_id = str(key)
        else:
            # Fallback: timestamp + rank + gpu to ensure uniqueness
            q_id = f"{int(time.time()*1000)}_n{self.node_rank}_g{self.device_id}"
            
        benchmark_name = paths.get('Benchmark_name', 'Benchmark')
        question_name = os.path.join(benchmark_name, f"question{q_id}")
        
        process_logs = {
            'initialization': [],
            'iterations': []
        }
        
        # Phase 1: Initialization
        for idx, v_path in valid_videos:
            if self.config['parameters'].get('number_type') == "123":
                v_name = f"Video {idx}"
            else:
                v_name = f"Video {chr(ord('A') + idx - 1)}"
                
            # Pre-calculate duration
            v_duration = self._get_video_duration(v_path)

            TextBank['videos'][v_name] = {
                'path': v_path,
                'idx': idx,
                'duration': v_duration,
                'iteration_count': 0,
                'description': "Initial observation.", # Simple description
                'priority': 1.0, # Will be calculated
                'status': 'active',
                'last_acceleration': self.config['parameters']['agent']['initial_acceleration'],
                'last_score': 0.5, # Will be initialized by DescAgent
                'current_frames': [],
                'frame_bank': []
            }
            
            # Uniform sampling
            safe_v_name = v_name.replace(" ", "_")
            frames_info = self._extract_uniform_frames(v_path, 8, q_id, safe_v_name)
            
            # Use DescAgent to generate initial description and score
            v_label_str = v_name
            # For initialization, we don't have other descriptions yet
            other_descs = {}
            
            # Pass frames paths
            frame_paths = [f['path'] for f in frames_info]
            
            # Use describe_and_evaluate
            # Note: describe_and_evaluate expects desc_old. For init, use empty string.
            desc_new_part, score_new, v_term, g_term = desc_agent.describe_and_evaluate(
                question, 
                frames=frame_paths, 
                desc_old="", 
                other_descs=other_descs, 
                video_label=v_label_str
            )
            
            # Format initial description with time range (0 - duration)
            initial_desc = f"0.00-{v_duration:.2f}: {desc_new_part}"

            # We don't add frames to bank here, they will be scored by ToolAgent in loop or if terminated
            
            TextBank['videos'][v_name]['current_frames'] = frames_info
            TextBank['videos'][v_name]['description'] = initial_desc
            TextBank['videos'][v_name]['last_score'] = score_new # Set initial score from model
            # Compute initial priority immediately after initialization
            initial_priority = calculate_priority_score(
                TextBank['videos'][v_name]['last_score'],
                TextBank['videos'][v_name]['last_acceleration'],
                self.config
            )
            TextBank['videos'][v_name]['priority'] = initial_priority
            
            # Check termination immediately
            if v_term:
                 TextBank['videos'][v_name]['status'] = 'desc_terminated'
                 for f_info in frames_info:
                     TextBank['videos'][v_name]['frame_bank'].append((f_info['path'], 1.0, f_info['time']))
            
            process_logs['initialization'].append({
                'video': v_name,
                'status': 'initialized',
                'initial_description': initial_desc,
                'acceleration': self.config['parameters']['agent']['initial_acceleration'],
                'score': score_new,
                'priority': TextBank['videos'][v_name].get('priority', 1.0)
            })

        # Phase 2: Main Iteration Loop
        max_iterations = self.config['parameters'].get('agent', {}).get('global_max_iterations', 10)
        skip_iteration = self.config['parameters'].get('agent', {}).get('skip_iteration', False)

        global_terminated = False
        if skip_iteration:
            global_terminated = True

        iteration_count = 0
        
        while not global_terminated and iteration_count < max_iterations:
            agent_cfg = self.config['parameters'].get('agent', {})
            min_accel = agent_cfg.get('min_acceleration_threshold', 0.2)
            max_score = agent_cfg.get('max_score_threshold', 0.8)

            active_videos = {k: v for k, v in TextBank['videos'].items() if v['status'] == 'active'}
            if not active_videos:
                break

            # Calculate priority for all active videos to select one
            for v_k, v_v in active_videos.items():
                v_v['priority'] = calculate_priority_score(v_v['last_score'], v_v['last_acceleration'], self.config)

            # --- 终止逻辑：加速度低于阈值则终止该视频 ---
            for v_k, v_v in active_videos.items():
                if v_v['last_acceleration'] < min_accel:
                    v_v['status'] = 'accel_terminated'
                    v_v['last_score'] = max_score

            # --- 终止逻辑：所有视频平均分超过阈值则整体终止 ---
            all_scores = [v['last_score'] for v in TextBank['videos'].values()]
            if all_scores and sum(all_scores)/len(all_scores) >= max_score:
                global_terminated = True
                break

            # 重新筛选活跃视频
            active_videos = {k: v for k, v in TextBank['videos'].items() if v['status'] == 'active'}
            if not active_videos:
                break

            v_curr_name = max(active_videos.keys(), key=lambda k: active_videos[k]['priority'])
            v_curr = TextBank['videos'][v_curr_name]

            v_idx = v_curr['idx']
            if self.config['parameters'].get('number_type') == "123":
                v_label = f"Video {v_idx}"
            else:
                v_label = f"Video {chr(ord('A') + v_idx - 1)}"

            duration = v_curr.get('duration', self._get_video_duration(v_curr['path']))

            # --- Tool Agent Work ---
            current_candidate_frames = v_curr['current_frames']
            current_vis_frames = self._add_timestamp_to_frames(current_candidate_frames)
            existing_bank_frames = v_curr['frame_bank']
            current_video_desc = v_curr.get('description', "No description yet.")
            other_videos_desc = {k: v.get('description', 'No description.') for k, v in TextBank['videos'].items() if k != v_curr_name}
            candidate_scores, option, target_start, target_end = tool_agent.decide_action(
                question,
                existing_bank_frames,
                current_vis_frames,
                duration,
                video_label=v_label,
                current_video_desc=current_video_desc,
                other_videos_desc=other_videos_desc
            )
            
            # Update frame bank with SCORED candidates
            # Important: We store the ORIGINAL (clean) frames in the bank, not the timestamped ones
            # Timestamps are only for Agent's "eyes" during reasoning.
            # Deduplicate: Only add if not present, or if new score is higher
            existing_indices = {item[0]: i for i, item in enumerate(v_curr['frame_bank'])}

            for i, f_info in enumerate(current_candidate_frames):
                path = f_info['path']
                score = candidate_scores[i] if i < len(candidate_scores) else 0.5
                time_val = f_info['time']

                if path in existing_indices:
                    # If exists, check if new score is higher
                    idx = existing_indices[path]
                    old_path, old_score, old_time = v_curr['frame_bank'][idx]
                    if score > old_score:
                        v_curr['frame_bank'][idx] = (path, score, time_val)
                else:
                    # New frame
                    v_curr['frame_bank'].append((path, score, time_val))
                    # Update index map for subsequent checks in this loop (unlikely to have dups in one batch but safe)
                    existing_indices[path] = len(v_curr['frame_bank']) - 1

            # Enforce Frame Bank Capacity (keep top K scores)
            bank_limit = self.config['parameters']['agent'].get('frame_bank_size', 16)
            v_curr['frame_bank'].sort(key=lambda x: x[1], reverse=True)
            if len(v_curr['frame_bank']) > bank_limit:
                v_curr['frame_bank'] = v_curr['frame_bank'][:bank_limit]
            
            iter_log = {
                'iteration': iteration_count + 1,
                'selected_video': v_curr_name,
                'priority_before': v_curr['priority'],
                'action': {
                    'option': option,
                    'target_start': target_start,
                    'target_end': target_end
                }
            }

            if option == 6:
                v_curr['status'] = 'tool_terminated'
                iter_log['status'] = 'tool_terminated'
                process_logs['iterations'].append(iter_log)
                continue
            
            # Execute Action
            safe_v_curr_name = v_curr_name.replace(" ", "_")
            new_frames_info = self._sample_frames(v_curr['path'], option, target_start, target_end, v_curr['current_frames'], duration, q_id, safe_v_curr_name, iteration_count)
            new_frame_paths = [f['path'] for f in new_frames_info]
            
            if not new_frame_paths:
                v_curr['status'] = 'tool_terminated'
                iter_log['status'] = 'tool_terminated_no_frames'
                process_logs['iterations'].append(iter_log)
                continue

            # Update Current Frames for NEXT iteration
            if option in [1, 3, 5]:
                v_curr['current_frames'] = new_frames_info
                
            # Note: We do NOT add new frames to bank yet. They will be scored in the NEXT iteration.
            
            # --- Desc Agent Work ---
            # Describe NEW frames AND Evaluate status in ONE call
            other_descs = {k: v['description'] for k, v in TextBank['videos'].items() if k != v_curr_name}
            
            desc_new_part, score_new, v_term, g_term = desc_agent.describe_and_evaluate(
                question, 
                frames=new_frame_paths, 
                desc_old=v_curr['description'], 
                other_descs=other_descs, 
                video_label=v_label
            )
            
            # Update Description
            # Get accurate times from actual sampled frames
            start_desc = 0.0
            end_desc = duration
            if new_frames_info:
                start_desc = new_frames_info[0]['time']
                end_desc = new_frames_info[-1]['time']
            else:
                start_desc = target_start
                end_desc = target_end
                
            formatted_desc_part = f"{start_desc:.2f}-{end_desc:.2f}: {desc_new_part}"

            if v_curr['description'] == "Initial observation." or v_curr['description'] == "":
                 v_curr['description'] = formatted_desc_part
            else:
                 # Append new observation to existing history
                 v_curr['description'] += f"\n{formatted_desc_part}"

            old_score = v_curr['last_score']
            acceleration = (score_new - old_score) / old_score if old_score > 0 else 0.0
            
            # Update state
            v_curr['last_score'] = score_new
            v_curr['last_acceleration'] = acceleration
            v_curr['iteration_count'] += 1
            
            if v_term:
                v_curr['status'] = 'desc_terminated'
                # If desc_agent terminates, force add current frames with 1.0 score
                for f_info in v_curr['current_frames']:
                     v_curr['frame_bank'].append((f_info['path'], 1.0, f_info['time']))
                # Re-sort and truncate
                v_curr['frame_bank'].sort(key=lambda x: x[1], reverse=True)
                if len(v_curr['frame_bank']) > self.config['parameters']['agent'].get('frame_bank_size', 16):
                    v_curr['frame_bank'] = v_curr['frame_bank'][:self.config['parameters']['agent'].get('frame_bank_size', 16)]
            if g_term:
                global_terminated = True
                # Global termination -> force add ALL active current frames (for the current video)
                if not v_term: # If not already handled above
                     for f_info in v_curr['current_frames']:
                        v_curr['frame_bank'].append((f_info['path'], 1.0, f_info['time']))
                     v_curr['frame_bank'].sort(key=lambda x: x[1], reverse=True)
                     if len(v_curr['frame_bank']) > self.config['parameters']['agent'].get('frame_bank_size', 16):
                        v_curr['frame_bank'] = v_curr['frame_bank'][:self.config['parameters']['agent'].get('frame_bank_size', 16)]

            iter_log.update({
                'new_description_part': desc_new_part,
                'full_description': v_curr['description'],
                'old_score': old_score,
                'new_score': score_new,
                'acceleration': acceleration,
                'video_terminated': v_term,
                'global_terminated': g_term
            })
            process_logs['iterations'].append(iter_log)
            
            iteration_count += 1
            
        # Phase 3: Result Generation
        final_frame_paths = []
        final_descriptions_list = []
        for v_name, v_data in TextBank['videos'].items():
            safe_v_name = v_name.replace(" ", "_")
            # Apply watermark
            # Use cached duration if available
            v_duration = v_data.get('duration', 0.0)
            if v_duration <= 0 and os.path.exists(v_data['path']):
                v_duration = self._get_video_duration(v_data['path'])

            watermarked_frames = self._apply_watermark_to_bank(
                v_data['frame_bank'], 
                v_data['idx'], 
                q_id, 
                safe_v_name, 
                video_duration=v_duration
            )
            final_frame_paths.extend(watermarked_frames)
            
            # v_label is already the key (v_name), or can be re-derived if needed for consistency check
            v_label = v_name
            
            desc_text = v_data.get('description', 'No description available.')
            final_descriptions_list.append(f"{v_label}: {desc_text}")
        
        final_descriptions_str = "\n".join(final_descriptions_list)
            
        if not final_frame_paths:
             # Default check, will be refined below based on mode
             pass 

        # --- Dynamic Prompt Selection based on input types ---
        use_visual = self.config['parameters'].get('use_visual_answer', True)
        use_text = self.config['parameters'].get('use_text_answer', True)

        if not use_visual:
            final_frame_paths = []
        if not use_text:
            final_descriptions_str = ""

        # Validate inputs based on mode
        if not use_visual and not use_text:
             return {'error': "Config Error: Both use_visual_answer and use_text_answer are False.", 'success': False, 'key': key}
        
        if use_visual and not final_frame_paths:
             return {'error': "No frames selected (Visual Answer required)", 'success': False, 'key': key}
             
        if use_text and not final_descriptions_str and not use_visual:
             # If text-only mode and no text, we can't proceed.
             return {'error': "No descriptions available (Text-only Answer required)", 'success': False, 'key': key}

        try:
            if prompt_override:
                current_prompt_template = prompt_override
            else:
                # Select template from config
                if use_visual and use_text:
                    current_prompt_template = prompts.get('answer_combined')
                    # Fallback to legacy 'answer' key if 'answer_combined' missing
                    if not current_prompt_template:
                        current_prompt_template = prompts.get('answer')
                elif use_visual and not use_text:
                    current_prompt_template = prompts.get('answer_visual_only')
                elif not use_visual and use_text:
                    current_prompt_template = prompts.get('answer_text_only')
            
            # Create a default template if missing (safety net)
            if not current_prompt_template:
                if use_visual and not use_text:
                     current_prompt_template = "Question: {QUESTION}\nOptions: {OPTIONS}\nAnswer based on the images."
                elif not use_visual and use_text:
                     current_prompt_template = "Video Descriptions:\n{DESCRIPTIONS}\n\nQuestion: {QUESTION}\nOptions: {OPTIONS}\nAnswer based on descriptions."
                else:
                     current_prompt_template = "Video Descriptions:\n{DESCRIPTIONS}\n\nQuestion: {QUESTION}\nOptions: {OPTIONS}\nAnswer based on images and descriptions."

            # Inject Descriptions if needed
            if "{DESCRIPTIONS}" in current_prompt_template:
                current_prompt_template = current_prompt_template.replace("{DESCRIPTIONS}", final_descriptions_str)
            else:
                # Fallback: if template doesn't have placeholder but we have text to show (and we are in a text mode), append it.
                if use_text and final_descriptions_str:
                    # Prepend description is standard practice
                    current_prompt_template = current_prompt_template.replace("Question:", f"Video Descriptions:\n{final_descriptions_str}\n\nQuestion:")

            output_text = answer(
                video_frames=final_frame_paths,
                question=question,
                options=options_text,
                prompt_template=current_prompt_template,
                device_id=self.device_id,
                model_path=self.config['models']['main_model_path']
            )
            
            ans_output_clean = output_text.strip()
            predicted_ans = ans_output_clean if ans_output_clean else ""
                    
        except Exception as e:
            predicted_ans = ""
            output_text = str(e)
            
        is_correct = False
        if ground_truth and predicted_ans:
            if predicted_ans.upper() == ground_truth.upper():
                is_correct = True
                 
        total_end = time.time()
        timings['total_pipeline_duration'] = round(total_end - total_start, 3)

        model_outputs = {
            'final_description': {k: v['description'] for k, v in TextBank['videos'].items()},
            'answer_generation': {
                'raw_output': output_text
            }
        }
        
        try:
            # 清理 key_frames 目录 (如果配置为不保存)
            if not self.config['parameters'].get('save_key_frames', False):
                try:
                    question_dir = os.path.join(self.config['paths']['key_frames_dir'], q_id)
                    if os.path.exists(question_dir):
                        import shutil
                        shutil.rmtree(question_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up key frames directory: {e}")
                    
            # 清理 video_frames 目录 (如果配置为不保存)
            if not self.config['parameters'].get('save_video_frames', False):
                try:
                    question_dir = os.path.join(self.config['paths']['video_frames_dir'], q_id)
                    if os.path.exists(question_dir):
                        import shutil
                        shutil.rmtree(question_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up video frames directory: {e}")
        except Exception as e:
             print(f"Warning: Global cleanup failed in run_on_sample: {e}")

        return {
            'key': key,
            'question': question,
            'task': task_name,
            'predicted_answer': predicted_ans,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'process_logs': process_logs,
            'timings': timings,
            'iterations': iteration_count,
            'model_outputs': model_outputs,
            'success': True
        }

    def _get_video_duration(self, video_path: str) -> float:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 0.0

    def _extract_frame_at_time(self, video_path: str, time_sec: float, output_dir: str, frame_name: str) -> str:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = int(time_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret: 
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)
                save_path = os.path.join(output_dir, frame_name)
                frame_image.save(save_path)
                return save_path
        except Exception as e:
            print(f"Error extracting frame: {e}")
        return None

    def _extract_uniform_frames(self, video_path: str, num_frames: int, q_id: str, v_name: str, offset: float = 0.0) -> List[dict]:
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            return []
            
        output_dir = os.path.join(self.config['paths']['video_frames_dir'], q_id, v_name)
        os.makedirs(output_dir, exist_ok=True)
        
        times = np.linspace(0, duration, num_frames + 2)[1:-1]
        times = [min(max(t + offset, 0), duration) for t in times]
        
        frames_info = []
        for i, t in enumerate(times):
            f_name = f"uniform_{i}_{t:.2f}.jpg"
            f_path = self._extract_frame_at_time(video_path, t, output_dir, f_name)
            if f_path:
                frames_info.append({'path': f_path, 'time': t})
        return frames_info

    def _add_timestamp_to_frames(self, frames_info: List[dict]) -> List[dict]:
        """
        Add timestamp watermark to temporary copies of frames for Tool Agent visualization.
        Returns a list of NEW frame paths (temporary).
        """
        new_frames = []
        for f_info in frames_info:
            src_path = f_info['path']
            time_val = f_info['time']
            
            if not os.path.exists(src_path):
                continue
                
            # Create a temp path in the same directory but with _ts suffix
            dir_name = os.path.dirname(src_path)
            file_name = os.path.basename(src_path)
            name, ext = os.path.splitext(file_name)
            dst_path = os.path.join(dir_name, f"{name}_ts{ext}")
            
            # If already exists (maybe from previous call), just use it
            if not os.path.exists(dst_path):
                img = cv2.imread(src_path)
                if img is not None:
                    # Add timestamp watermark
                    text = f"{time_val:.2f}s"
                    # Yellow text, size 1.0, thickness 2
                    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imwrite(dst_path, img)
                    
            new_frames.append({'path': dst_path, 'time': time_val})
            
        return new_frames

    def _sample_frames(self, video_path: str, option: int, target_start: float, target_end: float, current_frames: List[dict], duration: float, q_id: str, v_name: str, iteration: int) -> List[dict]:
        output_dir = os.path.join(self.config['paths']['video_frames_dir'], q_id, v_name)
        os.makedirs(output_dir, exist_ok=True)
        
        if option == 5:
            # Global uniform with offset
            offset = duration / (8 + 1) * 0.25
            return self._extract_uniform_frames(video_path, 8, q_id, v_name, offset)
            
        # For options 1-4, sample within [target_start, target_end]
        times = np.linspace(target_start, target_end, 8 + 2)[1:-1]
        frames_info = []
        for i, t in enumerate(times):
            f_name = f"iter{iteration}_{i}_{t:.2f}.jpg"
            f_path = self._extract_frame_at_time(video_path, t, output_dir, f_name)
            if f_path:
                frames_info.append({'path': f_path, 'time': t})
        return frames_info

    def _wm_visual(self, img, video_idx, time_sec):
        """
        Mode 1: (video_tag) - Add Video Label ONLY.
        """
        if self.config['parameters'].get('number_type') == "123":
            cv2.putText(img, f"Video {video_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(img, f"Video {chr(ord('A') + video_idx - 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
        
        return img

    def _wm_temporal(self, img, video_idx, time_sec, duration):
        """
        Mode 2: (temporal_tag) - Add Timestamp (or progress?) ONLY.
        """
        # Requirement: "帧在视频中的时间戳"
        # Previous implementation was progress bar. 
        # But user says "temporal_tag: 帧在视频中的时间戳".
        # Let's just put the text.
        
        text = f"{time_sec:.2f}s"
        cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
        
        return img

    def _wm_trans_sequence(self, frame_bank, video_idx, output_dir):
        """
        Mode 3: (trans_frame)
        Insert a transition frame [Video X] BEFORE the frames of the video.
        Format: [Video1][frame1]...[Video2][frame1]...
        """
        watermarked_paths = []
        
        # Determine label
        if self.config['parameters'].get('number_type') == "123":
            label = f"Video {video_idx}"
        else:
            label = f"Video {chr(ord('A') + video_idx - 1)}"

        # Create separator frame
        # Use simple size or detect from first real frame
        h, w = 720, 1280
        if frame_bank and os.path.exists(frame_bank[0][0]):
            tmp = cv2.imread(frame_bank[0][0])
            if tmp is not None:
                h, w = tmp.shape[:2]

        trans_img = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Calculate text position
        text_size = cv2.getTextSize(label, font, 2.0, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(trans_img, label, (text_x, text_y), font, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
        
        trans_path = os.path.join(output_dir, f"trans_{video_idx}.jpg")
        cv2.imwrite(trans_path, trans_img)
        
        # Add separator frame FIRST
        watermarked_paths.append(trans_path)
            
        # Add regular frames (just copy them to be safe, or just return path if no other watermark)

        for i, (src, _, _) in enumerate(frame_bank):
             watermarked_paths.append(src)
             
        return watermarked_paths

    def _apply_watermark_to_bank(self, frame_bank: List[tuple], video_idx: int, q_id: str, v_name: str, video_duration: float = 0.0) -> List[str]:
        # Sort frame bank by timestamp before processing
        # frame_bank items are (path, score, time)
        sorted_bank = sorted(frame_bank, key=lambda x: x[2])
        
        wm_type = self.config['parameters'].get('type_watermark', 'none')
        
        # If disabled or unknown
        if wm_type == 'none':
            return [f[0] for f in sorted_bank]
            
        output_dir = os.path.join(self.config['paths']['key_frames_dir'], q_id, v_name)
        os.makedirs(output_dir, exist_ok=True)
        

            
        watermarked_paths = []
        
        # 1. Apply visual/temporal watermarks first
        processed_frames_with_time = [] 

        for i, (src, score, t_sec) in enumerate(sorted_bank, 1):
            if os.path.exists(src):
                img = cv2.imread(src)
                if img is not None:
                    
                    if "video_tag" in wm_type:
                        img = self._wm_visual(img, video_idx, t_sec)
                    
                    if 'temporal_tag' in wm_type:
                        # Use passed duration directly
                        img = self._wm_temporal(img, video_idx, t_sec, video_duration)
                    
                    dst = os.path.join(output_dir, f"watermarked_{i}.jpg")
                    cv2.imwrite(dst, img)
                    watermarked_paths.append(dst)
                    # Keep track of time for trans sequence if needed (though transition usually doesn't need time)
                    processed_frames_with_time.append((dst, score, t_sec))

        # 2. Add transition frame if needed
        # Special case for 'trans' because it changes the list structure (adds frames)
        if 'trans_frame' in wm_type:
            # Pass the already processed (watermarked) frames effectively
            return self._wm_trans_sequence(processed_frames_with_time, video_idx, output_dir)
                    
        return watermarked_paths
