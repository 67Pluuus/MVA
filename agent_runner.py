import os
import sys
import yaml
import json
import time
from typing import Dict, Any, List
import math

# Adjust path to import from root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils import DescribeVideo_qwen3_vl, init, answer, Qwen_VL
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
        
        # --- Step 0: Initial Description ---
        timings['video_description'] = {
            'start': time.time(),
            'per_video': {},
            'total': 0.0
        }
        
        descriptions = {}
        video_description_outputs = {} 
        valid_videos = []
        
        for i, v_path in enumerate(full_video_paths, 1):
            if not os.path.exists(v_path):
                print(f"Warning: Video not found {v_path}")
                continue
            valid_videos.append((i, v_path))
            
            if params.get('number_type') == "123":
                v_key = f"Video {i}"
            else:
                v_key = f"Video {chr(ord('A') + i - 1)}"
            
            desc_start = time.time()
            desc = DescribeVideo_qwen3_vl(
                v_path, 
                question, 
                sampled_frame=params.get('describe_frames', 8),
                prompt_template=prompts.get('video_description', ""), 
                device_id=self.device_id,
                model_path=self.config['models']['main_model_path'],
                temp_base_dir=self.config['paths'].get('temp_frames_dir', None),
                node_rank=self.node_rank
            )
            desc_end = time.time()
            
            descriptions[v_key] = desc
            video_description_outputs[v_key] = desc
            timings['video_description']['per_video'][v_key] = round(desc_end - desc_start, 3)

        timings['video_description']['end'] = time.time()
        timings['video_description']['total'] = round(timings['video_description']['end'] - timings['video_description']['start'], 3)
        
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
                v_label = v_name
            else:
                v_name = f"Video {chr(ord('A') + idx - 1)}"
                v_label = v_name
                
            # Pre-calculate duration to avoid repeated IO
            v_duration = self._get_video_duration(v_path)

            TextBank['videos'][v_name] = {
                'path': v_path,
                'idx': idx,
                'duration': v_duration,
                'iteration_count': 0,
                'description': descriptions.get(v_name, ""),
                'priority': 1.0,
                'status': 'active',
                'last_acceleration': self.config['parameters'].get('agent', {}).get('initial_acceleration', 1.0),
                'last_score': 0.0,
                'current_frames': [],
                'frame_bank': []
            }
            
            # Basic exploration (Option 5)
            # Create a safe directory name by replacing spaces
            safe_v_name = v_name.replace(" ", "_")
            frames_info = self._extract_uniform_frames(v_path, 8, q_id, safe_v_name)
            frame_paths = [f['path'] for f in frames_info]
            
            # v_label is already defined at start of loop
            
            scores = tool_agent.score_frames(question, frame_paths, video_label=v_label)
            
            for f_info, score in zip(frames_info, scores):
                TextBank['videos'][v_name]['frame_bank'].append((f_info['path'], score, f_info['time']))
            TextBank['videos'][v_name]['frame_bank'].sort(key=lambda x: x[1], reverse=True)
            TextBank['videos'][v_name]['frame_bank'] = TextBank['videos'][v_name]['frame_bank'][:self.config['parameters'].get('agent', {}).get('frame_bank_size', 10)]
            
            desc_raw = tool_agent.generate_raw_description(question, frame_paths, video_label=v_label)
            
            desc_refined, score_new, v_term, g_term = desc_agent.refine_and_evaluate(
                question, TextBank['videos'][v_name]['description'], desc_raw, {}, video_label=v_label
            )
            
            # Force termination signals to False during initialization
            v_term = False
            g_term = False
            
            TextBank['videos'][v_name]['description'] = desc_refined
            TextBank['videos'][v_name]['last_score'] = score_new
            TextBank['videos'][v_name]['current_frames'] = frames_info
            
            if v_term:
                TextBank['videos'][v_name]['status'] = 'desc_terminated'
            
            process_logs['initialization'].append({
                'video': v_name,
                'sampled_frames': [{'time': f['time'], 'score': s} for f, s in zip(frames_info, scores)],
                'raw_description': desc_raw,
                'refined_description': desc_refined,
                'score': score_new,
                'video_terminated': v_term,
                'global_terminated': g_term
            })

            if g_term:
                break
                
        # Phase 2: Main Iteration Loop
        max_iterations = self.config['parameters'].get('agent', {}).get('global_max_iterations', 10)
        global_terminated = False
        iteration_count = 0
        
        while not global_terminated and iteration_count < max_iterations:
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

            # Use cached duration
            duration = v_curr.get('duration', self._get_video_duration(v_curr['path']))
            option, target_start, target_end = tool_agent.decide_action(question, v_curr['current_frames'], duration, video_label=v_label)
            
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
            
            # Use safe name for directory creation
            safe_v_curr_name = v_curr_name.replace(" ", "_")
            new_frames_info = self._sample_frames(v_curr['path'], option, target_start, target_end, v_curr['current_frames'], duration, q_id, safe_v_curr_name, iteration_count)
            new_frame_paths = [f['path'] for f in new_frames_info]
            
            # v_label is already defined above
            
            if not new_frame_paths:
                v_curr['status'] = 'tool_terminated'
                iter_log['status'] = 'tool_terminated_no_frames'
                process_logs['iterations'].append(iter_log)
                continue
                
            scores = tool_agent.score_frames(question, new_frame_paths, video_label=v_label)
            
            for f_info, score in zip(new_frames_info, scores):
                v_curr['frame_bank'].append((f_info['path'], score, f_info['time']))
            v_curr['frame_bank'].sort(key=lambda x: x[1], reverse=True)
            v_curr['frame_bank'] = v_curr['frame_bank'][:self.config['parameters'].get('agent', {}).get('frame_bank_size', 10)]
            
            desc_raw = tool_agent.generate_raw_description(question, new_frame_paths, video_label=v_label)
            
            other_descs = {k: v['description'] for k, v in TextBank['videos'].items() if k != v_curr_name}
            desc_refined, score_new, v_term, g_term = desc_agent.refine_and_evaluate(
                question, v_curr['description'], desc_raw, other_descs, video_label=v_label
            )
            
            old_score = v_curr['last_score']
            acceleration = (score_new - old_score) / old_score if old_score > 0 else 0.0
            
            v_curr['description'] = desc_refined
            v_curr['last_score'] = score_new
            v_curr['last_acceleration'] = acceleration
            v_curr['iteration_count'] += 1
            
            if option in [1, 3, 5]:
                v_curr['current_frames'] = new_frames_info
                
            if v_term:
                v_curr['status'] = 'desc_terminated'
            if g_term:
                global_terminated = True

            P = calculate_priority_score(score_new, acceleration, self.config)
            v_curr['priority'] = P

            iter_log.update({
                'sampled_frames': [{'time': f['time'], 'score': s} for f, s in zip(new_frames_info, scores)],
                'raw_description': desc_raw,
                'refined_description': desc_refined,
                'old_score': old_score,
                'new_score': score_new,
                'acceleration': acceleration,
                'priority_after': P,
                'video_terminated': v_term,
                'global_terminated': g_term
            })
            process_logs['iterations'].append(iter_log)
            
            if g_term:
                break
            
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
            return {'error': "No frames selected", 'success': False, 'key': key}

        try:
            current_prompt_template = prompt_override if prompt_override else prompts.get('answer')
            if "{DESCRIPTIONS}" in current_prompt_template:
                current_prompt_template = current_prompt_template.replace("{DESCRIPTIONS}", final_descriptions_str)
            else:
                # Fallback if placeholder missing but we want to include descriptions
                # Prepend to Question
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
            f_name = f"uniform_{time.time()}_{i}.jpg"
            f_path = self._extract_frame_at_time(video_path, t, output_dir, f_name)
            if f_path:
                frames_info.append({'path': f_path, 'time': t})
        return frames_info

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
            f_name = f"iter{iteration}_{time.time()}_{i}.jpg"
            f_path = self._extract_frame_at_time(video_path, t, output_dir, f_name)
            if f_path:
                frames_info.append({'path': f_path, 'time': t})
        return frames_info

    def _wm_tag(self, img, video_idx, time_sec):
        """
        Mode 1: (tag) - Add Video Label and Timestamp to top-left corner.
        """
        if self.config['parameters'].get('number_type') == "123":
            cv2.putText(img, f"Video {video_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, f"Video {chr(ord('A') + video_idx - 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        text = f"{time_sec:.2f}s"
        cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return img

    def _wm_temporal(self, img, video_idx, time_sec, duration):
        """
        Mode 3: (temporal) - Add Video Label to top-left, and a progress bar at the bottom.
        """
        h, w = img.shape[:2]
        
        # 1. Video Label
        if self.config['parameters'].get('number_type') == "123":
            cv2.putText(img, f"Video {video_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, f"Video {chr(ord('A') + video_idx - 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
        # 2. Progress Bar
        bar_height = int(h * 0.05) # 5% of height
        bar_y = h - bar_height - 10
        
        # Background bar (gray)
        cv2.rectangle(img, (10, bar_y), (w - 10, bar_y + bar_height), (100, 100, 100), -1)
        
        # Progress (red or green)
        if duration > 0:
            progress = min(max(time_sec / duration, 0.0), 1.0)
            progress_width = int((w - 20) * progress)
            cv2.rectangle(img, (10, bar_y), (10 + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            
        # Timestamp text next to bar or on top? 
        # Requirement says: "indicates current frame position", progress bar does that visually.
        # But adding text is helpful. Let's add it above the bar.
        text = f"{time_sec:.2f}s / {duration:.2f}s"
        cv2.putText(img, text, (10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return img

    def _wm_trans_sequence(self, frame_bank, video_idx, output_dir):
        """
        Mode 2: (trans) - Add transition frames between regular frames.
        Returns a list of paths (strings).
        """
        watermarked_paths = []
        # Sort by time just in case, though usually sorted by score.
        # Wait, the prompt says "previewing the NEXT frame's info". 
        # If sorted by score, the order is not chronological. Transition frames imply chronological viewing.
        # However, the model input is a sequence of images. If the images are shuffled by score, a "transition" implies a jump in time/content.
        # Usually for VLM input, we sort by time if we want temporal coherence.
        # But here `frame_bank` is sorted by SCORE in `run_on_sample`.
        # "TextBank['videos'][v_name]['frame_bank'].sort(key=lambda x: x[1], reverse=True)"
        # If we insert transition frames, it suggests we want to guide the model through the frames.
        # If the frames are out of order (score-based), "next frame" just means the next one in the list presented to the model.
        # So I will respect the list order (which is score-based descending currently).
        
        for i in range(len(frame_bank)):
            src, score, t_sec = frame_bank[i]
            
            # Process current frame
            if os.path.exists(src):
                img = cv2.imread(src)
                if img is not None:
                    # Apply minimal tag (Video X) to keep context
                    if self.config['parameters'].get('number_type') == "123":
                        label = f"Video {video_idx}"
                    else:
                        label = f"Video {chr(ord('A') + video_idx - 1)}"
                    
                    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, f"{t_sec:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    dst = os.path.join(output_dir, f"frame_{i}.jpg")
                    cv2.imwrite(dst, img)
                    watermarked_paths.append(dst)

            # Add transition frame if there is a next frame
            if i < len(frame_bank) - 1:
                next_src, next_score, next_t_sec = frame_bank[i+1]
                
                # Create black image
                # Use same size as last image or default
                if 'img' in locals() and img is not None:
                    h, w = img.shape[:2]
                else:
                    h, w = 720, 1280 # Default fallback
                    
                trans_img = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Text: "Next: Video X - <time>s"
                if self.config['parameters'].get('number_type') == "123":
                    next_label = f"Video {video_idx}"
                else:
                    next_label = f"Video {chr(ord('A') + video_idx - 1)}"
                    
                text1 = "NEXT FRAME PREVIEW"
                text2 = f"{next_label}"
                text3 = f"Time: {next_t_sec:.2f}s"
                
                # Center text
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.putText(trans_img, text1, (50, h//2 - 40), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(trans_img, text2, (50, h//2 + 10), font, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.putText(trans_img, text3, (50, h//2 + 60), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
                
                trans_dst = os.path.join(output_dir, f"trans_{i}_{i+1}.jpg")
                cv2.imwrite(trans_dst, trans_img)
                watermarked_paths.append(trans_dst)
                
        return watermarked_paths

    def _apply_watermark_to_bank(self, frame_bank: List[tuple], video_idx: int, q_id: str, v_name: str, video_duration: float = 0.0) -> List[str]:
        wm_type = self.config['parameters'].get('type_watermark', 'none')
        
        # If disabled or unknown
        if wm_type == 'none':
            return [f[0] for f in frame_bank]
            
        output_dir = os.path.join(self.config['paths']['key_frames_dir'], q_id, v_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Special case for 'trans' because it changes the list structure (adds frames)
        if wm_type == 'trans':
            return self._wm_trans_sequence(frame_bank, video_idx, output_dir)
            
        watermarked_paths = []
        
        for i, (src, score, t_sec) in enumerate(frame_bank, 1):
            if os.path.exists(src):
                img = cv2.imread(src)
                if img is not None:
                    
                    if wm_type == 'visual_token' or wm_type == 'tag':
                        img = self._wm_tag(img, video_idx, t_sec)
                    elif wm_type == 'temporal':
                        # Use passed duration directly
                        img = self._wm_temporal(img, video_idx, t_sec, video_duration)
                    # Add more types here
                    
                    dst = os.path.join(output_dir, f"watermarked_{i}.jpg")
                    cv2.imwrite(dst, img)
                    watermarked_paths.append(dst)

        if 'trans' in wm_type:
            return self._wm_trans_sequence(frame_bank, video_idx, output_dir)
                    
        return watermarked_paths
