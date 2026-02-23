import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import shutil
import re
import time
import heapq

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VCA:
    def __init__(self, video_path: str, question: str, descriptions: Dict, model, processor, video_idx, config: Dict, path_prefix: str = "", question_name: str = ""):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.question = question
        self.descriptions = descriptions
        self.model = model
        self.processor = processor
        self.video_idx = video_idx
        self.config = config
        
        # 树状搜索参数
        vca_params = self.config['parameters'].get('vca', {})
        self.initial_segments_n = vca_params.get('initial_segments', 8)
        self.memory_size = vca_params.get('memory_size', 8)
        self.sub_segments_n = vca_params.get('sub_segments', 8)
        
        # New parameters for adaptive iteration
        self.min_iterations = vca_params.get('min_iterations', 1)
        self.max_iterations = vca_params.get('max_iterations', 5) # Default/Safety limit
        self.score_threshold = vca_params.get('video_score_threshold', 0.6)
        # self.initial_acceleration = vca_params.get('initial_acceleration', 1.0)
        
        # 存储原始描述和反馈原因
        self.original_description = None
        self.feedback_reason = None
        self.final_selection = None # Store (path, time) for later token addition
        
        # 路径设置
        video_frames_dir = self.config['paths'].get('video_frames_dir', './video_frames')
        key_frames_dir = self.config['paths'].get('key_frames_dir', './key_frames')
        
        if path_prefix:
            self.video_frames_path = os.path.join(video_frames_dir, path_prefix, self.video_name)
            self.key_frames_path = os.path.join(key_frames_dir, path_prefix, question_name, self.video_name)
        else:
            self.video_frames_path = os.path.join(video_frames_dir, self.video_name)
            self.key_frames_path = os.path.join(key_frames_dir, self.video_name)
            
        os.makedirs(self.video_frames_path, exist_ok=True)
        os.makedirs(self.key_frames_path, exist_ok=True)

    def add_visual_tokens(self) -> List[str]:
        """
        Add visual tokens (Video ID, Timestamp) to the final selected frames.
        This should be called only after the final iteration.
        Returns:
            List[str]: paths to the images with tokens.
        """
        if not self.final_selection:
            return []
            
        latest_dst_dir = os.path.join(self.key_frames_path, "latest_selection")
        final_paths_w = []
        
        for i, (src, t_sec) in enumerate(self.final_selection, 1):
            # The clean image should already be at latest_dst_dir/i.jpg
            dst = os.path.join(latest_dst_dir, f"{i}_{t_sec:.2f}.jpg")
            
            if os.path.exists(dst):
                img = cv2.imread(dst)
                if img is not None:
                    # Video Label (Top Left) - Creating smaller font
                    # Scale 1.0 (was 2.0), Thickness 2
                    if self.config.get('number_type') == "123":
                        cv2.putText(img, f"Video {self.video_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(img, f"Video {chr(ord('A') + self.video_idx - 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Timestamp Label (Below Video Label)
                    # Scale 0.8 (was 1.5), Thickness 2
                    text = f"{t_sec:.2f}s"
                    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Save as _w+.jpg per convention
                    final_path_w = os.path.join(latest_dst_dir, f"{i}_{t_sec:.2f}_w+.jpg")
                    cv2.imwrite(final_path_w, img)
                    final_paths_w.append(final_path_w)
            else:
                print(f"Warning: Clean frame not found at {dst}")
                
        return final_paths_w

    def Qwen_VL(self, messages):
        try:
            model, processor = self.model, self.processor
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0] if output_text else ""
        except Exception as e:
            print(f"Error in Qwen_VL: {e}")
            return ""
    
    def _get_video_duration(self) -> float:
        try:
            cap = cv2.VideoCapture(self.video_path)
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
    
    def _get_video_frame_info(self) -> Tuple[int, float]:
        """获取视频总帧数和时长"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return 0, 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0.0
            cap.release()
            return total_frames, duration
        except Exception as e:
            print(f"Error getting video frame info: {e}")
            return 0, 0.0

    def _extract_frame_at_time(self, time_sec: float, frame_name: str, output_dir=None) -> Optional[str]:
        """提取指定时间点的帧"""
        if output_dir is None:
            output_dir = self.video_frames_path
            
        try:
            cap = cv2.VideoCapture(self.video_path)
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

    def _evaluate_frames_batch(self, frames: List[str]) -> Tuple[List[Dict], str, str]:
        """
        批量评估一组帧，返回每段落的 (score, start, end) 等信息和原始输出
        """
        if not frames:
            return [], 0.0, ""
            
        prompt_template = self.config['prompts'].get('vca_main', "")
        
        # 获取视频信息
        total_frames, duration = self._get_video_frame_info()
        
        # 构建片段信息
        segment_info = "/* Segment Information */\n"
        for i, frame_path in enumerate(frames, 1):
            # 使用<image>占位符
            segment_info += f"Frame {i}: <image>\n"
        
        # 检查反馈信息有效性
        has_valid_feedback = (
            self.original_description and 
            self.feedback_reason and
            self.feedback_reason.strip() and
            not self.feedback_reason.startswith("Failed") and
            not self.feedback_reason.startswith("Empty") and
            not self.feedback_reason.startswith("Error")
        )
        
        # 如果有反馈信息，添加到片段信息中
        if has_valid_feedback:
            segment_info += f"\n/* Historical Context */\n"
            segment_info += f"Previous Feedback: {self.feedback_reason}\n"
            segment_info += f"Original Description: {self.original_description}\n"
            segment_info += f"IMPORTANT: When scoring these frames, prioritize frames that address the issues mentioned in the feedback.\n"
        
        # 填充 prompt 模板中的占位符
        prompt = prompt_template.replace("{FRAME_COUNT}", str(total_frames))
        prompt = prompt.replace("{DURATION:.2f}", f"{duration:.2f}")
        prompt = prompt.replace("{QUESTION}", self.question)
        prompt = prompt.replace("{SEGMENT_INFO}", segment_info)
        prompt = prompt.replace("{threshold}", str(self.config['parameters']['vca']['video_score_threshold']))

        content = []
        for f_path in frames:
            content.append({"type": "image", "image": f_path})
        content.append({"type": "text", "text": prompt})
        
        output = self.Qwen_VL([{"role": "user", "content": content}])
        
        # 解析输出：新格式为两行数字
        # Line 1: N-1 个数字（0.0-1.0），代表每个段落的分数
        # Line 2: 1 个数字（0.0-1.0），代表所有帧是否足够回答问题
        results = []
        
        lines = output.strip().split('\n')
        segment_scores = []
        overall_sufficiency = 0.0
        
        # 解析第一行：N-1 个段落的分数
        if len(lines) >= 1:
            line1 = lines[0].strip()
            # 提取所有浮点数
            score_pattern = r'\b(0\.\d+|1\.0|0|1)\b'
            matches = re.findall(score_pattern, line1)
            if matches:
                try:
                    segment_scores = [float(m) for m in matches]
                    # 确保分数在 0-1 范围内
                    segment_scores = [max(0.0, min(1.0, s)) for s in segment_scores]
                except:
                    segment_scores = []
        
        # 解析第二行：整体充分性分数
        if len(lines) >= 2:
            line2 = lines[1].strip()
            match = re.search(r'\b(0\.\d+|1\.0|0|1)\b', line2)
            if match:
                try:
                    overall_sufficiency = float(match.group(1))
                    overall_sufficiency = max(0.0, min(1.0, overall_sufficiency))
                except:
                    overall_sufficiency = 0.0

        # 将结果直接作为段落存储到 memory buffer
        # 数据格式以段落为单位，包含段落分数和首尾帧路径
        results = []
        if len(segment_scores) > 0:
            # 理想情况：N 帧对应 N-1 个段落分数
            # 但我们需要处理潜在的长度不匹配
            num_segments = min(len(segment_scores), len(frames) - 1)
            
            for i in range(num_segments):
                results.append({
                    "score": segment_scores[i],
                    "start_path": frames[i],
                    "end_path": frames[i+1],
                })
        else:
            # 如果没有解析出分数，可能是单帧或者解析错误
            # 如果只有一帧，无法形成段落，返回空
            pass
            
        return results, overall_sufficiency, output   # 返回段落结果列表和分数和原始输出

    def update_feedback(self, original_description: str, feedback_reason: str):
        self.original_description = original_description
        self.feedback_reason = feedback_reason

    # def _evaluate_keyframes_reward(self, keyframes_dict: Dict[str, List[str]], video_key: str) -> Tuple[float, str]:
    #     reward_prompt_template = self.config['prompts']['vca_main']
        
    #     # 获取视频信息
    #     total_frames, duration = self._get_video_frame_info()
        
    #     # 构建关键帧信息
    #     keyframes_info = "Keyframes from all videos:\n"
    #     for v_idx, (v_key, v_frames) in enumerate(keyframes_dict.items(), start=1):
    #         keyframes_info += f"\n[Video {v_idx}] - {len(v_frames)} keyframes selected:\n"
    #         for i, frame_path in enumerate(v_frames, 1):
    #             keyframes_info += f"  Keyframe {i}: {frame_path}\n"
        
    #     # 填充 prompt 模板中的占位符
    #     reward_prompt = reward_prompt_template.replace("{FRAME_COUNT}", str(total_frames))
    #     reward_prompt = reward_prompt.replace("{DURATION:.2f}", f"{duration:.2f}")
    #     reward_prompt = reward_prompt.replace("{QUESTION}", self.question)
    #     reward_prompt = reward_prompt.replace("{VIDEO_IDX}", str(self.video_idx))
    #     reward_prompt = reward_prompt.replace("{KEYFRAMES_INFO}", keyframes_info)

    #     content = []
    #     for frame in keyframes_dict[video_key]:
    #         content.append({"type": "image", "image": frame})
    #     content.append({"type": "text", "text": reward_prompt})

    #     try:
    #         messages = [{"role": "user", "content": content}]
    #         output = self.Qwen_VL(messages)
            
    #         if not output: return 0.3, "Failed output"
            
    #         lines = output.strip().split('\n')
    #         if not lines: return 0.3, "Empty output"
            
    #         # 尝试提取分数（论文格式：0-100的整数）
    #         score_line = lines[0].strip()
    #         # 优先匹配0-100的整数分数（论文格式）
    #         match = re.search(r"\b([0-9]{1,2}|100)\b", score_line)
    #         if match:
    #             score_int = int(match.group(1))
    #             score = score_int / 100.0  # 归一化到0.0-1.0
    #         else:
    #             # 回退：尝试提取浮点数
    #             match_float = re.search(r"(\d+(\.\d+)?)", score_line)
    #             if match_float:
    #                 score = float(match_float.group(1))
    #                 score = max(0.0, min(1.0, score))
    #                 if score > 1.0 and score <= 10.0: # 如果是0-10分制，归一化
    #                     score = score / 10.0
    #                 elif score > 10.0 and score <= 100.0: # 如果是0-100分制，归一化
    #                     score = score / 100.0
    #             else:
    #                 score = 0.3
            
    #         score = max(0.0, min(1.0, score))
            
    #         # 提取原因（尝试从 <reason> 标签中提取，或使用后续行）
    #         reason = ""
    #         if len(lines) > 1:
    #             reason_text = '\n'.join(lines[1:])
    #             if '<reason>' in reason_text and '</reason>' in reason_text:
    #                 start_tag = reason_text.find('<reason>')
    #                 end_tag = reason_text.find('</reason>')
    #                 reason = reason_text[start_tag + 8:end_tag].strip()
    #             else:
    #                 reason = reason_text.strip()
            
    #         # 如果分数很低，增强反馈信息的指导性
    #         if score < 0.3 and reason:
    #             if len(reason) < 50:  # 如果反馈太简短，补充指导
    #                 reason = f"Score {score:.2f}: {reason} The selected keyframes lack sufficient relevant information. Please select frames that directly show elements related to: {self.question}"
    #         elif score < 0.3 and not reason:
    #             reason = f"Score {score:.2f}: The selected keyframes do not contain sufficient information relevant to the question. Please try selecting frames that better address: {self.question}"
            
    #         return score, reason, output  # 返回原始输出
    #     except Exception as e:
    #         print(f"Error in _evaluate_keyframes_reward: {e}")
    #         return 0.3, "Error", ""

    def run_once(self, iteration_idx=0) -> Dict:
        """
        执行树状搜索探索
        """
        start_time = time.time()
        timings = {
            'initialization': {},
            'iterative_exploration': [],
            'final_output': {},
            'total': 0.0
        }
        
        duration = self._get_video_duration()
        if duration <= 0:
            return {"selected_frames": [], "score": 0.0, "detail": "Video load error", "time_elapsed": 0.0, "timings": timings}
        
        # 为本次运行创建唯一子目录，避免迭代时文件冲突和累积
        run_timestamp = int(time.time() * 1000000)  # 微秒级时间戳
        run_id = f"{run_timestamp}_{os.getpid()}"
        current_video_frames_path = os.path.join(self.video_frames_path, f"run_{run_id}")
        os.makedirs(current_video_frames_path, exist_ok=True)

        is_debug = self.config.get('parameters').get('debug', False)
        iterator_dir = None
        if is_debug:
            iterator_dir = os.path.join(self.key_frames_path, f"iterator{iteration_idx}")
            os.makedirs(iterator_dir, exist_ok=True)
        
        # --- 1. Initialization ---
        init_start = time.time()
        
        # 节点结构: (score, start_time, end_time, start_path, end_path, depth)
        # 我们使用优先队列存储待扩展节点： (-score, ...) 因为 heapq 是 min-heap
        candidate_nodes = [] 
        
        # Memory: list of segments (dict)
        memory = []
        
        initial_times = [duration * i / (self.initial_segments_n + 1) for i in range(0, self.initial_segments_n)]
        
        initial_frames = []
        
        # 提取初始帧
        frame_extraction_start = time.time()
        
        # Debug Mode: Save to step0 directory
        init_frame_dir = current_video_frames_path
        if is_debug:
            init_frame_dir = os.path.join(iterator_dir, "step0")
            os.makedirs(init_frame_dir, exist_ok=True)

        valid_initial_indices = []
        for i, t in enumerate(initial_times):
            f_path = self._extract_frame_at_time(t, f"init_{i}_{t:.2f}.jpg", init_frame_dir)
            if f_path:
                initial_frames.append(f_path)
                valid_initial_indices.append(i)
            else:
                initial_frames.append(None) # Handle error?
                print(f"[VCA Warning] Frame extraction failed for {self.video_name} at time {t:.2f}s (index {i}).")

        frame_extraction_end = time.time()
        timings['initialization']['frame_extraction'] = frame_extraction_end - frame_extraction_start
        timings['initialization']['frames_extracted'] = len(valid_initial_indices)

        valid_frames = [f for f in initial_frames if f]

        if len(valid_frames) < len(initial_times):
             print(f"[VCA Warning] Initialization: Expected {len(initial_times)} frames, but only extracted {len(valid_frames)} for {self.video_name}.")
        valid_times = [initial_times[i] for i in valid_initial_indices]
        
        # 批量评估
        eval_start = time.time()
        memory_result, overall_score, init_eval_output = self._evaluate_frames_batch(valid_frames)
        eval_end = time.time()
        timings['initialization']['evaluation'] = eval_end - eval_start
        timings['initialization']['frames_evaluated'] = len(valid_frames)
        timings['initialization']['evaluation_output'] = init_eval_output
        
        # 构建初始节点和 Memory
        node_building_start = time.time()
        
        # eval_results 是段落列表。segment[k] 对应 valid_frames[k] 到 valid_frames[k+1]
        for k, segment in enumerate(memory_result):
            if k < len(valid_times) - 1:
                t_start = valid_times[k]
                t_end = valid_times[k+1]
                
                # 更新 segment 信息
                segment['start_time'] = t_start
                segment['end_time'] = t_end
                
                # 创建节点 (注意：score 取负用于最小堆)
                node = (-segment['score'], t_start, t_end, segment['start_path'], segment['end_path'], 0) # depth 0
                heapq.heappush(candidate_nodes, node)
                
                memory.append(segment)
                
        node_building_end = time.time()
        timings['initialization']['node_building'] = node_building_end - node_building_start
        init_end = time.time()
        timings['initialization']['total'] = init_end - init_start
        
        if not candidate_nodes:
            print(f"\n[VCA Warning] Initialization failed for video {self.video_idx} ({self.video_name}): No candidate nodes created.")
            print(f"  Valid init frames: {len(valid_frames)}")
            print(f"  Memory results: {len(memory_result)}")
            if memory_result:
                print(f"  First result keys: {memory_result[0].keys()}")
            print(f"  Init Eval Output Prompt (first 100 chars): {init_eval_output[:100].replace(chr(10), ' ')}...")
            
        # --- 2. Iterative Exploration ---
        iter_explore_start = time.time()
        
        # 将 Initialization 记录为 Iteration 1
        timings['iterations'] = []
        iterations_data = [] # Store non-timing info here
        
        timings['iterations'].append({
            'iteration': 1,
            'type': 'initialization',
            'frame_extraction_duration': timings['initialization']['frame_extraction'],
            'evaluation_duration': timings['initialization']['evaluation'],
            'frames_evaluated': timings['initialization']['frames_evaluated'],
            'node_building_duration': timings['initialization']['node_building'],
            'total_duration': timings['initialization']['total']
        })
        
        # Prepare detailed segments for iteration 1 from memory_result
        init_segments_detail = []
        current_step_scores = []
        best_segment_for_next = None
        max_segment_score = -1.0

        for segment in memory_result:
             s_score = segment.get('score', 0.0)
             current_step_scores.append(s_score)
             
             seg_detail = {
                 'start_time': round(segment.get('start_time', 0.0), 2),
                 'end_time': round(segment.get('end_time', 0.0), 2),
                 'score': s_score
             }
             init_segments_detail.append(seg_detail)
             
             if s_score > max_segment_score:
                 max_segment_score = s_score
                 best_segment_for_next = seg_detail

        init_step_avg = sum(current_step_scores) / len(current_step_scores) if current_step_scores else 0.0

        iterations_data.append({
            'iteration': 1,
            'type': 'initialization',
            'step_avg_score': init_step_avg,
            'overall_sufficiency': overall_score,
            'segments': init_segments_detail,
            'next_focus_segment': best_segment_for_next,
            'evaluation_output': timings['initialization'].get('evaluation_output', '')
        })
        
        step = 1 # Already done step 1 (init)
        
        # Loop condition based on user requirements:
        # 1. Must run at least min_iterations
        # 2. If score < threshold, continue until max_iterations
        
        while candidate_nodes:
            # Check stopping conditions at start of loop (before performing next step)
            # Current iteration count done is `step`
            
            # --- Move check logically BEFORE expanding new nodes ---

             # 1. Adaptively check if score is already good enough
            # Only check this if we have completed at least min_iterations
            if step >= self.min_iterations:
                # Calculate current score estimate from memory
                # (Same logic as final score calculation)
                current_positive_scores = [item['score'] for item in memory if item['score'] > 0]
                current_avg_score = sum(current_positive_scores) / len(current_positive_scores) if current_positive_scores else 0.0
                
                # Check if we should stop
                if current_avg_score >= self.score_threshold:
                    break

            # 2. Hard Check: If we reached max iterations, stop.
            if step >= self.max_iterations: # Renamed max_possible_steps to match definition
                break
            
            step_start = time.time()
            # 取出当前分数最高的段落进行扩展
            node_selection_start = time.time()
            neg_score, s_start, s_end, start_path, end_path, depth = heapq.heappop(candidate_nodes)
            score = -neg_score
            node_selection_end = time.time()
            
            # 检查节点是否仍在 memory 中 (Top K)
            # This ensures we only refine segments that are currently considered "good enough" to be in memory
            is_in_memory = False
            for m in memory:
                # 使用 1e-4 容差进行浮点数比较
                if abs(m['start_time'] - s_start) < 1e-4 and abs(m['end_time'] - s_end) < 1e-4:
                    is_in_memory = True
                    break
            
            if not is_in_memory:
                continue

            # 扩展：在 [s_start, s_end] 内细分
            # 如果区间太小，停止细分
            if (s_end - s_start) < 1.0: # 小于1秒不再细分
                continue
                
            step += 1
            current_iteration_num = step
            
            # 细分采样 M 个子帧
            sub_frame_extraction_start = time.time()
            
            # Debug Mode: Save to step{step} directory
            step_frame_dir = current_video_frames_path
            if is_debug:
                step_frame_dir = os.path.join(iterator_dir, f"step{step}")
                os.makedirs(step_frame_dir, exist_ok=True)
                
            sub_times = np.linspace(s_start, s_end, self.sub_segments_n + 3)[1:-1]
            sub_frames = []
            valid_sub_indices = []
            
            for k, st in enumerate(sub_times):
                sf_path = self._extract_frame_at_time(st, f"step{step}_{k}_{st:.2f}.jpg", step_frame_dir)
                if sf_path:
                    sub_frames.append(sf_path)
                    valid_sub_indices.append(k)
            sub_frame_extraction_end = time.time()
            # --- 评估子帧 ---
            # 即使没有提取到子帧，也可能需要重新评估父区间的细分（虽然这里我们跳过了）
            if not sub_frames:
                continue
                
            sub_eval_start = time.time()
            
            # 构建评估序列：Start Frame + Sub Frames + End Frame
            # 必须包含父节点的边界，以保证段落覆盖整个区间 [s_start, s_end]
            frames_to_eval = [start_path] + sub_frames + [end_path]
            # 对应的时间点
            times_to_eval = [s_start] + [sub_times[i] for i in valid_sub_indices] + [s_end]
            
            sub_eval_result, sub_overall, sub_eval_output = self._evaluate_frames_batch(frames_to_eval)
            
            # 计算当前步骤的平均分（用于计算提升率）
            current_step_scores = [s['score'] for s in sub_eval_result]
            current_step_avg = sum(current_step_scores) / len(current_step_scores) if current_step_scores else 0.0
            
            # --- 2.5 Detailed Segment Scoring ---
            # Create a detailed list of segments with their scores and times for this step
            segments_detail = []
            
            # frames_to_eval has length N, times_to_eval has length N
            # sub_eval_result has length N-1 (segments between frames)
            
            # Find the best segment for next iteration (highest score)
            best_segment_for_next = None
            max_segment_score = -1.0
            
            for k, segment_score_data in enumerate(sub_eval_result):
                if k < len(times_to_eval) - 1:
                    t_start = times_to_eval[k]
                    t_end = times_to_eval[k+1]
                    s_score = segment_score_data['score']
                    
                    segments_detail.append({
                        'start_time': round(t_start, 2),
                        'end_time': round(t_end, 2),
                        'score': s_score
                    })
                    
                    if s_score > max_segment_score:
                        max_segment_score = s_score
                        best_segment_for_next = {
                            'start_time': round(t_start, 2),
                            'end_time': round(t_end, 2),
                            'score': s_score
                        }

            sub_segments = sub_eval_result
            sub_eval_end = time.time()
            # Removed incorrect index access to timings['iterative_exploration'][-1]
            # 构建子节点和更新 Memory
            node_creation_start = time.time()
            for k, segment in enumerate(sub_eval_result):
                if k < len(times_to_eval) - 1:
                    t_start = times_to_eval[k]
                    t_end = times_to_eval[k + 1]

                    # 更新 segment 信息
                    segment['start_time'] = t_start
                    segment['end_time'] = t_end

                    # 创建子节点 (注意：score 取负用于最小堆)
                    sub_node = (-segment['score'], t_start, t_end, segment['start_path'], segment['end_path'], depth + 1)
                    heapq.heappush(candidate_nodes, sub_node)

                    memory.append(segment)
            node_creation_end = time.time()
            # Removed incorrect index access to timings['iterative_exploration'][-1]

            # --- 3. Memory Management ---
            # 保留分数最高的 K 个段落
            memory_management_start = time.time()
            memory.sort(key=lambda x: x['score'], reverse=True)
            if len(memory) > self.memory_size:
                memory = memory[:self.memory_size]
            memory_management_end = time.time()
           
            step_end = time.time()
            step_record = {
                'iteration': current_iteration_num,
                'type': 'refinement',
                'node_selection_duration': node_selection_end - node_selection_start,
                'sub_frame_extraction_duration': sub_frame_extraction_end - sub_frame_extraction_start,
                'sub_frames_extracted': len(sub_frames),
                'sub_evaluation_duration': sub_eval_end - sub_eval_start,
                'sub_frames_evaluated': len(frames_to_eval),
                'node_creation_duration': node_creation_end - node_creation_start,
                'memory_management_duration': memory_management_end - memory_management_start,
                'total_step_duration': step_end - step_start
            }
            step_data_record = {
                'iteration': current_iteration_num,
                'type': 'refinement',
                'step_avg_score': current_step_avg,
                'overall_sufficiency': sub_overall,
                # 'evaluation_output': sub_eval_output if 'sub_eval_output' in locals() else "",
                # Detailed information for analysis
                'segments': segments_detail,
                'next_focus_segment': best_segment_for_next
            }
            timings['iterations'].append(step_record)
            iterations_data.append(step_data_record)
        iter_explore_end = time.time()
        timings['iterative_exploration_total'] = iter_explore_end - iter_explore_start
        timings['iterative_exploration_steps'] = step
        # --- 4. Final Output ---
        final_output_start = time.time()
        
        # fiFilter memory to only include segments with score > 0
        valid_memory = [m for m in memory if m['score'] > 0]
        
        # 从 valid_memory (segments) 中提取唯一关键帧
        unique_frames = {} # path -> time
        
        for seg in valid_memory:
            unique_frames[seg['start_path']] = seg['start_time']
            unique_frames[seg['end_path']] = seg['end_time']
            
        sorting_start = time.time()
        final_selection = sorted(unique_frames.items(), key=lambda x: x[1])
        selected_paths = [p for p, t in final_selection]
        sorting_end = time.time()
        timings['final_output']['sorting'] = sorting_end - sorting_start
        
        # 复制到 key_frames 目录并重命名
        frame_processing_start = time.time()
        final_paths = []
        
        # Debug Mode: Also save final selection to iterator directory
        latest_dst_dir = os.path.join(self.key_frames_path, "latest_selection")
        os.makedirs(latest_dst_dir, exist_ok=True)
        if is_debug and iterator_dir:
            final_dst_dir_debug = os.path.join(iterator_dir, "final_selection")
            os.makedirs(final_dst_dir_debug, exist_ok=True)

        self.final_selection = final_selection

        for i, (src, t_sec) in enumerate(final_selection, 1):
            if os.path.exists(src):
                # 1. Save to main key_frames dir (Overwrite for latest status)
                # 使用时间戳作为文件名的一部分（保留两位小数）
                dst = os.path.join(latest_dst_dir, f"{i}_{t_sec:.2f}.jpg")
                shutil.copy(src, dst)
                
                final_paths.append(dst)

                # 2. Debug Mode: Save copy to iterator dir
                if is_debug and iterator_dir:
                    # Copy original
                    dst_debug = os.path.join(final_dst_dir_debug, f"{i}_{t_sec:.2f}.jpg")
                    shutil.copy(src, dst_debug)

        frame_processing_end = time.time()
        timings['final_output']['frame_processing'] = frame_processing_end - frame_processing_start
        timings['final_output']['frames_selected'] = len(final_paths)
        
        # 计算综合分数 (使用 memory 中所有段落的平均分)
        score_calculation_start = time.time()
        
        # Average of positive scores only
        positive_scores = [item['score'] for item in memory if item['score'] > 0]
        avg_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0

        # 计算每次迭代的平均分和 overall 分数
        iteration_avg_scores = [step.get('step_avg_score', 0.0) for step in iterations_data if 'step_avg_score' in step]

        # 计算提升率
        improvement_rates = []
        for i in range(1, len(iteration_avg_scores)):
            prev_score = iteration_avg_scores[i - 1]
            curr_score = iteration_avg_scores[i]
            if prev_score > 0:
                improvement_rates.append((curr_score - prev_score) / prev_score)

        # # 记录提升率的平均值
        # avg_improvement_rate = sum(improvement_rates) / len(improvement_rates) if improvement_rates else 0.0
        
        # # User Req: Record acceleration (last round's improvement rate).
        # # If only 1 round (no subsequent iterations), set to default 1.0 as per new logic.
        # if len(iterations_data) <= 1:
        #     acceleration = 1.0
        # else:
        #     # Acceleration is the improvement of the LAST step relative to the one before it.
        #     if improvement_rates:
        #         acceleration = improvement_rates[-1]
        #     else:
        #         # This can happen if the previous score was 0, making improvement rate undefined.
        #         # Or if there were no iterations after the first one.
        #         # If the current score is > 0 and prev was 0, it's infinite improvement.
        #         # Let's check the scores directly.
        #         if len(iteration_avg_scores) > 1:
        #             prev_score = iteration_avg_scores[-2]
        #             curr_score = iteration_avg_scores[-1]
        #             if prev_score > 0:
        #                 acceleration = (curr_score - prev_score) / prev_score
        #             elif curr_score > 0:
        #                 acceleration = 1.0 # High improvement from zero
        #             else:
        #                 acceleration = 0.0
        #         else:
        #             acceleration = 0.0 # No basis for calculation

        score_calculation_end = time.time()
        timings['final_output']['score_calculation'] = score_calculation_end - score_calculation_start

        # 得分细节
        detail = f"Final Average Score: {avg_score:.2f}\n"

        # 显示 Top Segments
        for item in memory:
            detail += f"- Segment {item['start_time']:.2f}-{item['end_time']:.2f}s (Score: {item['score']:.2f})\n"


        final_output_end = time.time()
        timings['final_output']['total'] = final_output_end - final_output_start
        
        elapsed = time.time() - start_time
        timings['total'] = elapsed
        
        # 清理本次运行的临时帧文件（保留关键帧）
        if not self.config['parameters'].get('save_video_frames', False):
            try:
                if os.path.exists(current_video_frames_path):
                    shutil.rmtree(current_video_frames_path)
                # 不要删除父目录，因为其他 run_id 可能正在使用
                # if os.path.exists(os.path.dirname(current_video_frames_path)):
                #     shutil.rmtree(os.path.dirname(current_video_frames_path))
            except Exception as e:
                # 如果清理失败，记录警告但不影响结果
                print(f"Warning: Failed to clean up temporary frames in {current_video_frames_path}: {e}")
        
        # 注意：VCA 类内部不应该删除 key_frames，因为 AgentRunner 还需要使用它们来生成最终答案。
        # 如果需要清理，应该在 AgentRunner 完成整个任务后进行。
        # 这里只清理 debug 模式下产生的额外文件（如果配置为不保存）
        if not self.config['parameters'].get('save_key_frames', False):
            try:
                if is_debug and iterator_dir and os.path.exists(iterator_dir):
                    shutil.rmtree(iterator_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up debug iterator dir {iterator_dir}: {e}")
        
        return {
            "selected_frames": final_paths,
            "score": avg_score,
            "overall_sufficiency": overall_score,
            # "improvement_rate": avg_improvement_rate,
            # "acceleration": acceleration, # Added acceleration
            "detail": detail,
            "time_elapsed": elapsed,
            "timings": timings,
            "iterations_data": iterations_data
        }
