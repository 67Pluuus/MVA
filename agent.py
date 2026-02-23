import os
import cv2
import numpy as np
from PIL import Image
import re
from typing import List, Dict, Tuple, Any
import time

class ToolAgent:
    def __init__(self, model, processor, config, device_id):
        self.model = model
        self.processor = processor
        self.config = config
        self.device_id = device_id

    def Qwen_VL(self, messages, max_tokens=512):
        from utils import Qwen_VL
        return Qwen_VL(messages, self.device_id, self.config['models']['main_model_path'], max_tokens)

    def decide_action(self, question: str, current_frames_info: List[dict], video_duration: float, video_label: str = "ONE of the videos") -> Tuple[int, float, float]:
        """
        Decide the next sampling action.
        current_frames_info: [{'path': str, 'time': float}, ...]
        Returns: (option_id, target_start, target_end)
        """
        if not current_frames_info:
            return 5, 0.0, video_duration

        start_time = current_frames_info[0]['time']
        end_time = current_frames_info[-1]['time']
        is_global = (end_time - start_time) >= video_duration * 0.9

        prompt_template = self.config['prompts'].get('tool_decide_action')
        if prompt_template:
            prompt = prompt_template.replace("{QUESTION}", question) \
                                    .replace("{VIDEO_LABEL}", video_label) \
                                    .replace("{START_TIME}", f"{start_time:.2f}") \
                                    .replace("{END_TIME}", f"{end_time:.2f}") \
                                    .replace("{VIDEO_DURATION}", f"{video_duration:.2f}") \
                                    .replace("{IS_GLOBAL}", str(is_global))
        else:
             prompt = f"""
You are a Tool Agent for video analysis. Your goal is to decide the next frame sampling strategy to answer the question.
Question: "{question}"
Current receptive field: {start_time:.2f}s to {end_time:.2f}s.
Video duration: {video_duration:.2f}s.
Is global receptive field: {is_global}

Options:
1. Focus Middle, Replace current frames
2. Focus Middle, Keep current frames
3. Focus Start/End, Replace current frames (Only available if global receptive field)
4. Focus Start/End, Keep current frames (Only available if global receptive field)
5. Global Uniform Sampling with offset, Replace current frames
6. Terminate video exploration (if no more useful information can be found)

Choose an option (1-6). If choosing 1-4, also specify the target time range [start, end] within the current receptive field to sample 8 frames.
Output exactly in this format:
Option: <number>
Range: [<start>, <end>]
"""
        content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        
        output = self.Qwen_VL(messages, max_tokens=128)
        
        option = 5
        target_start = 0.0
        target_end = video_duration
        
        opt_match = re.search(r'Option:\s*(\d)', output)
        if opt_match:
            option = int(opt_match.group(1))
            
        range_match = re.search(r'Range:\s*\[([\d\.]+),\s*([\d\.]+)\]', output)
        if range_match:
            target_start = float(range_match.group(1))
            target_end = float(range_match.group(2))
            
        if option in [3, 4] and not is_global:
            option = 1 # Fallback to middle if not global
            
        return option, target_start, target_end

    def score_frames(self, question: str, frames: List[str], video_label: str = "ONE of the videos") -> List[float]:
        """
        Score each frame's importance for answering the question.
        Returns a list of 8 floats (0.01-1.00).
        """
        prompt_template = self.config['prompts'].get('tool_score_frames')
        if prompt_template:
            prompt = prompt_template.replace("{QUESTION}", question) \
                                    .replace("{VIDEO_LABEL}", video_label) \
                                    .replace("{NUM_FRAMES}", str(len(frames)))
        else:
            prompt = f"""
You are a Tool Agent. Evaluate the importance of each of the following {len(frames)} frames for answering the question.
Question: "{question}"

Output exactly {len(frames)} floating-point numbers between 0.01 and 1.00, separated by spaces.
Example output:
0.10 0.85 0.40 0.90 0.20 0.15 0.70 0.50
"""
        content = []
        for f in frames:
            content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        output = self.Qwen_VL(messages, max_tokens=128)
        
        scores = []
        matches = re.findall(r'\b(0\.\d+|1\.00?|0)\b', output)
        for m in matches:
            try:
                scores.append(float(m))
            except Exception as e:
                print(e)
                
        # Pad or truncate to match len(frames)
        while len(scores) < len(frames):
            scores.append(0.01)
        scores = scores[:len(frames)]
        
        return scores

    def generate_raw_description(self, question: str, frames: List[str], video_label: str = "ONE of the videos") -> str:
        """
        Generate a preliminary description based on the frames.
        """
        prompt_template = self.config['prompts'].get('tool_generate_raw_description')
        if prompt_template:
            prompt = prompt_template.replace("{QUESTION}", question) \
                                    .replace("{VIDEO_LABEL}", video_label)
        else:
            prompt = f"""
You are a Tool Agent. Based on the provided frames, generate a preliminary description of the visual content relevant to answering the question.
Question: "{question}"
Keep the description concise and factual.
"""
        content = []
        for f in frames:
            content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        return self.Qwen_VL(messages, max_tokens=256)


class DescAgent:
    def __init__(self, model, processor, config, device_id):
        self.model = model
        self.processor = processor
        self.config = config
        self.device_id = device_id

    def Qwen_VL(self, messages, max_tokens=4096):
        from utils import Qwen_VL
        return Qwen_VL(messages, self.device_id, self.config['models']['main_model_path'], max_tokens)

    def refine_and_evaluate(self, question: str, desc_old: str, desc_raw: str, other_descs: Dict[str, str], video_label: str = "ONE video") -> Tuple[str, float, bool, bool]:
        """
        Fuse information, score the new description, and decide termination.
        Returns: (desc_refined, score_new, video_terminated, global_terminated)
        """
        other_descs_text = "\n".join([f"{k}: {v}" for k, v in other_descs.items() if v])
        
        prompt_template = self.config['prompts'].get('desc_refine_and_evaluate')
        if prompt_template:
            prompt = prompt_template.replace("{QUESTION}", question) \
                                    .replace("{VIDEO_LABEL}", video_label) \
                                    .replace("{DESC_OLD}", desc_old) \
                                    .replace("{DESC_RAW}", desc_raw) \
                                    .replace("{OTHER_DESCS_TEXT}", other_descs_text)
        else:
            prompt = f"""
You are a Desc Agent. Your task is to fuse information, evaluate the quality of the description, and decide if we should stop exploring.
Question: "{question}"

Previous description of this video: "{desc_old}"
New preliminary description of this video: "{desc_raw}"
Descriptions of other videos:
{other_descs_text}

Tasks:
1. Information Fusion: Combine the previous and new descriptions into a refined, concise description that best answers the question.
2. Description Scoring: Assign a score (0.01-1.00) to the refined description based on how well it helps answer the question.
3. Termination Check 1 (Current Video): Have we fully mined the useful information from this video? (True/False)
4. Termination Check 2 (Global Task): Combining all videos' information, do we have enough information to answer the question completely? (True/False)

Output exactly in this format:
Refined Description: <your refined description>
Score: <score>
Video Terminated: <True/False>
Global Terminated: <True/False>
"""
        content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        
        output = self.Qwen_VL(messages, max_tokens=512)
        
        desc_refined = desc_raw
        score_new = 0.5
        video_terminated = False
        global_terminated = False
        
        desc_match = re.search(r'Refined Description:\s*(.*?)\nScore:', output, re.DOTALL)
        if desc_match:
            desc_refined = desc_match.group(1).strip()
            
        score_match = re.search(r'Score:\s*(0\.\d+|1\.00?|0)', output)
        if score_match:
            score_new = float(score_match.group(1))
            
        v_term_match = re.search(r'Video Terminated:\s*(True|False)', output, re.IGNORECASE)
        if v_term_match:
            video_terminated = v_term_match.group(1).lower() == 'true'
            
        g_term_match = re.search(r'Global Terminated:\s*(True|False)', output, re.IGNORECASE)
        if g_term_match:
            global_terminated = g_term_match.group(1).lower() == 'true'
            
        return desc_refined, score_new, video_terminated, global_terminated
