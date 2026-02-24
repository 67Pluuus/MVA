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

    def decide_action(self, question: str, frame_bank: List[Tuple[str, float, float]], current_frames_info: List[dict], video_duration: float, video_label: str = "ONE of the videos") -> Tuple[List[float], int, float, float]:
        """
        Score newly sampled frames (current_frames_info) AND decide the next sampling action in one go.
        
        frame_bank: [(path, old_score, time), ...] - The accumulated best frames (Already scored). used for context.
        current_frames_info: [{'path': str, 'time': float}, ...] - The frames from the LAST action (current receptive field), which need SCORING.
        
        Returns: (scores_for_current_frames, option_id, target_start, target_end)
        """
        # Prepare inputs
        # 1. We want to score `current_frames_info`. So we must show these images.
        # 2. We use `frame_bank` as context (maybe text or limit images?). 
        #    To save tokens/complexity, we might just list the frame bank timestamps for context, 
        #    since `current_frames_info` are the ones needing visual evaluation.
        
        current_paths = [x['path'] for x in current_frames_info]
        bank_paths = [x[0] for x in frame_bank] # Optional: visual context from bank? 
        # If we include ALL bank images + current images, it might be too many (16+8=24).
        # Strategy: Show CURRENT frames visually. Describe BANK frames textually (or skip visual context of bank if valid).
        # Considering the user wants to score the *new* frames, let's focus on them.
        
        start_time = 0.0
        end_time = video_duration
        if current_frames_info:
            start_time = current_frames_info[0]['time']
            end_time = current_frames_info[-1]['time']
            
        is_global = (end_time - start_time) >= video_duration * 0.9

        prompt_template = self.config['prompts'].get('tool_combined_action')
        
        # Build text context for the frame bank
        bank_context_str = "None"
        if frame_bank:
            bank_times = [f"{x[2]:.2f}s(Score:{x[1]:.2f})" for x in frame_bank]
            bank_context_str = ", ".join(bank_times)

        if prompt_template:
            # You would update prompts.yaml to support this
            prompt = prompt_template.replace("{QUESTION}", question) \
                                    .replace("{VIDEO_LABEL}", video_label) \
                                    .replace("{START_TIME}", f"{start_time:.2f}") \
                                    .replace("{END_TIME}", f"{end_time:.2f}") \
                                    .replace("{IS_GLOBAL}", str(is_global))
        else:
             prompt = f"""
You are a Tool Agent. 
Task 1: Evaluate the importance of each of the {len(current_paths)} provided frames (the newly sampled batch) for answering the question.
Task 2: Decide the next frame sampling strategy based on the current exploration.

Question: "{question}"
Current receptive field (just explored): {start_time:.2f}s to {end_time:.2f}s.
Video duration: {video_duration:.2f}s.
Is global receptive field: {is_global}

Existing Frame Bank (Best frames found so far): {bank_context_str}

Options:
1. Focus Middle, Replace current frames
2. Focus Middle, Keep current frames
3. Focus Start/End, Replace current frames (Only available if global receptive field)
4. Focus Start/End, Keep current frames (Only available if global receptive field)
5. Global Uniform Sampling with offset, Replace current frames
6. Terminate video exploration (if no more useful information can be found)

Output format:
Scores: <space_separated_scores_for_current_frames>
Decision: Option <number> [Range: <start>, <end>] (Range is optional for Option 5/6)
"""
        content = []
        # Add visual content ONLY for current frames (to score them)
        for f in current_paths:
            content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        output = self.Qwen_VL(messages, max_tokens=256)
        
        # Parse Scores
        scores = []
        score_line_match = re.search(r'Scores:\s*([0-9\.\s]+)', output)
        if score_line_match:
            raw_scores = score_line_match.group(1).strip()
            matches = re.findall(r'\b(0\.\d+|1\.00?|0)\b', raw_scores)
            for m in matches:
                try:
                    scores.append(float(m))
                except: pass
        
        # Pad/Truncate scores
        while len(scores) < len(current_paths):
            scores.append(0.5) # Default neutral score if missing
        scores = scores[:len(current_paths)]
        
        # Parse Decision
        option = 5
        target_start = 0.0
        target_end = video_duration
        
        decision_line_match = re.search(r'Decision:.*', output)
        if decision_line_match:
            decision_text = decision_line_match.group(0)
            
            opt_match = re.search(r'Option\s*(\d)', decision_text)
            if opt_match:
                option = int(opt_match.group(1))
                
            # Improved regex to handle various separators (comma, space, hyphen) and optional brackets
            range_match = re.search(r'Range:.*?([\d\.]+)[,\s\-]+([\d\.]+)', decision_text)
            if range_match:
                target_start = float(range_match.group(1))
                target_end = float(range_match.group(2))
        
        if option in [3, 4] and not is_global:
            option = 1
            
        return scores, option, target_start, target_end





class DescAgent:
    def __init__(self, model, processor, config, device_id):
        self.model = model
        self.processor = processor
        self.config = config
        self.device_id = device_id

    def Qwen_VL(self, messages, max_tokens=4096):
        from utils import Qwen_VL
        return Qwen_VL(messages, self.device_id, self.config['models']['main_model_path'], max_tokens)

    def describe_and_evaluate(self, question: str, frames: List[str], desc_old: str, other_descs: Dict[str, str], video_label: str = "ONE of the videos") -> Tuple[str, float, bool, bool]:
        """
        Generate a description from frames AND evaluate status (score, termination) in one go.
        frames: New frames to describe.
        desc_old: The accumulated description.
        other_descs: Descriptions of other videos.
        
        Returns: (desc_new_part, score_new, video_terminated, global_terminated)
        """
        other_descs_text = "\n".join([f"{k}: {v}" for k, v in other_descs.items() if v])
        
        prompt_template = self.config['prompts'].get('desc_combined_action')
        # If no template, use default
        if prompt_template:
            prompt = prompt_template.replace("{QUESTION}", question) \
                                    .replace("{VIDEO_LABEL}", video_label) \
                                    .replace("{DESC_OLD}", desc_old) \
                                    .replace("{OTHER_DESCS_TEXT}", other_descs_text)
        else:
            prompt = f"""
You are a Desc Agent.
Task 1: Generate a concise description of the new visual content from the provided frames, relevant to the Question.
Task 2: Evaluate the quality of the FULL description (Previous + New) and decide if we should stop exploring.

Question: "{question}"
Previous description of this video: "{desc_old}"
Descriptions of other videos:
{other_descs_text}

Output format:
New Description: <concise description of provided frames>
---
Score: <score 0.01-1.00 for the FULL description>
Video Terminated: <True/False>
Global Terminated: <True/False>
"""
        content = []
        for f in frames:
            content.append({"type": "image", "image": f})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        output = self.Qwen_VL(messages, max_tokens=384)
        
        desc_new = "No new description."
        score_new = 0.5
        video_terminated = False
        global_terminated = False
        
        # Parse Description
        desc_match = re.search(r'New Description:\s*(.*?)---', output, re.DOTALL)
        if desc_match:
            desc_new = desc_match.group(1).strip()
        else:
            # Fallback if separator missing
            desc_match = re.search(r'New Description:\s*(.*)', output, re.DOTALL)
            if desc_match:
                # Be careful not to include Score line
                all_text = desc_match.group(1)
                score_idx = all_text.find("Score:")
                if score_idx != -1:
                    desc_new = all_text[:score_idx].strip()
                else:
                    desc_new = all_text.strip()
                    
        # Parse Status
        score_match = re.search(r'Score:\s*(0\.\d+|1\.00?|0)', output)
        if score_match:
            try:
                score_new = float(score_match.group(1))
            except: pass
            
        v_term_match = re.search(r'Video Terminated:\s*(True|False)', output, re.IGNORECASE)
        if v_term_match:
            video_terminated = v_term_match.group(1).lower() == 'true'
            
        g_term_match = re.search(r'Global Terminated:\s*(True|False)', output, re.IGNORECASE)
        if g_term_match:
            global_terminated = g_term_match.group(1).lower() == 'true'
            
        return desc_new, score_new, video_terminated, global_terminated
        
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

    def evaluate_status(self, question: str, description: str, other_descs: Dict[str, str], video_label: str = "ONE video") -> Tuple[float, bool, bool]:
        """
        Evaluate the current description and decide termination.
        Returns: (score, video_terminated, global_terminated)
        """
        other_descs_text = "\n".join([f"{k}: {v}" for k, v in other_descs.items() if v])
        
        prompt_template = self.config['prompts'].get('desc_evaluate_status')
        if prompt_template:
            prompt = prompt_template.replace("{QUESTION}", question) \
                                    .replace("{VIDEO_LABEL}", video_label) \
                                    .replace("{DESCRIPTION}", description) \
                                    .replace("{OTHER_DESCS_TEXT}", other_descs_text)
        else:
            prompt = f"""
You are a Desc Agent. Your task is to evaluate the quality of the description and decide if we should stop exploring.
Question: "{question}"

Current description of this video: "{description}"
Descriptions of other videos:
{other_descs_text}

Tasks:
1. Description Scoring: Assign a score (0.01-1.00) to the current description based on how well it helps answer the question.
2. Termination Check 1 (Current Video): Have we fully mined the useful information from this video? (True/False)
3. Termination Check 2 (Global Task): Combining all videos' information, do we have enough information to answer the question completely? (True/False)

Output exactly in this format:
Score: <score>
Video Terminated: <True/False>
Global Terminated: <True/False>
"""
        content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        
        output = self.Qwen_VL(messages, max_tokens=128)
        
        score_new = 0.5
        video_terminated = False
        global_terminated = False
        
        score_match = re.search(r'Score:\s*(0\.\d+|1\.00?|0)', output)
        if score_match:
            try:
                score_new = float(score_match.group(1))
            except ValueError:
                pass
            
        v_term_match = re.search(r'Video Terminated:\s*(True|False)', output, re.IGNORECASE)
        if v_term_match:
            video_terminated = v_term_match.group(1).lower() == 'true'
            
        g_term_match = re.search(r'Global Terminated:\s*(True|False)', output, re.IGNORECASE)
        if g_term_match:
            global_terminated = g_term_match.group(1).lower() == 'true'
            
        return score_new, video_terminated, global_terminated
