
# from zai import ZhipuAiClient
import base64
import cv2
import os
from functools import lru_cache

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 全局模型缓存，按GPU ID存储
_model_cache = {}
_processor_cache = {}

def init(model_path: str="Qwen3-VL-2B-Instruct", device_id: int=None):
    """
    初始化模型和处理器
    
    Args:
        model_path: 模型路径
        device_id: GPU设备ID (0-7)，如果为None则使用device_map="auto"
    """
    # 如果指定了device_id，使用缓存
    if device_id is not None:
        cache_key = f"{model_path}_{device_id}"
        if cache_key in _model_cache:
            return _model_cache[cache_key], _processor_cache[cache_key]
    
    # 设置设备映射
    if device_id is not None:
        device_map = f"cuda:{device_id}"
    else:
        device_map = "auto"
    
    # Check if model exists in cache to avoid reloading even without device_id if previously loaded with same params
    # But simplifying: always trust cache if device_id provided.
    
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            dtype="auto",
            device_map=device_map
        )

        processor = AutoProcessor.from_pretrained(model_path)
    except Exception as e:
        # Fallback or error handling
        print(f"Error loading model {model_path}: {e}")
        raise e
    
    # 缓存模型和处理器
    if device_id is not None:
        cache_key = f"{model_path}_{device_id}"
        _model_cache[cache_key] = model
        _processor_cache[cache_key] = processor
    
    return model, processor


def Qwen_VL(messages, device_id=None, model_path="Qwen3-VL-2B-Instruct", max_tokens=2048):
    model, processor = init(model_path=model_path, device_id=device_id)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""



def DescribeVideo_qwen3_vl(video_path, question, sampled_frame=None, prompt_template=None, device_id=None, model_path="Qwen3-VL-2B-Instruct", node_rank=None, temp_base_dir=None, video_label="ONE of the videos"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to open video file")
        return "Video load failed."
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Safety check for empty video
    if frame_count == 0:
        return "Empty video."
        
    frame_indices = [int(frame_count * i / sampled_frame) for i in range(sampled_frame)]
    
    # Create unique temp directory
    # Format: [base]/[video_name]_[timestamp]_[node]_[gpu] to ensure uniqueness
    import time
    timestamp = int(time.time() * 1000)
    
    if temp_base_dir:
        # If base dir provided, use it
        dir_name = f"{video_name}_{timestamp}_n{node_rank}_g{device_id}"
        temp_dir = os.path.join(temp_base_dir, dir_name)
    else:
        # Valid fallback if not using new log structure
        pid = os.getpid()
        if node_rank is not None and device_id is not None:
            temp_dir = f"temp_frames_node_{node_rank}_gpu_{device_id}_{pid}"
        elif device_id is not None:
            temp_dir = f"temp_frames_gpu_{device_id}_{pid}"
        else:
            temp_dir = f"temp_frames_{pid}"
    
    os.makedirs(temp_dir, exist_ok=True)
    saved_frames = []
    output_text = "Error generating description."
    
    try:
        try:
            for idx, frame_idx in enumerate(frame_indices, 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(temp_dir, f"frame_{idx}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved_frames.append(frame_path)
        finally:
            cap.release()

        content = []
        for i in saved_frames:
            content.append({
                "type": "image",
                "image": i
            })
        
        if prompt_template:
            prompt_text = prompt_template.replace("{QUESTION}", question).replace("{VIDEO_LABEL}", video_label)
        else:
            prompt_text = f"Describe the video content relevant to the question: {question}"

        content.append({
            "type": "text",
            "text": prompt_text
        })

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        try:
            output_text = Qwen_VL(messages, device_id=device_id, model_path=model_path, max_tokens=512)
        except Exception as e:
            print(f"Error in DescribeVideo: {e}")
            
    finally:
        # Cleanup - 确保临时目录被清理，即使在异常情况下也尝试清理
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            # 如果清理失败，至少记录警告（多进程环境下可能重复打印）
            print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")
        
    return output_text

def answer(video_frames, question, options, prompt_template=None, device_id=None, model_path="Qwen3-VL-2B-Instruct"):
    """
    Generate an answer based on video frames and a question.
    
    Args:
        video_frames: List of image paths or PIL images.
        question: The question string.
        options: The options string (e.g., "A. ...\nB. ...").
        prompt_template: Optional prompt template. Can contain {QUESTION} and {OPTIONS} placeholders.
                        If provided, it replaces the default prompt construction.
        device_id: GPU ID.
        model_path: Model path.
    """
    # User provided template. Try to format it if it has placeholders.
    # Use safe formatting to avoid errors if keys are missing in template but present in args, or vice-versa
    try:
        # Check if template expects formatting
        if "{QUESTION}" in prompt_template or "{OPTIONS}" in prompt_template:
            prompt_text = prompt_template.replace("{QUESTION}", question).replace("{OPTIONS}", options)
            # Remove {FRAMES} placeholder if present, as frames are passed as images
            prompt_text = prompt_text.replace("{FRAMES}", "")
            # Remove {BBOX} placeholder if present (currently not supported by this function, might need to add if needed)
            prompt_text = prompt_text.replace("{BBOX}", "") 
        else:
            # If no standard placeholders, treat as prefix and append question/options
            prompt_text = f"{prompt_template}\n\nQuestion: {question}\nOptions:\n{options}"
    except Exception as e:
        print(f"Warning: Failed to format prompt template: {e}")
        prompt_text = f"{prompt_template}\n\nQuestion: {question}\nOptions:\n{options}"

    content = [
        {
            "type": "text",
            "text": prompt_text
        }
    ]
    
    for frame in video_frames:
        content.append({
            "type": "image",
            "image": frame
        })
        
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    try:
        output_text = Qwen_VL(messages, device_id=device_id, model_path=model_path, max_tokens=512)
    except Exception as e:
        print(f"Error in answer generation: {e}")
        output_text = "Error generating answer."

    return output_text
