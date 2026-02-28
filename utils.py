
# from zai import ZhipuAiClient
import base64
import cv2
import os
from functools import lru_cache

from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
        if "Qwen3" in model_path:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, 
                dtype="auto",
                device_map=device_map
            )
        elif "Qwen2.5" in model_path:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    
    # Qwen2.5 uses patch size 14, Qwen3 uses 16 
    if "Qwen2.5" in model_path:
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
    else:
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    # Optional debug printing of model I/O controlled by environment variable PRINT_MODEL_IO
    try:
        import os
        _print_io = os.environ.get('PRINT_MODEL_IO', '').lower() in ('1', 'true', 'yes')
    except Exception:
        _print_io = False

    if _print_io:
        img_paths = []
        try:
            for m in messages:
                content = m.get('content', []) if isinstance(m, dict) else []
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get('type') == 'image':
                            img_paths.append(c.get('image'))
        except Exception:
            pass

        print("=== MODEL INPUT START ===")
        print("PROMPT:")
        try:
            print(text)
        except Exception:
            print("(failed printing prompt)")
        print("IMAGES:")
        for p in img_paths:
            print(p)
        print("=== MODEL INPUT END ===")

    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    final_output = output_text[0] if output_text else ""
    try:
        import os
        if os.environ.get('PRINT_MODEL_IO', '').lower() in ('1', 'true', 'yes'):
            print("=== MODEL OUTPUT START ===")
            print(final_output)
            print("=== MODEL OUTPUT END ===")
    except Exception as e:
        print(f"Error printing model output: {e}")
    return final_output


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
