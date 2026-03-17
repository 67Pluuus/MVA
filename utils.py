
# from zai import ZhipuAiClient
import base64
import cv2
import os
from openai import OpenAI


def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def vllm_models(messages, device_id=None, model_path="Qwen3-VL-2B-Instruct", max_tokens=2048, port=8007):
    api_key = "sk-abc123"
    api_base = f"http://localhost:{port}/v1"
    client = OpenAI(
        api_key = api_key,
        base_url = api_base
    )
    try:
        if "Qwen3.5" in model_path:
            chat_response = client.chat.completions.create(
                model = model_path,
                messages = messages,
                max_tokens = max_tokens,
                temperature = 0.0,
                top_p = 1.0,
                seed = 42,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
        else:
            chat_response = client.chat.completions.create(
                model = model_path,
                messages = messages,
                max_tokens = max_tokens,
                temperature = 0.0,
                top_p = 1.0,
                seed = 42,
            )
        model_output = chat_response.choices[0].message.content if chat_response.choices else ""
        return model_output
    except Exception as e:
        import traceback
        print("\n\n=============== [API Request Error] ===============")
        traceback.print_exc()
        print("===================================================\n")
        raise RuntimeError(f"API Error details: {str(e)}")

def answer(video_frames, question, options, prompt_template=None, device_id=None, model_path="Qwen3-VL-2B-Instruct", print_data=False, skip_iteration=False, port=8007):
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
        print_data: Whether to print the input data for debugging.
        skip_iteration: Whether to skip the iteration step.
    """
    # User provided template. Try to format it if it has placeholders.
    # Use safe formatting to avoid errors if keys are missing in template but present in args, or vice-versa
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

    if print_data:
        print("=== Initial frames in frame bank ===")
        
    if not skip_iteration:
        filtered_frames = {}
        for k, v in video_frames.items():
            filtered_frames[k] = []
            for item in v:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    path = item[0]
                    score = item[1]
                else:
                    path = item
                    score = 1.0 # Default keep for non-scored items (compatibility)
                    
                if print_data:
                    print(f"Path: {path}, Score: {score}")
                    
                if score >= 0.5:
                    filtered_frames[k].append((path, score))
        
        video_frames = filtered_frames

    if print_data:
        print("=== Answering ===")
        print("Question:", question)
        print("Options:", options)
        print(f"Number of filtered video frames: {len(video_frames)}")
        for k, v in video_frames.items():
            print(f"{k}: ")
            for frame in v:
                print(f"  Path: {frame[0]}, Score: {frame[1]}")
        print("==================")

    content = []
    
    # for k, v in video_frames.items():
    #     # print(k)
    #     content.append({
    #         "type": "text",
    #         "text": f"The following is the {k}"
    #     })
    #     for frame in v:
    #         content.append({
    #             "type": "image",
    #             "image": frame[0] if isinstance(frame, tuple) else frame
    #         })
    # content.append({
    #     "type": "text",
    #     "text": prompt_text
    # })
        
    # messages = [
    #     {
    #         "role": "user",
    #         "content": content
    #     }
    # ]
    for k, v in video_frames.items():
        # print(k)
        content.append({
            "type": "text",
            "text": f"The following is the {k}"
        })
        for frame in v:
            f_path = frame[0] if isinstance(frame, tuple) else frame
            b64_url = encode_image_base64(f_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": b64_url}
            })
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
    
    # try:
    #     output_text = Qwen_VL(messages, device_id=device_id, model_path=model_path, max_tokens=512)
    # except Exception as e:
    #     print(f"Error in answer generation: {e}")
    #     output_text = "Error generating answer."

    output_text = vllm_models(messages, device_id=device_id, model_path=model_path, max_tokens=512, port=port)

    if print_data:
        print("======Model output========\n", output_text)

    return output_text

def question_analyse(question, options, prompt_template=None, device_id=None, model_path="Qwen3-VL-2B-Instruct", print_data=False, port=8007):
    """
    Analyze the question and options to determine the strategy.
    
    Args:
        question: The question string.
        options: The options string.
        prompt_template: Template for the analysis prompt.
        device_id: GPU ID.
        model_path: Model path.
        print_data: Whether to print debug info.
    """
    if prompt_template and ("{QUESTION}" in prompt_template or "{OPTIONS}" in prompt_template):
        prompt_text = prompt_template.replace("{QUESTION}", question).replace("{OPTIONS}", options)
    else:
        prompt_text = f"{prompt_template}\n\nQuestion: {question}\nOptions:\n{options}" if prompt_template else f"Question: {question}\nOptions:\n{options}"

    if print_data:
        print("=== Question Analysis ===")
        print("Prompt:", prompt_text)
        print("=======================")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
    ]
    
    output_text = vllm_models(messages, device_id=device_id, model_path=model_path, max_tokens=512, port=port)

    if print_data:
        print("======Analysis Output========\n", output_text)

    return output_text
