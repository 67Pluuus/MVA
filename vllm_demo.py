from openai import OpenAI
# Configured by environment variables
api_key = "sk-abc123"
api_base = "http://localhost:8007/v1"
client = OpenAI(
    api_key = api_key,
    base_url = api_base
)
messages = [
    {"role": "user", "content": "Give me a short introduction to large language models."},
]

chat_response = client.chat.completions.create(
    model="Qwen2.5-VL-7B-Instruct",
    messages=messages,
    max_tokens=32768,
    temperature=0.0,
    top_p=1.0,
    presence_penalty=2.0,
    extra_body={
        "top_k": 20,
    }, 
)

model_output = chat_response.choices[0].message.content

print("Model output:", model_output)