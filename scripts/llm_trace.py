import openai
import mlflow

# Calling the autolog API will enable trace logging by default.
mlflow.set_tracking_uri(uri="http://127.0.0.1:5801")
mlflow.openai.autolog()
mlflow.set_experiment("OpenAI")

openai_client = openai.OpenAI(api_key="api_key", base_url="http://localhost:8953/v1")

messages = [
    {
        "role": "user",
        "content": "How can I improve my resting metabolic rate most effectively?",
    }
]

response = openai_client.chat.completions.create(
    model="Qwen/Qwen3-8B-FP8",
    messages=messages,
    temperature=0.09,
)

print(response)
