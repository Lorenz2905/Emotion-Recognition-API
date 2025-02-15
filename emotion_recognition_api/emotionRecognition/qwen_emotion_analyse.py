from emotionRecognition.emotion_analyser import EmotionAnalyzer
import subprocess
from openai import OpenAI


class QwenEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self):
        cmd = [
        "vllm", "serve", "Qwen/Qwen2.5-VL-7B-Instruct",
        "--port", "8001",
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--limit-mm-per-prompt", "image=5,video=5"
        ]

        process = subprocess.Popen(cmd)
        print(f"Server läuft mit PID: {process.pid}")



    def analyze_video_emotions(self, images_path: list[str], text_message: str, system_prompt: str, streaming: bool = False):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )


        content = [{"type": "image", "image": f"file://{path}"} for path in images_path]
        content.append({"type": "text", "text": text_message})


        messages = [
            {
                "role": "user",
                "content": content,
            },
            {"role": "system", "content": system_prompt},

        ]
        print(messages)

        chat_response = client.chat.completions.create(
            model="Qwen2.5-VL-7B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": messages
                },
            ],
        )

        return chat_response