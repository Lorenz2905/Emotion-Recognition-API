from emotionRecognition.emotion_analyser import EmotionAnalyzer
import subprocess
from openai import OpenAI
import base64
import requests

def check_service():
    url = "http://localhost:8001/v1/models"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Service erreichbar!")
            print("📌 Geladene Modelle:", response.json())
        else:
            print(f"⚠️ Service antwortet, aber Fehler: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("❌ vLLM-Service nicht erreichbar!")


class QwenEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self):
        pass


    def analyze_video_emotions(self, images_path: list[str], text_message: str, system_prompt: str, streaming: bool = False):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )


        content = []

        for image_path in images_path:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read())
            encoded_image_text = encoded_image.decode("utf-8")
            base64_qwen = f"data:image;base64,{encoded_image_text}"
            content.append({"type": "image_url", "image_url": {"url":base64_qwen}})

        content.append({"type": "text", "text": text_message})


        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        print("messages:")
        print(messages)

        print("test if servis is available")
        check_service()
        print("test done")

        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages,
        )

        return chat_response