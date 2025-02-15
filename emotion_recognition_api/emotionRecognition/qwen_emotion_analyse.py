from config_loader import CONFIG
from emotionRecognition.emootion_analyser_utils import check_service
from emotionRecognition.emotion_analyser import EmotionAnalyzer
from openai import OpenAI
import config_loader as config
import base64


class QwenEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self):
        check_service()

        openai_api_key = config.get_api_key_url()
        openai_api_base = config.get_api_base_url()

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )


    def analyze_video_emotions(self, images_path: list[str], text_message: str, system_prompt: str, streaming: bool = False):
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

        chat_response = self.client.chat.completions.create(
            model=config.get_qwen_model_path(),
            messages=messages,
        )

        return chat_response