from typing import List

from fastapi.responses import StreamingResponse
from emotionRecognition.emootion_analyser_utils import check_service
from emotionRecognition.emotion_analyser import EmotionAnalyzer
from openai import OpenAI
import config_loader as config
import base64


def stream_generator(response_stream):
    for chunk in response_stream:
        if chunk.choices:
            yield chunk.choices[0].delta.content

def _generate_massage(images_path: list[str], text_message: str, system_prompt: str):
    content = []

    for image_path in images_path:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        content.append({"type": "image_url", "image_url": {"url": base64_qwen}})

    content.append({"type": "text", "text": text_message})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return messages

class QwenEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self):
        check_service()

        openai_api_key = config.get_api_key()
        openai_api_base = config.get_api_base_url()

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )


    def analyze_video_emotions(self, images_path: list[str], text_message: str, system_prompt: str):
        messages = _generate_massage(images_path, text_message, system_prompt)

        chat_response = self.client.chat.completions.create(
            model=config.get_qwen_model_path(),
            messages=messages,
        )

        return chat_response

    def analyze_stream_video_emotions(self, images_path: List[str], text_message: str, system_prompt: str) -> StreamingResponse:
        messages = _generate_massage(images_path, text_message, system_prompt)

        response_stream = self.client.chat.completions.create(
            model=config.get_qwen_model_path(),
            messages=messages,
            stream=True,
        )

        return StreamingResponse(stream_generator(response_stream), media_type="text/plain")
