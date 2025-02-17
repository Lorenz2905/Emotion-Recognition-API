from typing import List

from fastapi.responses import StreamingResponse
from emotionRecognition.emootion_analyser_utils import check_service, qwen_massage_generator, stream_generator
from emotionRecognition.emotion_analyser import EmotionAnalyzer
from openai import OpenAI
import config_loader as config

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
        messages = qwen_massage_generator(images_path, text_message, system_prompt)

        chat_response = self.client.chat.completions.create(
            model=config.get_qwen_model_path(),
            messages=messages,
        )

        return chat_response

    def analyze_stream_video_emotions(self, images_path: List[str], text_message: str, system_prompt: str) -> StreamingResponse:
        messages = qwen_massage_generator(images_path, text_message, system_prompt)

        response_stream = self.client.chat.completions.create(
            model=config.get_qwen_model_path(),
            messages=messages,
            stream=True,
        )

        return StreamingResponse(stream_generator(response_stream), media_type="text/plain")
