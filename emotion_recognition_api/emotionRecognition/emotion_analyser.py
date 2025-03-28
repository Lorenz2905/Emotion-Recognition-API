from abc import ABC, abstractmethod
from fastapi.responses import StreamingResponse
from typing import List

class EmotionAnalyzer(ABC):

    @abstractmethod
    def analyze_video_emotions(self, images_path: List[str], text_message: str, system_prompt: str) -> str:
        pass

    @abstractmethod
    def analyze_stream_video_emotions(self, images_path: List[str], text_message: str, system_prompt: str) -> StreamingResponse:
        pass