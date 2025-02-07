from abc import ABC, abstractmethod
from typing import List

class EmotionAnalyzer(ABC):

    @abstractmethod
    def analyze_video_emotions(self, images_path: List[str], text_message: str, system_prompt: str,  streaming: bool) -> str:
        pass
