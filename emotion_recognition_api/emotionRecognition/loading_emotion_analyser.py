import config_loader as config

from console_logging import log_info
from emotionRecognition.janus_emotion_analyser import JanusEmotionAnalyzer
from emotionRecognition.qwen_emotion_analyse import QwenEmotionAnalyzer
from emotionRecognition.emotion_analyser import EmotionAnalyzer

ANALYSER: EmotionAnalyzer = None


def load_analyser():
    global ANALYSER
    if ANALYSER is not None:
        return

    if config.get_use_janus():
        log_info("Use Janus Emotion Analyzer")
        ANALYSER = JanusEmotionAnalyzer()
    else:
        log_info("Use Qwen Emotion Analyzer")
        ANALYSER = QwenEmotionAnalyzer()


def get_analyser() -> EmotionAnalyzer:
    if ANALYSER is None:
        load_analyser()
    return ANALYSER