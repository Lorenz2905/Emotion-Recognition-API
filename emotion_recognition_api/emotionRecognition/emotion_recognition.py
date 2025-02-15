from emotionRecognition.emotion_analyser import EmotionAnalyzer
from emotionRecognition.janus_emotion_analyser import JanusEmotionAnalyzer
from emotionRecognition.qwen_emotion_analyse import QwenEmotionAnalyzer
from console_logging import log_info, log_error
import config_loader as config
import time

ANALYSER = None


def load_analyser():
    global ANALYSER
    if ANALYSER is not None:
        return

    if config.get_use_janus():
        ANALYSER = JanusEmotionAnalyzer()
    else:
        ANALYSER = QwenEmotionAnalyzer()


def video_emotion_analysis(temp_file_paths: list[str],user_prompt: str, system_prompt: str):
    log_info("starting emotion analysis")
    global ANALYSER

    result = None

    if isinstance(ANALYSER, EmotionAnalyzer):
        log_info("Calling ANALYSER")
        result = ANALYSER.analyze_video_emotions(temp_file_paths,user_prompt,system_prompt, False)
    else:
        log_error("Can not found an emotion analyser")

    return result


def video_emotion_analysis_stream(temp_file_paths: list[str],user_prompt: str, system_prompt: str):
    log_info("starting emotion analysis stream")
    global ANALYSER

    result = None

    if isinstance(ANALYSER, EmotionAnalyzer):
        log_info("Calling ANALYSER as stream")
        result = ANALYSER.analyze_video_emotions(temp_file_paths,user_prompt,system_prompt, True)
    else:
        log_error("Can not found an emotion analyser")

    return result