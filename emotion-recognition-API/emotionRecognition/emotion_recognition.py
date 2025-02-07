from emotionRecognition.qwen_emotion_analyse import EmotionAnalyzer
from console_logging import log_info

ANALYSER = None


def load_analyser():
    global ANALYSER
    ANALYSER = EmotionAnalyzer()


def video_emotion_analysis(temp_file_paths: list[str],user_prompt: str, system_prompt: str):
    log_info("starting emotion analysis")
    global ANALYSER

    result = None

    if isinstance(ANALYSER, EmotionAnalyzer):
        log_info("Calling QWEN2.5-VL")
        result = ANALYSER.analyze_video_emotions(temp_file_paths,user_prompt,system_prompt)
    else:
        load_analyser()
        video_emotion_analysis(temp_file_paths,user_prompt,system_prompt)

    return result