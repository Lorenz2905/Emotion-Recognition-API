from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import os
from fastapi.responses import StreamingResponse
import config_loader as config
from emotionRecognition.loading_emotion_analyser import get_analyser


router = APIRouter()

ANALYSER = get_analyser()

@router.post("/stream_analyser")
async def stream_analyser(
        files: List[UploadFile] = File(...),
        prompt: str = Form("Analyze the emotions in the images."),
        agents_behavior: str = Form("You are an assistant for emotion recognition"),
) -> StreamingResponse:
    global ANALYSER
    temp_dir = config.get_temp_dir()
    file_paths = []

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        file_paths.append(file_path)

        with open(file_path, "wb") as f:
            f.write(await file.read())

    return ANALYSER.analyze_stream_video_emotions(file_paths, prompt, agents_behavior)
