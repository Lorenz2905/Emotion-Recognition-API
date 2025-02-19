from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import os
from fastapi.responses import StreamingResponse
import config_loader as config
from emotionRecognition.emotion_recognition import video_emotion_analysis_stream

router = APIRouter()


@router.post("/stream_analyser")
async def stream_analyser(
        files: List[UploadFile] = File(...),
        prompt: str = Form("Analyze the emotions in the images."),
        agents_behavior: str = Form("You are an assistant for emotion recognition"),
) -> StreamingResponse:
    temp_dir = config.get_temp_dir()
    file_paths = []

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        file_paths.append(file_path)

        with open(file_path, "wb") as f:
            f.write(await file.read())

    output_stream = video_emotion_analysis_stream(file_paths, prompt, agents_behavior)
    print(output_stream)

    for file_path in file_paths:
        os.remove(file_path)

    return output_stream
