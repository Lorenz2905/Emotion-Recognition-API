import os
import config_loader as config

from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from emotionRecognition.emotion_recognition import video_emotion_analysis

router = APIRouter()

@router.post("/analyser")
async def analyser(
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    agents_behavior: str = Form(...),
):
    temp_dir = config.get_temp_dir()
    file_paths = []

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        file_paths.append(file_path)


    print(f"Dateipfade: {file_paths}")
    result = video_emotion_analysis(file_paths, prompt,agents_behavior)
    print(result)

    for file_path in file_paths:
        os.remove(file_path)


    return {"message": "Dateien hochgeladen und verarbeitet", "files": file_paths, "prompt": prompt,
            "agents_behavior": agents_behavior}
