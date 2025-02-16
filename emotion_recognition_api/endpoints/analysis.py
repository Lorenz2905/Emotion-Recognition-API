import os
import config_loader as config

from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from main import ANALYSER

router = APIRouter()

@router.post("/analyser")
async def analyser(
    files: List[UploadFile] = File(...),
    prompt: str = Form("Analyze the emotions in the images."),
    agents_behavior: str = Form("You are an assistant for emotion recognition"),
):
    temp_dir = config.get_temp_dir()
    file_paths = []

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        file_paths.append(file_path)

    result = ANALYSER.analyze_video_emotions(file_paths, prompt, agents_behavior)

    for file_path in file_paths:
        os.remove(file_path)


    return result
