import base64
from fastapi import APIRouter, HTTPException
from emotionRecognition.loading_emotion_analyser import get_analyser
from endpoints.model import AnalyseRequest

router = APIRouter()
ANALYSER = get_analyser()

@router.post("/analyser")
async def analyser(request: AnalyseRequest):
    for message_index, message in enumerate(request.messages):
        for content_index, item in enumerate(message.content):
            if item.type == "image":
                image_data = item.image

                if not image_data or not image_data.startswith("data:image;base64,"):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Image does not start with 'data:image;base64,'.",
                            "message_index": message_index,
                            "content_index": content_index
                        }
                    )

                base64_str = image_data.split(",", 1)[1]

                try:
                    base64.b64decode(base64_str, validate=True)
                except (base64.binascii.Error, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Invalid base64 encoding.",
                            "message_index": message_index,
                            "content_index": content_index
                        }
                    )

    result = ANALYSER.analyze_video_emotions(request)
    return result
