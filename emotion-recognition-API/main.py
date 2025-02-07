from contextlib import asynccontextmanager

from fastapi import FastAPI
from endpoints.image_analysis import router as upload_file_endpoint
from emotionRecognition.emotion_recognition import load_analyser
from console_logging import log_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_info("Loading Analyser")
    load_analyser()
    yield


app = FastAPI(
    title="Emotion Recognition API",
    description="This is an API for emotion recognition.",
    version="0.0.1",
    docs_url="/ui",
    redoc_url="/docs",
    lifespan=lifespan,
)

app.include_router(upload_file_endpoint, tags=["Image Analysis"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
