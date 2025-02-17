from contextlib import asynccontextmanager

from fastapi import FastAPI

from emotionRecognition.loading_emotion_analyser import load_analyser
from endpoints.analysis import router as analysis_endpoint
from endpoints.stream_analysis import router as analysis_stream_endpoint
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

app.include_router(analysis_endpoint, tags=["Multi Image Analysis"])
app.include_router(analysis_stream_endpoint, tags=["Multi Image Analysis"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
