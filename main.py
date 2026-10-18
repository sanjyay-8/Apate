
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from ai.inference import analyze_image
from ai.news_checker import analyze_news

app = FastAPI(title="APATE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "APATE API is running"
    }


@app.post("/analyze/image")
async def image_check(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        print("IMAGE RECEIVED:", image.size, image.mode)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    return analyze_image(contents)


@app.post("/analyze/news")
def news_check(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please enter news text.")

    return analyze_news(request.text)
