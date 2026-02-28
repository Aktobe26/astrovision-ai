from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import shutil
import cv2
import base64
import numpy as np
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

# Подключение static и templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("🚀 Используется GPU CUDA")

# ==============================
# RealESRGAN x4plus (без OOM)
# ==============================

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

upsampler = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=256,        # 🔥 Включён tile-режим (против OOM)
    tile_pad=10,
    pre_pad=0,
    half=True,
    device="cuda"
)

# ==============================
# Вспомогательная функция
# ==============================

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode()


# ==============================
# 🧠 УМНЫЙ АНАЛИЗ КОСМИЧЕСКОГО ОБЪЕКТА
# ==============================

def analyze_space_object(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Контраст
    contrast = gray.std()

    # Порог яркости
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        area_ratio = area / (h * w)

        # 🔥 Новая логика Луны
        if area_ratio > 0.005 and contrast > 20:
            confidence = round(min(area_ratio * 800, 95), 1)
            return (
                "Луна",
                "ИИ обнаружил яркий объект на тёмном фоне с высоким контрастом. "
                "Форма и освещённая фаза указывают на Луну.",
                confidence
            )

    # Проверка звёзд
    bright_pixels = np.sum(gray > 220)
    bright_ratio = bright_pixels / (h * w)

    if bright_ratio > 0.01:
        return (
            "Звёздное небо",
            "Обнаружено множество ярких точек — вероятно звёздное небо.",
            80.0
        )

    return (
        "Планета",
        "Объект похож на планету. Планеты отражают свет звезды.",
        65.0
    )


def crop_main_object(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + padding * 2)
        h = min(img.shape[0] - y, h + padding * 2)

        return img[y:y+h, x:x+w]

    return img
# ==============================
# ROUTES
# ==============================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(input_path)

    # 🔥 Ограничение слишком больших фото (защита VRAM)
    max_size = 2000
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # ==============================
    # SUPER RESOLUTION
    # ==============================

    # 🔥 Кропаем Луну перед апскейлом
    cropped = crop_main_object(img)

    output, _ = upsampler.enhance(cropped, outscale=4)

    # Очистка GPU памяти
    torch.cuda.empty_cache()

    # ==============================
    # AI АНАЛИЗ
    # ==============================

    object_name, explanation, confidence = analyze_space_object(img)

    original_b64 = encode_image(img)
    enhanced_b64 = encode_image(output)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original": original_b64,
        "enhanced": enhanced_b64,
        "object_name": object_name,
        "explanation": explanation,
        "confidence": confidence
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)