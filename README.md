#  AstroVision AI

AI-система улучшения и анализа астрономических изображений.

Проект использует:

-  RealESRGAN (повышение качества изображений)
-  PyTorch + GPU (ускорение вычислений)
-  FastAPI (веб-сервер)
-  HTML/CSS/JS (веб-интерфейс)

---

#  Возможности системы

✔ Улучшение качества космических изображений (x4)  
✔ Использование GPU CUDA  
✔ Автоматическое определение объекта (Луна / Планета / Звезда)  
✔ Облачный запуск  
✔ Публичный доступ через туннель  

---

# Архитектура проекта

```
Пользователь → FastAPI → RealESRGAN (GPU) → Анализ изображения → Результат
```

---

#  Установка проекта (локально)

## 1️ Клонировать репозиторий

```bash
git clone https://github.com/Aktobe26/astrovision-ai.git
cd astrovision-ai
```

## 2️ Создать виртуальное окружение

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```

## 3️ Установить зависимости

```bash
pip install -r requirements.txt
```

---

#  Запуск локально (без ngrok)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Открыть в браузере:

```
http://127.0.0.1:8000
```

---

#  Запуск с публичным доступом (ngrok)

## 1️ Установить ngrok

https://ngrok.com/download

## 2️ Добавить authtoken

```bash
ngrok config add-authtoken ВАШ_ТОКЕН
```

## 3️ Запустить сервер

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4️ В другом терминале

```bash
ngrok http 8000
```

Появится публичная ссылка:

```
https://xxxx.ngrok-free.app
```

---

#  Запуск в Google Colab (GPU)

1️ Включить GPU  
Среда выполнения → Сменить среду → GPU

2️ Клонировать проект

```python
!git clone https://github.com/Aktobe26/astrovision-ai.git
%cd astrovision-ai
```

3️ Установить зависимости

```python
!pip install fastapi uvicorn jinja2 python-multipart
!pip install opencv-python-headless
!pip install basicsr realesrgan
!pip install pyngrok
```

4️ Запуск с публичной ссылкой

```python
from pyngrok import ngrok
import uvicorn
import threading

def run():
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

thread = threading.Thread(target=run)
thread.start()

public_url = ngrok.connect(8000)
print("Публичная ссылка:", public_url)
```

---

#  Технологии

- Python 3.10+
- PyTorch
- CUDA
- RealESRGAN
- FastAPI
- HTML/CSS/JS

---

#  Автор
 
AstroVision AI Project  
2026
