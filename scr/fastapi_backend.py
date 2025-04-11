from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import numpy as np
from typing import Dict
import cv2
import pymysql
from passlib.hash import bcrypt
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import openai

# FastAPI Backend Setup
app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# OpenAI API Key (ensure that this key is kept secure)
openai.api_key = "sk-your-openai-api-key"

# MySQL connection setup (ensure you handle database errors or use an ORM like SQLAlchemy)
db_connection = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="Mahesh2005",
    database="chatbot"
)

# Function to get user details from MySQL
def get_user(username: str):
    try:
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
            user = cursor.fetchone()
        return user
    except Exception as e:
        print(f"Error retrieving user: {e}")
        return None

# Function to verify user password
def verify_password(stored_password, provided_password):
    return bcrypt.verify(provided_password, stored_password)

# Object detection function using YOLOv5
def detect_objects(image: Image.Image) -> Dict[str, str]:
    image_np = np.array(image)
    results = model(image_np)
    detected_objects = results.pandas().xyxy[0]

    if detected_objects.empty:
        return {"description": "No objects detected. 😔", "details": []}

    objects_description = ', '.join(detected_objects['name'].unique())
    details = detected_objects[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].to_dict(orient='records')

    return {
        "description": objects_description,
        "details": details
    }

# Webcam endpoint for object detection
@app.post("/api/v1/webcam")
async def webcam(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((640, 480))
        result = detect_objects(image)
        return {"detected_objects": result['description']}
    except Exception as e:
        return {"error": str(e)}

# VQA endpoint to detect objects and respond to a question
@app.post("/api/v1/ask")
async def ask(file: UploadFile = File(...), question: str = '') -> Dict[str, str]:
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((640, 480))
        result = detect_objects(image)
        detected_objects = result['description']
        vqa_response = f"Detected objects include: {detected_objects}. You asked: '{question}'"
        return {
            "detected_objects": detected_objects,
            "vqa_response": vqa_response
        }
    except Exception as e:
        return {"error": str(e)}

# Chat with GPT-4 endpoint
@app.post("/api/v1/chatgpt")
async def chat_with_gpt(prompt: str) -> Dict[str, str]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message['content']
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}

# CORS Middleware for allowing frontend like Streamlit to communicate with FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Running the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
