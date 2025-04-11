from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import numpy as np
from typing import Dict
import requests
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import openai
import threading
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import pymysql
from passlib.hash import bcrypt
import streamlit as st

# FastAPI Backend Setup
app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# MySQL connection setup
db_connection = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="Mahesh2005",
    database="chatbot"
)

def get_user(username: str):
    try:
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
            user = cursor.fetchone()
        return user
    except Exception as e:
        print(f"Error retrieving user: {e}")
        return None

def verify_password(stored_password, provided_password):
    return bcrypt.verify(provided_password, stored_password)

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

# CORS Middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Streamlit Frontend Setup
def login_page():
    st.title("🔐 Login to Your Account")

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login", help="Click to login after entering your credentials"):
        user = get_user(username)
        if user and verify_password(user[2], password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.show_login = False  # Set to False to indicate successful login
            st.session_state.page = 'main'  # Set the page to main
        else:
            st.error("⚠ Invalid username or password. Please try again.")
            st.info("🔒 Forgot your password? Contact support.")

    if st.button("Create Account"):
        new_username = st.text_input("New Username", placeholder="Enter new username")
        new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
        if st.button("Register"):
            if new_username and new_password:
                try:
                    hashed_password = bcrypt.hash(new_password)
                    with db_connection.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO users (username, password) VALUES (%s, %s)",
                            (new_username, hashed_password)
                        )
                        db_connection.commit()
                    st.success("🎉 Account created successfully! You can now log in.")
                except Exception as e:
                    st.error(f"⚠ Error occurred: {e}")
            else:
                st.error("⚠ Please provide both username and password.")

def display_live_webcam_feed():
    st.subheader("🎥 Live Webcam Feed")

    class VideoProcessor(VideoProcessorBase):
        def _init_(self):
            self.backend_url = 'http://127.0.0.1:8000/api/v1/webcam'
            self.detected_objects = "No objects detected. 😔"

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            response = requests.post(
                self.backend_url,
                files={"file": ("frame.png", img_bytes, "image/png")}
            ).json()

            self.detected_objects = response.get("detected_objects", "No objects detected. 😔")

            img = cv2.putText(img, self.detected_objects, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                              cv2.LINE_AA)
            return img

    webrtc_ctx = webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

    if webrtc_ctx and webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor
        st.write(f"🔍 Detected Objects: {processor.detected_objects}")

def display_upload_and_ask():
    st.subheader("🖼 Upload Your Image & Ask a Question!")
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])
    user_query = st.text_input("💬 Ask something about the image:")

    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption='🔥 Your Uploaded Image 🔥')

            if user_query:
                with st.spinner('⏳ Processing...'):
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    response = requests.post(
                        'http://127.0.0.1:8000/api/v1/ask',
                        files={"file": ("uploaded_image.png", img_bytes, "image/png")},
                        data={"question": user_query}
                    ).json()

                    if "error" in response:
                        st.error(f"⚠ Error occurred: {response['error']}")
                    else:
                        st.write(f"🔍 Detected Objects: {response['detected_objects']}")
                        st.write(f"🗣 Response to your question: {response['vqa_response']}")
        except Exception as e:
            st.error(f"⚠ Error processing image: {e}")

def display_gpt_chatbot():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🤖 GPT-4o - ChatBot")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask GPT-4o...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                *st.session_state.chat_history
            ]
        )

        assistant_response = response.choices[0].message['content']
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.page = 'login'

    if st.session_state.logged_in:
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio(
            "Choose a page:",
            ["📹 Live Webcam Feed", "🖼 Upload Your Image & Ask a Question!", "🤖 GPT-4o - ChatBot"]
        )

        if selection == "📹 Live Webcam Feed":
            display_live_webcam_feed()
        elif selection == "🖼 Upload Your Image & Ask a Question!":
            display_upload_and_ask()
        elif selection == "🤖 GPT-4o - ChatBot":
            display_gpt_chatbot()

        if st.sidebar.button("🔓 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = 'login'

    else:
        login_page()

if __name__ == "__main__":
    main()