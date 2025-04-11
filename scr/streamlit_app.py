import streamlit as st
import requests
from PIL import Image
import io
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from passlib.hash import bcrypt

# Streamlit Frontend Setup

def login_page(verify_password=None):
    st.title("🔐 Login to Your Account")

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login", help="Click to login after entering your credentials"):
        response = requests.post(
            'http://127.0.0.1:8000/api/v1/get_user',
            json={"username": username}
        ).json()
        user = response.get('user')
        if user and verify_password(user[2], password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.show_login = False
            st.session_state.page = 'main'
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
                    response = requests.post(
                        'http://127.0.0.1:8000/api/v1/create_account',
                        json={"username": new_username, "password": hashed_password}
                    ).json()
                    if response.get('status') == 'success':
                        st.success("🎉 Account created successfully! You can now log in.")
                    else:
                        st.error(f"⚠ Error occurred: {response.get('error')}")
                except Exception as e:
                    st.error(f"⚠ Error occurred: {e}")
            else:
                st.error("⚠ Please provide both username and password.")

def display_live_webcam_feed():
    st.subheader("🎥 Live Webcam Feed")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
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

        # Send the user prompt to FastAPI endpoint
        response = requests.post(
            'http://127.0.0.1:8000/api/v1/chatgpt',
            json={"prompt": user_prompt}
        ).json()

        if "error" in response:
            assistant_response = f"⚠ Error occurred: {response['error']}"
        else:
            assistant_response = response['response']

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "show_login" not in st.session_state:
        st.session_state.show_login = True

    if st.session_state.show_login:
        login_page()
    elif st.session_state.logged_in:
        st.sidebar.title(f"Welcome, {st.session_state.username}! 👋")
        st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False, "show_login": True}))
        menu_option = st.sidebar.radio("📑 Menu", ["Live Webcam", "Upload & Ask", "GPT-4o ChatBot"])

        if menu_option == "Live Webcam":
            display_live_webcam_feed()
        elif menu_option == "Upload & Ask":
            display_upload_and_ask()
        elif menu_option == "GPT-4o ChatBot":
            display_gpt_chatbot()

if __name__ == "__main__":
    main()
