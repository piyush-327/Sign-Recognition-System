import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import time
import base64
from gtts import gTTS


# 1. Load your Machine Learning models
@st.cache_resource
def load_models():
    with open('model1.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_numbers.pkl', 'rb') as f:
        model_numbers = pickle.load(f)
    return model, model_numbers


model, model_numbers = load_models()


# 2. Build the Video Processor
class SignLanguageProcessor:
    def __init__(self):
        # Initialize MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="dataset/hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO)

        self.landmarker = HandLandmarker.create_from_options(options)

        # State Variables (Notice the 'self.' prefix)
        self.current_word = ""
        self.consecutive_frames = 0
        self.previous_prediction = None
        self.current_mode = "Alphabet"
        self.frame_threshold = 20

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
            (5, 6), (6, 7), (7, 8), (5, 9), (0, 9),
            (9, 10), (10, 11), (11, 12), (9, 13), (0, 13),
            (13, 14), (14, 15), (15, 16), (13, 17),
            (17, 18), (18, 19), (19, 20), (0, 17)
        ]

    def recv(self, frame):
        # Convert internet packet to OpenCV image
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        current_time_ms = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, current_time_ms)

        if result.hand_landmarks:
            for hand_landmark in result.hand_landmarks:

                # Draw the dots and lines onto 'img'
                for landmark_id, landmark in enumerate(hand_landmark):
                    (pixel_x, pixel_y) = (int(landmark.x * w), int(landmark.y * h))
                    cv2.circle(img, (pixel_x, pixel_y), 6, (0, 255, 0), cv2.FILLED)

                for start_idx, end_idx in self.connections:
                    lm1 = hand_landmark[start_idx]
                    lm2 = hand_landmark[end_idx]
                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                    cv2.line(img, pt1, pt2, (255, 0, 0), 2)

                # Normalize Coordinates
                x_wrist = hand_landmark[0].x
                y_wrist = hand_landmark[0].y
                z_wrist = hand_landmark[0].z

                sub = []
                for landmark in hand_landmark:
                    sub.append(landmark.x - x_wrist)
                    sub.append(landmark.y - y_wrist)
                    sub.append(landmark.z - z_wrist)

                M = max(abs(val) for val in sub)
                if M == 0: M = 1.0

                normalized_coordinates = []
                for value in sub:
                    normalized_coordinates.append(value / M)

                # Predict based on mode
                current_pred = ""
                if self.current_mode == "Alphabet":
                    prediction_array = model.predict([normalized_coordinates])
                    current_pred = str(prediction_array[0])
                    cv2.putText(img, "Letter: " + current_pred, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                cv2.LINE_AA)
                elif self.current_mode == "number":
                    prediction_array = model_numbers.predict([normalized_coordinates])
                    current_pred = str(prediction_array[0])
                    cv2.putText(img, "Number: " + current_pred, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                cv2.LINE_AA)

                # Frame Counter Logic
                if current_pred == self.previous_prediction:
                    self.consecutive_frames += 1
                else:
                    self.consecutive_frames = 0
                    self.previous_prediction = current_pred

                if self.consecutive_frames == self.frame_threshold:
                    self.current_word += current_pred
                    self.consecutive_frames = 0

        # Draw current mode and word on screen
        cv2.putText(img, f"Mode: {self.current_mode}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(img, f"Word: {self.current_word}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# 3. Build the User Interface
st.title("Sign Language Web Translator")

# Start the webcam stream
ctx = webrtc_streamer(
    key="sign_translator",
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Web UI Buttons (Replaces keyboard keys)
if ctx.video_processor:
    st.write("### Controls")

    # Toggle Mode Button
    if st.button("Toggle Mode (Alphabet / Numbers)"):
        if ctx.video_processor.current_mode == "Alphabet":
            ctx.video_processor.current_mode = "number"
        else:
            ctx.video_processor.current_mode = "Alphabet"

    # Add Space Button
    if st.button("Add Space"):
        ctx.video_processor.current_word += " "

    # Backspace Button
    if st.button("Backspace"):
        ctx.video_processor.current_word = ctx.video_processor.current_word[:-1]

    # Clear Word Button
    if st.button("Clear Text"):
        ctx.video_processor.current_word = ""

    if st.button("Speak"):
        word_to_speak = ctx.video_processor.current_word.strip()

        if len(word_to_speak) > 0:
            tts = gTTS(text=word_to_speak, lang='en')
            tts.save("speech.mp3")

            with open("speech.mp3", "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()

            audio_html = f"""
                    <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
            st.markdown(audio_html, unsafe_allow_html=True)