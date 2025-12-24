import streamlit as st
import cv2
import numpy as np
import requests
from gtts import gTTS
import tempfile
import pandas

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Hand2Voice", layout="wide")
st.title("🤟 Hand2Voice")
st.write("Hand Gesture to Voice Conversion")

# ---------------- CONSTANTS ----------------
GESTURE_URL = "https://raw.githubusercontent.com/imarshbir/Hand2Voice/main/gestures/gesture_rules.json"

# ---------------- LOAD GESTURES ----------------
@st.cache_data
def load_gestures():
    return requests.get(GESTURE_URL).json()["gestures"]

# ---------------- LAZY MEDIAPIPE ----------------
@st.cache_resource
def load_mediapipe():
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

# ---------------- FINGER LOGIC ----------------
def get_finger_states(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [2, 6, 10, 14, 18]

    states = []

    states.append(
        1 if hand_landmarks.landmark[4].x >
             hand_landmarks.landmark[3].x else 0
    )

    for tip, base in zip(finger_tips[1:], finger_bases[1:]):
        states.append(
            1 if hand_landmarks.landmark[tip].y <
                 hand_landmarks.landmark[base].y else 0
        )

    return states

# ---------------- MATCH GESTURE ----------------
def match_gesture(states, rules):
    for name, info in rules.items():
        if states == info["pattern"]:
            return name
    return "Unknown Gesture"

# ---------------- RECOGNITION ----------------
def recognize_gesture(frame, hands, rules):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            states = get_finger_states(hand_landmarks)
            return match_gesture(states, rules)

    return "No Hand Detected"

# ---------------- TEXT TO SPEECH ----------------
def speak_text(text):
    tts = gTTS(text=text, lang="en")
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# ---------------- UI ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Camera")
    image = st.camera_input("Capture hand gesture")

with col2:
    st.subheader("📝 Output")

    if image:
        gestures = load_gestures()
        hands = load_mediapipe()

        img_bytes = image.getvalue()
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        gesture = recognize_gesture(frame, hands, gestures)

        st.success(f"🔊 {gesture}")

        if gesture not in ["Unknown Gesture", "No Hand Detected"]:
            audio = speak_text(gesture)
            st.audio(audio)
    else:
        st.info("Capture an image to start")
