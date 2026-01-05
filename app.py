import streamlit as st
import cv2
import mediapipe as mp
import json
import time
from gtts import gTTS
import tempfile
from pathlib import Path
from PIL import Image
import os
from wordfreq import top_n_list

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Hand2Voice", layout="wide")

# ---------------- LOAD CSS ----------------
def load_css(name):
    p = Path(name)
    if p.exists():
        st.markdown(f"<style>{p.read_text()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ---------------- SESSION STATE ----------------
st.session_state.setdefault("sentence", "")
st.session_state.setdefault("current_word", "")
st.session_state.setdefault("last_char", "")
st.session_state.setdefault("last_time", 0.0)
st.session_state.setdefault("prediction_id", 0)

# ---------------- LOAD GESTURES ----------------
with open("gestures_rules.json") as f:
    GESTURES = json.load(f)["gestures"]

# ---------------- WORD PREDICTION ----------------
WORDS = top_n_list("en", 50000)

def predict(prefix, k=5):
    if len(prefix) < 2:
        return []
    prefix = prefix.lower()
    return [w for w in WORDS if w.startswith(prefix)][:k]

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---------------- FUNCTIONS ----------------
def finger_states(hand):
    lm = hand.landmark
    return [
        1 if lm[4].x < lm[3].x else 0,
        1 if lm[8].y < lm[6].y else 0,
        1 if lm[12].y < lm[10].y else 0,
        1 if lm[16].y < lm[14].y else 0,
        1 if lm[20].y < lm[18].y else 0
    ]

def detect(states):
    for name, rule in GESTURES.items():
        if rule["pattern"] == states:
            return name
    return None

def speak(text):
    tts = gTTS(text)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(f.name)
    return f.name

# ---------------- CALLBACK ----------------
def apply_prediction(word):
    text = st.session_state.sentence.rstrip()
    if " " in text:
        text = text.rsplit(" ", 1)[0]
    st.session_state.sentence = f"{text} {word.upper()} "
    st.session_state.current_word = ""

# ================== HEADER WITH LOGO ==================
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
""", unsafe_allow_html=True)

logo_path = "C:/Users/Admin/Desktop/project/Hand2Voice/logo.png"
if os.path.exists(logo_path):
    try:
        logo = Image.open(logo_path)
        col1, col2, col3 = st.columns([5, 7, 5])
        with col2:
            st.image(logo, use_container_width=True)
    except:
        pass

st.markdown("""
        <div class="hero-box">
            <div class="hero-subtitle">
                AI-Powered Sign Language to Speech Translation System
            </div>
            <div class="hero-description">
                <h4>Bridging Communication Gaps with Computer Vision & NLP</h4>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== FEATURES ==================
st.markdown("""
<div class="section-header">
    <h2 class="section-title">Core Features</h2>
</div>
""", unsafe_allow_html=True)

cols = st.columns(3)

with cols[0]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-wrapper">👐</div>
        <h3 class="feature-card-title">Gesture Recognition</h3>
        <p class="feature-card-desc">
            Real-time hand gesture detection using MediaPipe.
        </p>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-wrapper">🤖</div>
        <h3 class="feature-card-title">AI Word Prediction</h3>
        <p class="feature-card-desc">
            Smart NLP-based word suggestions while signing.
        </p>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-wrapper">🗣️</div>
        <h3 class="feature-card-title">Speech Output</h3>
        <p class="feature-card-desc">
            Converts recognized gestures into natural speech.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================== MAIN APP ==================
st.markdown("""
<div class="section-header">
    <h2 class="section-title">Start Translating</h2>
    <p class="section-subtitle">
        Show your hand gestures to the camera and watch them turn into speech
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    start = st.checkbox("🎥 Start Camera")
    cam = st.empty()

with col2:
    out = st.empty()
    preds = st.empty()
    c1, c2 = st.columns(2)
    speak_btn = c1.button("🔊 Speak")
    clear_button = c2.button("🧹 Clear")
    audio = st.empty()

# ---------------- CLEAR BUTTON LOGIC ----------------
if clear_button:
    st.session_state.sentence = ""
    st.session_state.current_word = ""
    st.session_state.last_char = ""
    st.success("✅ Text cleared successfully!")
    st.rerun()

# ---------------- CAMERA LOOP ----------------
if start:
    cap = cv2.VideoCapture(0)

    while start:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            states = finger_states(hand)
            char = detect(states)
            now = time.time()

            if char and (char != st.session_state.last_char or now - st.session_state.last_time > 1.2):
                if char == "SPACE":
                    st.session_state.sentence += " "
                    st.session_state.current_word = ""

                elif char in ["DEL", "BACKSPACE"]:
                    st.session_state.sentence = st.session_state.sentence[:-1]
                    st.session_state.current_word = st.session_state.current_word[:-1]

                elif len(char) == 1:
                    st.session_state.sentence += char
                    st.session_state.current_word += char

                st.session_state.last_char = char
                st.session_state.last_time = now

        cam.image(frame, channels="BGR", use_container_width=True)

        out.markdown(
            f"<div class='output-text'>{st.session_state.sentence or '🤚 Start signing...'}</div>",
            unsafe_allow_html=True
        )

        suggestions = predict(st.session_state.current_word)

        if suggestions:
            preds.markdown("**🔮 Word Suggestions:**")
            cols = preds.columns(len(suggestions))
            pid = st.session_state.prediction_id

            for i, w in enumerate(suggestions):
                cols[i].button(
                    w.upper(),
                    key=f"pred_{pid}_{i}",
                    on_click=apply_prediction,
                    args=(w,)
                )

            st.session_state.prediction_id += 1
        else:
            preds.empty()

        if speak_btn and st.session_state.sentence.strip():
            audio.audio(speak(st.session_state.sentence))
            speak_btn = False

    cap.release()

# ================== FOOTER ==================
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <div class="footer-logo">
            <div class="footer-logo-icon">🤟</div>
            <div class="footer-logo-text">Hand2Voice</div>
        </div>
        <div class="footer-developer">
            <h4>Developed By</h4>
            <p class="developer-name">Arshbir Singh</p>
            <p class="developer-role">Project Lead & Developer</p>
            <a href="https://imarshbir.github.io" class="portfolio-link" target="_blank">
                🌐 View Portfolio
            </a>
        </div>
        <div class="footer-tech">
            <h4>Technologies Used</h4>
            <div class="tech-tags">
                <span class="tech-tag">Streamlit</span>
                <span class="tech-tag">OpenCV</span>
                <span class="tech-tag">MediaPipe</span>
                <span class="tech-tag">gTTS</span>
                <span class="tech-tag">Python</span>
            </div>
        </div>
    </div>
    <div class="footer-bottom">
        <p>© 2024 Hand2Voice Project. All rights reserved.</p>
        <p>Bridging communication gaps through AI innovation</p>
    </div>
</div>
""", unsafe_allow_html=True)
