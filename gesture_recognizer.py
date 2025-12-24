import cv2
from hand_tracker import hands
from gesture_logic import get_finger_states
from gesture_matcher import match_gesture
from gesture_loader import load_gestures

gesture_rules = load_gestures()

def recognize_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_states = get_finger_states(hand_landmarks)
            gesture = match_gesture(finger_states, gesture_rules)
            return gesture

    return None
