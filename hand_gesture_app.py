import time
import numpy as np
import cv2
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")
st.title("✋ Real-Time Hand Gesture Recognition (MediaPipe + Streamlit)")

# WebRTC needs a STUN server so the browser can connect from the cloud
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
min_det_conf = st.sidebar.slider("Min detection confidence", 0.3, 0.9, 0.6, 0.05)
min_trk_conf = st.sidebar.slider("Min tracking confidence", 0.3, 0.9, 0.6, 0.05)
flip_h = st.sidebar.checkbox("Flip video horizontally", value=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def finger_up(lm, tip_id, mcp_id, thresh=0.02):
    # y is top=0, bottom=1; finger "up" if tip above MCP
    return (lm[tip_id].y + 0.0) < (lm[mcp_id].y - thresh)

def thumb_up(lm, thresh=0.02):
    # thumb "up" if tip above IP (rough heuristic)
    return (lm[4].y + 0.0) < (lm[3].y - thresh)

def recognize_gesture(lm):
    fingers = {
        "index": finger_up(lm, 8, 5),
        "middle": finger_up(lm, 12, 9),
        "ring": finger_up(lm, 16, 13),
        "pinky": finger_up(lm, 20, 17),
    }
    th = thumb_up(lm)

    # ✌ Peace: index + middle up, ring+pinky down
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "✌ Peace"
    # 👍 Thumbs Up: thumb up, other four down
    if th and not any(fingers.values()):
        return "👍 Thumbs Up"
    # 👊 Fist: all down (thumb not up, all other down)
    if (not th) and not any(fingers.values()):
        return "👊 Fist"
    # ✋ Open Palm: all four up (thumb may vary)
    if all(fingers.values()):
        return "✋ Open Palm"
    return "🤚 Hand"

class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.gesture_text = "—"
        self.flip = flip_h
        # Create a Hands instance once (expensive)
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_trk_conf,
        )

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.flip:
            img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        label = "No Hand"
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                label = recognize_gesture(hand_lms.landmark)
        self.gesture_text = label

        cv2.putText(
            img,
            self.gesture_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start WebRTC component
st.markdown("Allow camera access when prompted. If it doesn’t connect, try Chrome.")
ctx = webrtc_streamer(
    key="hand-gesture",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=HandGestureTransformer,
    rtc_configuration=RTC_CONFIG,
)

# Live status panel
status = st.empty()
while ctx.state.playing:
    if ctx.video_transformer:
        status.markdown(f"### 👉 Gesture: **{ctx.video_transformer.gesture_text}**")
    time.sleep(0.1)
