import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, VideoProcessorBase
import threading
import time
import base64
import logging
from groq import Groq
from deepgram import DeepgramClient
import queue
import numpy as np
import cv2
import mediapipe as mp
from audio_recorder_streamlit import audio_recorder
from typing import Union, Optional
import os

# Configuration
AUDIO_SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # Process audio in 500ms chunks for low latency
MAX_QUESTIONS = 5

# Initialize API clients
@st.cache_resource
def get_groq_client():
    return Groq(api_key="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")

@st.cache_resource
def get_deepgram_client():
    deepgram_api_key="f81ccedd2038991069fb44343979cbb3a68169e8"
    Deepgram = DeepgramClient(api_key=deepgram_api_key)
    return Deepgram

groq_client = get_groq_client()
deepgram = get_deepgram_client()

# Initialize MediaPipe models
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class InterviewSession:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.audio_buffer = queue.Queue()
        self.current_question = 0
        self.responses = []
        self.interview_active = False
        self.awaiting_response = False
        self.last_activity = time.time()
        self.posture_metrics = {"good": 0, "total": 0}
        self.eye_contact_metrics = {"good": 0, "total": 0}
        self.hand_movement = []
        self.generated_questions = []
        self.interview_topics = []

def generate_questions(topics: list, position: str = "entry-level") -> list:
    prompt = f"""
    Generate {MAX_QUESTIONS} technical interview questions for a {position} position 
    covering these topics: {', '.join(topics)}. Make questions progressively challenging.
    Format as a JSON list of strings.
    """
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    try:
        return list(response.choices[0].message.content.values())[0]
    except:
        return [
            "Explain the difference between supervised and unsupervised learning",
            "Describe how gradient descent works",
            "What is a neural network activation function?",
            "Explain the attention mechanism in transformers",
            "How would you handle imbalanced datasets?"
        ]

class RealTimeAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.chunk_size = int(AUDIO_SAMPLE_RATE * CHUNK_DURATION)
        self.buffer = np.array([], dtype=np.int16)
        
    def recv(self, frame: np.ndarray) -> np.ndarray:
        audio_data = frame.to_ndarray()
        self.buffer = np.concatenate([self.buffer, audio_data])
        
        while len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            st.session_state.interview.audio_buffer.put(chunk.tobytes())
            self.buffer = self.buffer[self.chunk_size:]
        
        return frame

class BodyLanguageAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        self.prev_hands = None
        
    def process_pose(self, image):
        results = self.pose.process(image)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder_avg = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            hip_avg = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + 
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            return shoulder_avg < hip_avg
        return False
    
    def process_eye_contact(self, image):
        results = self.face_mesh.process(image)
        return bool(results.multi_face_landmarks)
    
    def process_hands(self, image):
        results = self.hands.process(image)
        movement = 0
        
        if results.multi_hand_landmarks:
            current_landmarks = results.multi_hand_landmarks
            if self.prev_hands:
                for prev, curr in zip(self.prev_hands, current_landmarks):
                    movement += sum(
                        (prev_lm.x - curr_lm.x)**2 + (prev_lm.y - curr_lm.y)**2
                        for prev_lm, curr_lm in zip(prev.landmark, curr.landmark)
                    )
            self.prev_hands = current_landmarks
            
        return movement
    
    def recv(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
        
        # Posture analysis
        if self.process_pose(img):
            st.session_state.interview.posture_metrics["good"] += 1
        st.session_state.interview.posture_metrics["total"] += 1
        
        # Eye contact detection
        if self.process_eye_contact(img):
            st.session_state.interview.eye_contact_metrics["good"] += 1
        st.session_state.interview.eye_contact_metrics["total"] += 1
        
        # Hand movement analysis
        st.session_state.interview.hand_movement.append(self.process_hands(img))
        
        return frame

def process_audio():
    while True:
        if not st.session_state.interview.audio_buffer.empty():
            audio_chunk = st.session_state.interview.audio_buffer.get()
            
            # Real-time transcription
            try:
                source = {"buffer": audio_chunk, "mimetype": 'audio/wav'}
                transcript = deepgram.transcription.sync_prerecorded(
                    source, {'punctuate': True, 'model': 'nova-2'}
                )
                text = transcript['results']['channels'][0]['alternatives'][0]['transcript']
                
                if text:
                    st.session_state.interview.last_response += text + " "
                    st.session_state.interview.last_activity = time.time()
                    
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")

def generate_feedback():
    prompt = f"""
    Analyze this interview response (Technical accuracy 1-10, Communication 1-5):
    Question: {st.session_state.interview.current_question}
    Response: {st.session_state.interview.last_response}
    
    Provide:
    1. Technical accuracy score with justification
    2. Communication score (clarity/structure)
    3. Key improvement areas
    4. Model answer excerpt
    5. Body language analysis (posture: {posture_score}%, eye contact: {eye_contact_score}%)
    
    Format as JSON with keys: technical_score, communication_score, improvements, model_answer, body_language
    """
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def text_to_speech(text: str) -> Optional[bytes]:
    try:
        response = deepgram.speak.v("1").stream(
            {"text": text},
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )
        return response.stream.read()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None


st.write("Real-time technical interview practice with AI feedback")

if "interview" not in st.session_state:
    st.session_state.interview = InterviewSession()

# Interview configuration
if not st.session_state.interview.interview_active:
    with st.form("config"):
        st.subheader("Configure Your Interview")
        topics = st.multiselect(
            "Select Technical Topics:",
            ["Machine Learning", "Data Structures", "Algorithms", "Deep Learning", "System Design"],
            default=["Machine Learning"]
        )
        level = st.selectbox("Experience Level:", ["Entry", "Mid", "Senior"])
        
        if st.form_submit_button("Start Interview"):
            st.session_state.interview.reset()
            st.session_state.interview.interview_topics = topics
            st.session_state.interview.generated_questions = generate_questions(topics, level)
            st.session_state.interview.interview_active = True
            st.session_state.interview.current_question = 0
            st.rerun()

# Main interview interface
if st.session_state.interview.interview_active:
    col1, col2, col3 = st.columns(3)
    with col1:
        posture_score = st.session_state.interview.posture_metrics["good"] / st.session_state.interview.posture_metrics["total"] * 100 if st.session_state.interview.posture_metrics["total"] > 0 else 0
        st.metric("Posture", f"{posture_score:.1f}%")
    with col2:
        eye_contact_score = st.session_state.interview.eye_contact_metrics["good"] / st.session_state.interview.eye_contact_metrics["total"] * 100 if st.session_state.interview.eye_contact_metrics["total"] > 0 else 0
        st.metric("Eye Contact", f"{eye_contact_score:.1f}%")
    with col3:
        hand_score = np.mean(st.session_state.interview.hand_movement) if st.session_state.interview.hand_movement else 0
        st.metric("Hand Movement", f"{hand_score:.2f}")

    # WebRTC components
    webrtc_ctx = webrtc_streamer(
        key="interview",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=RealTimeAudioProcessor,
        video_processor_factory=BodyLanguageAnalyzer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "audio": {"sampleRate": AUDIO_SAMPLE_RATE},
            "video": {"width": 640}
        },
        async_processing=True
    )

    # Question navigation
    if st.session_state.interview.current_question < len(st.session_state.interview.generated_questions):
        current_q = st.session_state.interview.generated_questions[st.session_state.interview.current_question]
        st.subheader(f"Question {st.session_state.interview.current_question + 1}/{MAX_QUESTIONS}")
        st.write(current_q)
        
        if st.button("Next Question"):
            feedback = generate_feedback()
            st.session_state.interview.responses.append(feedback)
            st.session_state.interview.current_question += 1
            st.rerun()
    else:
        st.success("Interview completed! Review your feedback below.")
        st.session_state.interview.interview_active = False

# Feedback review
if not st.session_state.interview.interview_active and st.session_state.interview.responses:
    st.subheader("Interview Report")
    for i, response in enumerate(st.session_state.interview.responses):
        with st.expander(f"Question {i+1} Feedback", expanded=i==0):
            st.json(response)
            
    if st.button("Start New Interview"):
        st.session_state.interview.reset()
        st.rerun()

# Start background processing
if "audio_thread" not in st.session_state:
    audio_thread = threading.Thread(target=process_audio, daemon=True)
    audio_thread.start()
    st.session_state.audio_thread = audio_thread