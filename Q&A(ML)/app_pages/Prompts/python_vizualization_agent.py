import streamlit as st
import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from app_pages.backend import PythonChatbot, InputData
from app_pages.graph.tools import complete_python_task
import streamlit_ace as st_ace
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import moviepy.editor as mp
import speech_recognition as sr
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, VideoProcessorBase
import threading
import time
import base64
import logging
from groq import Groq as gt 
from deepgram import DeepgramClient
import queue
import numpy as np
import cv2
from audio_recorder_streamlit import audio_recorder
from typing import Union, Optional
import os
from pydub import AudioSegment
from groq import Groq as gt
from moviepy.editor import VideoFileClip
from openai import OpenAI
from deepgram import DeepgramClient, SpeakOptions
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import streamlit as st
import pygame
from deepgram import Deepgram
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import re
from functools import lru_cache
import logging
import base64
import wave 
import streamlit.components.v1 as components

# Ensure uploads directory exists
uploads_dir = "uploads"
audio_files_dir = "audio_files"
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(audio_files_dir, exist_ok=True)

groq_api_key=""
deepgram_api_key=""

groq_client = gt(api_key=groq_api_key)
deepgram_client = DeepgramClient(api_key=deepgram_api_key)

audio_files_dir = "audio_files"
os.makedirs(audio_files_dir, exist_ok=True)

# Initialize Pygame for audio playback
pygame.mixer.init()
if 'visualisation_chatbot' not in st.session_state:
    st.session_state.visualisation_chatbot = PythonChatbot()
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

if 'visualisation_chatbot' not in st.session_state:
    st.session_state.visualisation_chatbot = PythonChatbot()
if 'play_tts' not in st.session_state:
    st.session_state.play_tts = False

# Custom CSS for a professional UI
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 20px;
        font-family: 'Arial', sans-serif;
    }
    .header {
        font-size: 28px;
        font-weight: bold;
        color: #fff;
        text-align: center;
        margin-bottom: 20px;
        background-color: #007BFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input, .stChatInput>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .chat-container {
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
        height: 500px;
        overflow-y: auto;
    }
    .stChatMessage.user {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.assistant {
        background-color: #F0F0F0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .welcome-text {
        font-size: 20px;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .teacher-emoji {
        font-size: 70px;
        animation: pulse 2s infinite alternate;
    }
    .animated-text {
        font-size: 26px;
        font-weight: bold;
        color: #0056b3;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes pulse {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">Clarvis AI Learning Companion</div>', unsafe_allow_html=True)

# Welcome message
st.markdown("""
    <div class="welcome-text">
        I’m your AI Learning Companion, here to guide, challenge, and support you in mastering Data Structures, Algorithms, and AI/ML concepts!
    </div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3,tab5 = st.tabs(["Learn with Me", "Ecllipse","ClarvisUPskill", "Debugging Info"])

# Deepgram API setup (replace with your actual API key)
deepgram_api = "f81ccedd2038991069fb44343979cbb3a68169e8"
deepgram_client = DeepgramClient(api_key=deepgram_api)

# Constants
sample_rate = 16000
is_recording = False
audio_frames = []

# Cached recognizer
@lru_cache(maxsize=None)
def get_recognizer():
    return sr.Recognizer()

def record_audio(file_path, timeout=10, phrase_time_limit=None, retries=3, energy_threshold=2000,
                 pause_threshold=1, phrase_threshold=0.1, dynamic_energy_threshold=True,
                 calibration_duration=1):
    recognizer = get_recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold
    recognizer.phrase_threshold = phrase_threshold
    recognizer.dynamic_energy_threshold = dynamic_energy_threshold

    for attempt in range(retries):
        try:
            with sr.Microphone() as source:
                logging.info("Calibrating for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=calibration_duration)
                logging.info("Recording started")
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logging.info("Recording complete")
                with open(file_path, "wb") as f:
                    f.write(audio_data.get_wav_data())
                return
        except sr.WaitTimeoutError:
            logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            logging.error(f"Failed to record audio: {e}")
            if attempt == retries - 1:
                raise
    logging.error("Recording failed after all retries")

def play_audio(file_path):
    """Play audio automatically using HTML5 audio with autoplay."""
    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """Transcribes audio from a WAV file to text using Whisper."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language=language
            )
        logging.info(f"Transcription: {transcription.text}")
        return transcription.text
    except Exception as e:
        st.error(f"Failed to transcribe audio: {e}")
        return ""

def text_to_speech(text, output_file_path):
    """Convert text to speech using Deepgram."""
    options = {"model": "aura-luna-en", "encoding": "linear16", "container": "wav"}
    try:
        response = deepgram_client.speak.v("1").save(output_file_path, {"text": text}, options)
        logging.info(f"TTS response saved to {output_file_path}")
        return response
    except Exception as e:
        st.error(f"Failed to convert text to speech: {e}")
        return None

def get_first_three_sentences(text):
    """Extract the first three sentences from text."""
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:3])

def handle_voice_input_and_output():
    """Handle voice input, transcription, and TTS output."""
    global is_recording, audio_frames
    audio_frames = []
    is_recording = True
    st.write("Recording... Speak now.")

    record_audio("user_voice_input.wav")
    transcription = transcribe_audio("user_voice_input.wav")
    if transcription:
        st.session_state.visualisation_chatbot.user_sent_message(transcription)
        if st.session_state.visualisation_chatbot.chat_history:
            last_message = st.session_state.visualisation_chatbot.chat_history[-1]
            if isinstance(last_message, AIMessage):
                full_response = last_message.content
                tts_text = get_first_three_sentences(full_response)
                text_to_speech(tts_text, "nurse_response.wav")
                if os.path.exists("nurse_response.wav"):
                    st.session_state.play_tts = True
                else:
                    st.error("Failed to generate response audio.")
        st.rerun()

with tab1:
    st.subheader("Your Learning Journey")

    def on_submit_user_query():
        user_query = st.session_state.get('user_input', '')
        if user_query:
            st.session_state.visualisation_chatbot.user_sent_message(user_query)

    chat_container = st.container()

    with chat_container:
        if not st.session_state.visualisation_chatbot.chat_history:
            st.markdown("""
                <div class="chat-container">
                    <div class="teacher-emoji">📚</div>
                    <span class="animated-text">Ready to master Data Structures, Algorithms, and AI/ML? Let’s get started!</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            for msg_index, msg in enumerate(st.session_state.visualisation_chatbot.chat_history):
                if isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(msg.content)
                elif isinstance(msg, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(msg.content)

    # Play TTS audio automatically when triggered
    if st.session_state.get('play_tts', False):
        play_audio("nurse_response.wav")
        st.session_state.play_tts = False

    # Improved UI alignment with spacing and column ratios
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    input_col, mic_col = st.columns([5, 1])
    with input_col:
        st.chat_input(placeholder="Ask me anything about Data Structures, Algorithms, or AI/ML...",
                      on_submit=on_submit_user_query, key='user_input')
    with mic_col:
        if st.button("🎙️", key="mic_button1"):
            handle_voice_input_and_output()

with tab2:
    client = gt(api_key="")  # Replace with your OpenAI API key
    deepgram_client = DeepgramClient(api_key="")

    # Load the main prompt for question generation
    @st.cache_resource
    def initialize_agent():
        """
        Initializes and returns the AI agent.
        """
        return Agent(
            name="Video AI Summarizer",
            model=Groq(id="llama-3.2-90b-vision-preview"),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    def extract_audio_from_video(video_path: str) -> str:
        """
        Extracts audio from a video file and saves it as a WAV file.
        """
        try:
            video = VideoFileClip(video_path)
            audio_path = video_path.replace(".mp4", ".wav")
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            st.error(f"Failed to extract audio: {e}")
            return None
        finally:
            if 'video' in locals():
                video.close()

    def transcribe_audio(audio_path: str, language: str = "en") -> str:
        """
        Transcribes audio from a WAV file to text using Whisper.
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language=language
                )
            return transcription.text
        except Exception as e:
            st.error(f"Failed to transcribe audio: {e}")
            return ""

    def generate_questions(transcript: str, question_type: str, num_questions: int) -> str:
        """
        Generates questions from the transcript using the LLM.
        """
        try:
            # Define the prompt for question generation
            prompt = f"""
            You are an expert in generating educational content. Based on the following video transcript, create {num_questions} {question_type} questions:

            Transcript:
            {transcript}

            Instructions:
            1. Ensure the questions are clear, concise, and relevant to the content.
            2. For multiple-choice questions, provide 4 options and indicate the correct answer.
            3. For true/false questions, provide the correct answer.
            4. For short answer and essay questions, provide a sample answer or key points.

            Output the questions in the following format:
            - Question 1: [Question text]
            Options (if applicable): [Option A, Option B, Option C, Option D]
            Answer: [Correct answer or key points]
            """
            
            # Generate questions using the LLM
            response = client.chat.completions.create(
                model="gemma2-9b-it",  # Replace with your model
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Failed to generate questions: {e}")
            return ""
    multimodal_agent = initialize_agent()
    # Streamlit app
    st.subheader("Ecllispe")
    st.markdown("Upload a video, and we'll generate a question paper for you!")

    # File uploader
    video_file = st.file_uploader(
        "Upload a video file (MP4, MOV, AVI)",
        type=['mp4', 'mov', 'avi'],
        help="Supported formats: MP4, MOV, AVI."
    )

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)

        # Language selection
        language = st.radio(
            "Select language for transcription:",
            options=["English", "Hindi"],
            index=0
        )
        language_code = "en" if language == "English" else "hi"

        # Question type selection
        question_type = st.selectbox(
            "Select question type:",
            options=["Multiple Choice", "Short Answer", "True/False", "Essay"],
            help="Choose the type of questions to generate."
        )

        # Number of questions
        num_questions = st.slider(
            "Number of questions to generate:",
            min_value=1,
            max_value=20,
            value=5,
            help="Select the number of questions to generate from the video."
        )

        if st.button("📝 Generate Question Paper", key="generate_questions_button"):
            try:
                with st.spinner("Processing video and generating questions..."):
                    # Step 1: Extract audio from the video
                    audio_path = extract_audio_from_video(video_path)
                    if not audio_path:
                        raise Exception("Failed to extract audio from the video.")

                    # Step 2: Transcribe audio to text using Whisper
                    transcript = transcribe_audio(audio_path, language=language_code)
                    if not transcript:
                        raise Exception("Failed to transcribe audio to text.")

                    # Step 3: Generate questions using the LLM
                    questions = generate_questions(transcript, question_type, num_questions)
                    if not questions:
                        raise Exception("Failed to generate questions.")

                    # Step 4: Display the generated questions
                    st.subheader("Generated Question Paper")
                    st.markdown(questions, unsafe_allow_html=True)

            except Exception as error:
                st.error(f"An error occurred during question generation: {error}")
            finally:
                # Clean up temporary files
                time.sleep(1)
                if Path(video_path).exists():
                    try:
                        Path(video_path).unlink(missing_ok=True)
                    except PermissionError:
                        st.warning(f"Could not delete {video_path}. It may still be in use.")
                if 'audio_path' in locals() and Path(audio_path).exists():
                    try:
                        Path(audio_path).unlink(missing_ok=True)
                    except PermissionError:
                        st.warning(f"Could not delete {audio_path}. It may still be in use.")
    else:
        st.info("Please upload a video file to generate a question paper.")

    # Customize text area height
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# with tab3:

#     client = gt(api_key="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")
#     openai_client = OpenAI(api_key="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")  # Replace with your Groq API key
#     deepgram_api = "f81ccedd2038991069fb44343979cbb3a68169e8"
# # Load Whisper model (via OpenAI API)
#     deepgram_client = DeepgramClient(api_key=deepgram_api)

#     @st.cache_resource
#     def initialize_agent():
#         """
#         Initializes and returns the AI agent.
#         """
#         return Agent(
#             name="Video AI Summarizer",
#             model=Groq(id="llama-3.2-90b-vision-preview"),
#             tools=[DuckDuckGo()],
#             markdown=True,
#         )

#     def extract_audio_from_video(video_path: str) -> str:
#         """
#         Extracts audio from a video file and saves it as a WAV file.

#         Args:
#             video_path (str): Path to the video file.

#         Returns:
#             str: Path to the extracted audio file.
#         """
#         try:
#             # Load the video file
#             video = VideoFileClip(video_path)
#             # Extract audio
#             audio_path = video_path.replace(".mp4", ".wav")
#             video.audio.write_audiofile(audio_path)
#             return audio_path
#         except Exception as e:
#             st.error(f"Failed to extract audio: {e}")
#             return None
#         finally:
#             # Ensure the video file is closed
#             if 'video' in locals():
#                 video.close()

#     def transcribe_audio(audio_path: str, language: str = "en") -> str:
#         """
#         Transcribes audio from a WAV file to text using Whisper.

#         Args:
#             audio_path (str): Path to the audio file.
#             language (str): Language code for transcription (e.g., "en" for English, "hi" for Hindi).

#         Returns:
#             str: Transcribed text.
#         """
#         try:
#             # Transcribe audio using Whisper
#             with open(audio_path, "rb") as audio_file:
#                 transcription = client.audio.transcriptions.create(
#                     model="whisper-large-v3",
#                     file=audio_file,
#                     language=language
#                 )
#             return transcription.text
#         except Exception as e:
#             st.error(f"Failed to transcribe audio: {e}")
#             return ""

#     def text_to_speech(text: str, output_file_path: str):
#         """
#         Converts text to speech using Deepgram's TTS API.

#         Args:
#             text (str): The text to convert to speech.
#             output_file_path (str): Path to save the generated audio file.
#         """
#         options = SpeakOptions(
#             model="aura-asteria-en",  # You can change the model as needed
#             encoding="linear16",
#             container="wav"
#         )
#         SPEAK_OPTIONS = {"text": text}
#         response = deepgram_client.speak.v("1").save(output_file_path, SPEAK_OPTIONS, options)
#         return response

#     def search_keyword_in_transcript(transcript: str, keyword: str) -> list:
#         """
#         Searches for a keyword in the transcript and returns the timestamps where it appears.

#         Args:
#             transcript (str): The full transcript of the video.
#             keyword (str): The keyword to search for.

#         Returns:
#             list: A list of timestamps (in seconds) where the keyword appears.
#         """
#         timestamps = []
#         words = transcript.split()
#         for i, word in enumerate(words):
#             if keyword.lower() in word.lower():
#                 # Estimate timestamp based on word position (assuming 10 words per second)
#                 timestamp = i / 10
#                 timestamps.append(timestamp)
#         return timestamps

#     # Streamlit app
#     st.subheader("Lunar")

#     # Initialize the agent
#     multimodal_agent = initialize_agent()

#     # File uploader
#     video_file = st.file_uploader(
#         "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
#     )

#     if video_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
#             temp_video.write(video_file.read())
#             video_path = temp_video.name

#         st.video(video_path, format="video/mp4", start_time=0)

#         # Language selection
#         language = st.radio(
#             "Select language for transcription and summarization:",
#             options=["English", "Hindi"],
#             index=0
#         )
#         language_code = "en" if language == "English" else "hi"

#         user_query = st.text_area(
#             "What insights are you seeking from the video?",
#             placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
#             help="Provide specific questions or insights you want from the video."
#         )

#         # Keyword search input
#         keyword = st.text_input(
#             "Search for a keyword in the video transcript:",
#             placeholder="Enter a keyword to find its occurrences in the video."
#         )

#         if st.button("🔍 Analyze Video", key="analyze_video_button"):
#             if not user_query and not keyword:
#                 st.warning("Please enter a question, insight, or keyword to analyze the video.")
#             else:
#                 try:
#                     with st.spinner("Processing video and gathering insights..."):
#                         # Step 1: Extract audio from the video
#                         audio_path = extract_audio_from_video(video_path)
#                         if not audio_path:
#                             raise Exception("Failed to extract audio from the video.")

#                         # Step 2: Transcribe audio to text using Whisper
#                         transcript = transcribe_audio(audio_path, language=language_code)
#                         if not transcript:
#                             raise Exception("Failed to transcribe audio to text.")

#                         # Step 3: Perform keyword search if a keyword is provided
#                         if keyword:
#                             timestamps = search_keyword_in_transcript(transcript, keyword)
#                             if timestamps:
#                                 st.success(f"Keyword '{keyword}' found at the following timestamps: {timestamps}")
#                             else:
#                                 st.warning(f"Keyword '{keyword}' not found in the transcript.")

#                         # Step 4: Summarize the transcript using Groq if a user query is provided
#                         if user_query:
#                             if language == "Hindi":
#                                 prompt = f"""
#                                 निम्नलिखित वीडियो ट्रांसक्रिप्ट का विश्लेषण करें और हिंदी में सारांश प्रदान करें:
#                                 {transcript}

#                                 निम्नलिखित प्रश्न का उत्तर दें:
#                                 {user_query}

#                                 कृपया एक विस्तृत, उपयोगकर्ता-अनुकूल और कार्रवाई योग्य प्रतिक्रिया प्रदान करें।
#                                 """
#                             else:
#                                 prompt = f"""
#                                 Analyze the following video transcript and provide insights:
#                                 {transcript}

#                                 Respond to the following query:
#                                 {user_query}

#                                 Provide a detailed, user-friendly, and actionable response.
#                                 """

#                             # Use Groq to generate the response
#                             response = client.chat.completions.create(
#                                 model="llama-3.2-90b-vision-preview",
#                                 messages=[
#                                     {"role": "system", "content": "You are a helpful AI assistant."},
#                                     {"role": "user", "content": prompt}
#                                 ]
#                             )

#                             # Extract the response content
#                             result = response.choices[0].message.content

#                             # Step 5: Convert the result to speech using Deepgram TTS
#                             tts_output_path = "output_audio.wav"
#                             text_to_speech(result, tts_output_path)

#                             # Display the result
#                             st.subheader("Analysis Result")
#                             if result and result.strip():
#                                 st.markdown(result)
#                                 st.audio(tts_output_path, format="audio/wav")
#                             else:
#                                 st.warning("The analysis did not produce any meaningful insights. Please check the video and try again.")
#                 except Exception as error:
#                     st.error(f"An error occurred during analysis: {error}")
#                 finally:
#                     # Clean up temporary files
#                     time.sleep(1)  # Add a small delay to ensure file handles are released
#                     if video_path and os.path.exists(video_path):
#                         try:
#                             Path(video_path).unlink(missing_ok=True)
#                         except PermissionError:
#                             st.warning(f"Could not delete {video_path}. It may still be in use.")
#                     if audio_path and os.path.exists(audio_path):
#                         try:
#                             Path(audio_path).unlink(missing_ok=True)
#                         except PermissionError:
#                             st.warning(f"Could not delete {audio_path}. It may still be in use.")
#                     if 'tts_output_path' in locals() and os.path.exists(tts_output_path):
#                         try:
#                             Path(tts_output_path).unlink(missing_ok=True)
#                         except PermissionError:
#                             st.warning(f"Could not delete {tts_output_path}. It may still be in use.")
#     else:
#         st.info("Upload a video file to begin analysis.")

#     # Customize text area height
#     st.markdown(
#         """
#         <style>
#         .stTextArea textarea {
#             height: 100px;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# with tab4:  
    
#     # Load CLIP model and processor
#     @st.cache_resource
#     def load_model():
#         model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#         processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#         return model, processor

#     model, processor = load_model()

#     # Streamlit App
#     st.title("🖼️ Image Analysis")
#     st.write("""
#     Upload an image, enter text prompts, and let the **CLIP model** determine which description fits best!
#     """)

#     # Image Upload
#     uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         # Text Prompts
#         st.write("### 📝 Enter Text Prompts")
#         prompt1 = st.text_input("Prompt 1:", "a photo of a cat")
#         prompt2 = st.text_input("Prompt 2:", "a photo of a dog")
#         prompts = [prompt1, prompt2]

#         # Analyze Button
#         if st.button("🔍 Analyze Image"):
#             with st.spinner("Analyzing..."):
#                 # Process input
#                 inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
#                 outputs = model(**inputs)
#                 logits_per_image = outputs.logits_per_image
#                 probs = logits_per_image.softmax(dim=1)

#             # Display Results
#             st.write("### ✅ Results:")
#             for i, prompt in enumerate(prompts):
#                 st.write(f"- **{prompt}**: {probs[0][i].item() * 100:.2f}%")

#             # Highlight Best Match
#             best_match = prompts[torch.argmax(probs)]
#             st.success(f"🎯 **Best Match:** {best_match}")
#     else:
#         st.info("Please upload an image to begin.")

import threading

# Create a global lock and audio buffer for interview audio (do not store in session_state)
interview_lock = threading.Lock()
interview_audio_buffer = []   # Global list to hold audio chunks (bytes)

# Initialize session state for interview safely (only JSON-serializable types)
def initialize_interview_session():
    if "interview" not in st.session_state:
        st.session_state.interview = {
            "current_question": 0,
            "responses": [],
            "interview_active": False,
            "generated_questions": [],
            "interview_topics": [],
            "last_response": ""
        }

# Call the initialization before using st.session_state.interview (e.g., at the top of tab3)
initialize_interview_session()

with tab3:
    AUDIO_SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.5  # 500ms chunks for low latency
    MAX_QUESTIONS = 5

    # Initialize API clients
    @st.cache_resource
    def get_groq_client():
        return Groq(api_key="")

    groq_client = get_groq_client()

    # Initialize session state
    def initialize_interview_session():
        if "interview" not in st.session_state:
            st.session_state.interview = {
                "current_question": 0,
                "responses": [],
                "interview_active": False,
                "generated_questions": [],
                "interview_topics": [],
                "last_response": ""
            }

# Call the initialization before using st.session_state.interview (e.g., at the top of tab3)


    # Generate interview questions
    def generate_questions(topics: list, position: str = "entry-level") -> list:
        prompt = f"""
        Generate {MAX_QUESTIONS} technical interview questions for a {position} position 
        covering these topics: {', '.join(topics)}. Make questions progressively challenging.
        Format as a JSON list of strings.
        """
        
        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return list(json.loads(response.choices[0].message.content).values())[0]
        except Exception as e:
            st.error(f"Failed to generate questions: {str(e)}")
            return [
                "Explain the difference between supervised and unsupervised learning",
                "Describe how gradient descent works",
                "What is a neural network activation function?",
                "Explain the attention mechanism in transformers",
                "How would you handle imbalanced datasets?"
            ]

    # Audio processing class
    class RealTimeAudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.chunk_size = int(AUDIO_SAMPLE_RATE * CHUNK_DURATION)
            self.buffer = np.array([], dtype=np.int16)
            
        def recv(self, frame: np.ndarray) -> np.ndarray:
            audio_data = frame.to_ndarray()
            self.buffer = np.concatenate([self.buffer, audio_data])
            
            while len(self.buffer) >= self.chunk_size:
                chunk = self.buffer[:self.chunk_size]
                with interview_lock:
                    interview_audio_buffer.append(chunk.tobytes())
                self.buffer = self.buffer[self.chunk_size:]
            
            return frame

    # Global variables for audio processing
    interview_lock = threading.Lock()
    interview_audio_buffer = []

    def process_audio():
        while True:
            with interview_lock:
                if interview_audio_buffer:
                    audio_chunk = interview_audio_buffer.pop(0)
                else:
                    audio_chunk = None
            if audio_chunk:
                # Simulate transcription (replace with actual Deepgram API call)
                time.sleep(1)  # Simulate processing delay
                transcript = "Simulated transcription of audio chunk."
                st.session_state.interview["last_response"] += transcript + " "
            
            time.sleep(0.1)  # Avoid busy waiting

    # Generate feedback
    def generate_feedback(question: str, response: str) -> dict:
        prompt = f"""
        Analyze this technical interview response:
        Question: {question}
        Response: {response}
        
        Provide JSON with:
        - technical_score (1-10)
        - communication_score (1-5)
        - key_improvements
        - model_answer
        """
        
        try:
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Feedback generation error: {str(e)}")
            return {
                "technical_score": 0,
                "communication_score": 0,
                "key_improvements": "Error generating feedback",
                "model_answer": "Error generating model answer"
            }

    # Text-to-speech function
    def text_to_speech(text: str) -> Optional[bytes]:
        try:
            # Simulate TTS (replace with actual Deepgram TTS API call)
            return b"Simulated TTS audio data"
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
            return None

    # Streamlit UI
    st.title("AI Interview Coach 🤖💬")
    st.write("Technical interview practice")

    initialize_interview_session()

    # Interview configuration
    if not st.session_state.interview["interview_active"]:
        with st.form("config"):
            topics = st.multiselect(
                "Technical Topics:",
                ["ML Basics", "Data Structures", "Algorithms", "Neural Networks"],
                default=["ML Basics"]
            )
            level = st.selectbox("Level:", ["Entry", "Mid", "Senior"])
            
            if st.form_submit_button("Start Interview"):
                st.session_state.interview.update({
                    "current_question": 0,
                    "responses": [],
                    "interview_active": True,
                    "generated_questions": generate_questions(topics, level),
                    "interview_topics": topics,
                    "last_response": ""
                })
                st.rerun()

    # Main interview interface
    if st.session_state.interview["interview_active"]:
        webrtc_ctx = webrtc_streamer(
            key="interview",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=RealTimeAudioProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True
        )

        if st.session_state.interview["current_question"] < len(st.session_state.interview["generated_questions"]):
            current_q = st.session_state.interview["generated_questions"][st.session_state.interview["current_question"]]
            st.subheader(f"Question {st.session_state.interview['current_question'] + 1}")
            st.write(current_q)
            
            if st.button("Submit Response"):
                feedback = generate_feedback(current_q, st.session_state.interview["last_response"])
                st.session_state.interview["responses"].append({
                    "question": current_q,
                    "response": st.session_state.interview["last_response"],
                    "feedback": feedback
                })
                
                # Get AI response
                ai_response = text_to_speech(feedback["model_answer"])
                if ai_response:
                    st.audio(ai_response, format="audio/wav")
                
                st.session_state.interview.update({
                    "current_question": st.session_state.interview["current_question"] + 1,
                    "last_response": ""
                })
                
                if st.session_state.interview["current_question"] >= len(st.session_state.interview["generated_questions"]):
                    st.session_state.interview["interview_active"] = False
                st.rerun()
        else:
            st.success("Interview Complete!")
            st.session_state.interview["interview_active"] = False

    # Feedback display
    if st.session_state.interview["responses"]:
        st.subheader("Feedback Summary")
        for i, response in enumerate(st.session_state.interview["responses"]):
            with st.expander(f"Question {i+1}", expanded=i==0):
                st.write(f"**Question:** {response['question']}")
                st.write(f"**Your Response:** {response['response']}")
                st.json(response["feedback"])

    # Start processing thread
    if "audio_thread" not in st.session_state:
        audio_thread = threading.Thread(target=process_audio, daemon=True)
        audio_thread.start()
        st.session_state.audio_thread = audio_thread

with tab5:
    st.subheader("Debugging Info")
    if 'visualisation_chatbot' in st.session_state:
        for i, output in enumerate(st.session_state.visualisation_chatbot.intermediate_outputs):
            with st.expander(f"Step {i+1}"):
                if isinstance(output, dict):
                    if 'thought' in output:
                        st.markdown("### Thought Process")
                        st.markdown(output['thought'])
                    if 'code' in output:
                        st.markdown("### Code")
                        st.code(output['code'], language="python")
                    if 'output' in output:
                        st.markdown("### Output")
                        st.text(output['output'])
                else:
                    st.markdown("### Output")
                    st.text(output)
    else:
        st.info("No debug information available yet. Start a conversation to see intermediate outputs.")

