from pytube import YouTube
import moviepy
from moviepy.editor import *
import speech_recognition as sr
from transformers import pipeline
import re
from langchain_core.tools import tool
from langchain.tools import tool
import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from groq import Groq as gt

from pytube import YouTube
import moviepy.editor as mp
import speech_recognition as sr
from transformers import pipeline
import re
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import openai
from moviepy.editor import VideoFileClip
import yt_dlp
import os 
import openai
 # Replace with actual path


import datetime

# Initialize Groq client
client1 = gt(api_key="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")

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL: No video ID found")

# Function to extract metadata
def extract_metadata(url):
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(r.text, features="html.parser")
        
        title = soup.find("title").text if soup.find("title") else "No Title Found"
        channel = soup.find("link", itemprop="name")['content'] if soup.find("link", itemprop="name") else "No Channel Found"
        
        return title, channel
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch metadata: {e}")

# Function to download the thumbnail
def download_thumbnail(video_id):
    image_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    try:
        img_data = requests.get(image_url).content
        with open('thumbnail.jpg', 'wb') as handler:
            handler.write(img_data)
    except requests.RequestException as e:
        raise ValueError(f"Failed to download thumbnail: {e}")

# Function to download audio using yt_dlp
def download_audio(youtube_url, output_path=None):
    # Generate a unique filename if none is provided
    if not output_path:
        video_id = extract_video_id(youtube_url)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"youtube_audio_{video_id}_{timestamp}.mp3"

    # Specify the custom path to the directory containing ffmpeg and ffprobe
    ffmpeg_dir = r"C:\ffmpeg-2025-01-30-git-1911a6ec26-full_build\bin"
    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe.exe")

    # Check if ffmpeg and ffprobe exist
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg not found at: {ffmpeg_path}")
    if not os.path.exists(ffprobe_path):
        raise FileNotFoundError(f"ffprobe not found at: {ffprobe_path}")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": output_path,
        "ffmpeg_location": ffmpeg_dir,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Return the correct filename after extraction
        return f"{output_path}.mp3"  # yt-dlp appends .mp3 to the filename
    except yt_dlp.utils.DownloadError as e:
        print(f"Failed to download audio: {e}")
        return None

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path: str, language: str = "en") -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client1.audio.transcriptions.create(
                model="whisper-large-v3",  # OpenAI's Whisper model (supports Large V3)
                file=audio_file,
                language=language
            )
        return transcription.text
    except Exception as e:
        print(f"Failed to transcribe audio: {e}")
        return ""

# Function to summarize the transcript using OpenAI GPT
def summarize_transcript(transcript):
    if not transcript:
        return "No transcript available for summarization."
    
    try:
        response = client1.chat.completions.create(
            model="gemma2-9b-it",  # Use an appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Summarize the following transcript:\n{transcript}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to summarize transcript: {e}")
        return ""

# Function to extract audio from a video file
def extract_audio_from_video(video_path):
    try:
        video = VideoFileClip(video_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f"extracted_audio_{timestamp}.wav"
        video.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        print(f"Failed to extract audio from video: {e}")
        return None

# Function to find keyword timestamp
def find_keyword_timestamp(transcript, keyword):
    if not transcript:
        return None
    
    for entry in transcript:
        if keyword.lower() in entry['text'].lower():
            return entry['start']
    return None

# Function to extract URL from text
def extract_url_from_text(text):
    url_pattern = re.compile(r'https?://(?:www\.)?youtube\.com/watch\?v=[a-zA-Z0-9_-]+')
    match = url_pattern.search(text)
    return match.group(0) if match else None


def utube_summarize_video(url: str = None, video_path: str = None, keyword: str = None) -> tuple:
    """
    Summarizes a YouTube video or a manually uploaded video by extracting its transcript and generating a summary.
    If a keyword is provided, it returns the timestamp of the keyword in the video.

    Args:
        url (str): The URL of the YouTube video.
        video_path (str): The path to the uploaded video file.
        keyword (str): A keyword to search for in the video transcript.

    Returns:
        tuple: A tuple containing the results and a dictionary with intermediate outputs.
              Format: (results, {"intermediate_outputs": [{"output": results}]})
    """
    try:
        # If URL is not provided, try to extract it from video_path or keyword
        if not url:
            if video_path:
                url = extract_url_from_text(video_path)
            elif keyword:
                url = extract_url_from_text(keyword)
            if not url:
                return "No YouTube URL provided or found in the input.", {
                    "intermediate_outputs": [{"output": "No URL found"}]
                }

        if url:
            # Handle YouTube video
            video_id = extract_video_id(url)
            if not video_id:
                return "Invalid YouTube URL. Please provide a valid URL in the format 'https://www.youtube.com/watch?v=VIDEO_ID'.", {
                    "intermediate_outputs": [{"output": "Invalid YouTube URL"}]
                }

            # Extract metadata
            try:
                title, channel = extract_metadata(url)
                print(f"Title: {title}")
                print(f"Channel: {channel}")
            except Exception as e:
                return f"Failed to fetch video metadata. The video might be private or unavailable. Error: {e}", {
                    "intermediate_outputs": [{"output": f"Failed to fetch metadata: {e}"}]
                }

            # Download thumbnail
            try:
                download_thumbnail(video_id)
                print("Thumbnail downloaded as 'thumbnail.jpg'")
            except Exception as e:
                print(f"Failed to download thumbnail. Error: {e}")

            # Get transcript
            audio_path = download_audio(url)  # Unique filename will be generated automatically
            print(f"Audio downloaded to: {audio_path}")

            # Transcribe audio to text
            transcript_text = transcribe_audio(audio_path)
            if not transcript_text:
                return "Failed to transcribe the audio. Please ensure the audio is clear and in a supported language.", {
                    "intermediate_outputs": [{"output": "Failed to transcribe audio"}]
                }

            # Summarize transcript
            try:
                summary = summarize_transcript(transcript_text)
                if not summary:
                    return "Failed to generate a summary. The transcript might be too short or unclear.", {
                        "intermediate_outputs": [{"output": "Failed to generate summary"}]
                    }

                # Prepare results
                results = summary  # Directly return the summary

                if keyword:
                    # For YouTube videos, keyword search is supported
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    timestamp = find_keyword_timestamp(transcript, keyword) if transcript else None
                    if timestamp:
                        results += f"\nKeyword '{keyword}' found at {timestamp} seconds."
                    else:
                        results += f"\nKeyword '{keyword}' not found in the transcript."

                return results, {"intermediate_outputs": [{"output": results}]}
            except Exception as e:
                return f"Failed to summarize the transcript. Error: {e}", {
                    "intermediate_outputs": [{"output": f"Failed to summarize transcript: {e}"}]
                }

        elif video_path:
            # Handle uploaded video
            try:
                # Extract audio from video
                audio_path = extract_audio_from_video(video_path)
                print(f"Audio extracted to: {audio_path}")

                # Transcribe audio to text
                transcript = transcribe_audio(audio_path)
                if not transcript:
                    return "Failed to transcribe the audio. Please ensure the audio is clear and in a supported language.", {
                        "intermediate_outputs": [{"output": "Failed to transcribe audio"}]
                    }

                # Summarize transcript
                try:
                    summary = summarize_transcript(transcript)
                    if not summary:
                        return "Failed to generate a summary. The transcript might be too short or unclear.", {
                            "intermediate_outputs": [{"output": "Failed to generate summary"}]
                        }

                    # Prepare results
                    results = summary  # Directly return the summary

                    if keyword:
                        # For uploaded videos, keyword search is not supported directly
                        results += "\nKeyword search is not supported for uploaded videos."

                    return results, {"intermediate_outputs": [{"output": results}]}
                except Exception as e:
                    return f"Failed to summarize the transcript. Error: {e}", {
                        "intermediate_outputs": [{"output": f"Failed to summarize transcript: {e}"}]
                    }
            except Exception as e:
                return f"Failed to process the uploaded video. Error: {e}", {
                    "intermediate_outputs": [{"output": f"Failed to process video: {e}"}]
                }

        else:
            return "Please provide either a YouTube URL or a video file path.", {
                "intermediate_outputs": [{"output": "No input provided"}]
            }

    except Exception as e:
        return str(e), {"intermediate_outputs": [{"output": str(e)}]}
