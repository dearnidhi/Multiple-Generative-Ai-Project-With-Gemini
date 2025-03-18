# Import necessary libraries
import streamlit as st  # For creating the web app
from dotenv import load_dotenv  # To load environment variables from .env file
import os  # For accessing environment variables
import google.generativeai as genai  # For using Google Gemini AI
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting YouTube video transcripts

# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# Configure Google Gemini API with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define a prompt for Gemini AI to summarize the transcript
prompt = """You are a YouTube video summarizer. You will take the transcript text
and summarize the entire video, providing the important summary in points
within 250 words. Please provide the summary of the text given here: """

# Function to extract transcript from a YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        # Check if the URL contains a video ID (basic validation)
        if "v=" not in youtube_video_url:
            raise ValueError("Invalid YouTube URL")  # Raise an error for invalid links

        # Extract the video ID from the URL (everything after "v=")
        video_id = youtube_video_url.split("v=")[1].split("&")[0]

        # Fetch the transcript using the YouTubeTranscriptApi library
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine the transcript segments into a single string
        transcript = ""
        for segment in transcript_text:
            transcript += " " + segment["text"]  # Append each text segment

        return transcript  # Return the full transcript text

    except Exception as e:
        st.error(f"Error: {str(e)}")  # Display error message in Streamlit
        return None  # Return None in case of failure

# Function to generate a summarized content using Google Gemini AI
def generate_gemini_content(transcript_text, prompt):
    try:
        # Create an instance of the latest Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        # Generate the summarized response using the transcript and prompt
        response = model.generate_content(prompt + transcript_text)

        return response.text  # Return the summarized text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")  # Display error message in Streamlit
        return None  # Return None in case of failure

# Streamlit UI setup
st.title("YouTube Transcript to Detailed Notes Converter")  # Web app title

# Input box for entering YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

# If a valid YouTube link is provided
if youtube_link:
    # Extract video ID to fetch the thumbnail image
    video_id = youtube_link.split("v=")[1].split("&")[0]

    # Display the YouTube video thumbnail in the app
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    # Button to trigger transcript extraction and summarization
    if st.button("Get Detailed Notes"):
        # Extract transcript from the provided video link
        transcript_text = extract_transcript_details(youtube_link)

        # If transcript is successfully extracted
        if transcript_text:
            # Generate the summary using Google Gemini AI
            summary = generate_gemini_content(transcript_text, prompt)

            # If summary is successfully generated, display it
            if summary:
                st.markdown("## Detailed Notes:")  # Section heading for the summary
                st.write(summary)  # Display the summarized notes
