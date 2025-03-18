# Import necessary libraries
from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import io

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to process images for Gemini API
def convert_image_to_bytes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    return img_bytes.getvalue()

# Function to get Gemini response
def get_gemini_response(input_text, image):
    model = genai.GenerativeModel('gemini-1.5-pro')  # Using Gemini 1.5 Pro

    request_payload = []

    # Ensure there's always a text prompt
    if input_text.strip() == "":
        input_text = "Describe this image in detail."  # Default prompt
    
    request_payload.append(input_text)  # Add text input

    if image:
        img_bytes = convert_image_to_bytes(image)
        image_data = {
            "mime_type": "image/png",
            "data": img_bytes
        }
        request_payload.append(image_data)  # Add image data

    try:
        response = model.generate_content(request_payload)
        return response.text if hasattr(response, "text") else "No valid response received."
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.set_page_config(page_title="Gemini Image Demo")
st.header("Gemini Application")

# Input prompt field
input_text = st.text_input("Input Prompt: ", key="input")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)

# Submit button
submit = st.button("Tell me about the image")

if submit:
    response = get_gemini_response(input_text, image)
    st.subheader("The Response is")
    st.write(response)
