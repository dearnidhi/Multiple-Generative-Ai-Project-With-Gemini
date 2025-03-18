#from langchain.llms import OpenAI  # Commented out, previously used for OpenAI models

from dotenv import load_dotenv  # For loading environment variables from .env file

load_dotenv()  # Load .env file variables into the environment

import streamlit as st  # Streamlit is used for creating the web interface
import os  # Used for interacting with environment variables

import google.generativeai as genai  # Google's Gemini API for AI-generated responses

os.getenv("GOOGLE_API_KEY")  # Retrieve API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Set up Google Gemini API authentication
model = genai.GenerativeModel('gemini-1.5-pro-latest')  # Load Gemini Pro model
chat = model.start_chat(history=[])  # Create a chat session with empty history
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)  # Send message and get streaming response
    return response
st.set_page_config(page_title="Q&A Demo")  # Set webpage title in Streamlit

st.header("Gemini Application")  # Display a header
input = st.text_input("Input: ", key="input")  # Create an input text box for user questions
submit = st.button("Ask the question")  # Create a button to submit the question
if submit:  # If the user clicks the button
    response = get_gemini_response(input)  # Call the function to get AI response
    
    st.subheader("The Response is")  # Display a subheader for the response
    
    for chunk in response:  # Stream the response in chunks
        print(st.write(chunk.text))  # Display each chunk in Streamlit
        print("_" * 80)  # Print a separator line in console
    
    st.write(chat.history)  # Show the entire conversation history
