# Import necessary libraries
from dotenv import load_dotenv  # To load environment variables from a .env file
import streamlit as st  # For building the web interface
import os  # To interact with the operating system (e.g., reading environment variables)
import google.generativeai as genai  # Google's Generative AI library for interacting with Gemini models

# Load environment variables from the .env file
load_dotenv()

# Configure the Gemini API using the API key from the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to interact with the Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Load the Gemini Pro model
chat = model.start_chat(history=[])  # Start a chat session with an empty history

def get_gemini_response(question):
    """
    This function sends a question to the Gemini Pro model and streams the response.
    
    Parameters:
    - question: The input question provided by the user.
    
    Returns:
    - The streamed response from the Gemini Pro model.
    """
    response = chat.send_message(question, stream=True)  # Send the message and enable streaming
    return response  # Return the streamed response

## Initialize the Streamlit app
st.set_page_config(page_title="Q&A Demo")  # Set the page title for the Streamlit app
st.header("Gemini LLM Application")  # Display a header for the app

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # Create an empty list to store chat history

# Create a text input field for the user to provide their question
input = st.text_input("Input: ", key="input")

# Create a button for the user to submit their question
submit = st.button("Ask the question")

# If the "Ask the question" button is clicked and the input is not empty
if submit and input:
    response = get_gemini_response(input)  # Get the response from the Gemini Pro model
    
    # Add the user's question to the chat history
    st.session_state['chat_history'].append(("You", input))
    
    st.subheader("The Response is")
    
    # Stream the response and display it in the app
    for chunk in response:
        st.write(chunk.text)  # Write each chunk of the response
        # Add the bot's response to the chat history
        st.session_state['chat_history'].append(("Bot", chunk.text))

# Display the chat history
st.subheader("The Chat History is")

# Iterate through the chat history and display each message
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")  # Format and display the role (You/Bot) and the message