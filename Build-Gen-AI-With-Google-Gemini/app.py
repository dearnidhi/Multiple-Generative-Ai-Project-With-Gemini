# Import necessary libraries
#from langchain.llms import OpenAI  # (Commented out) Used for OpenAI models but not needed here
from dotenv import load_dotenv  # Import dotenv to load environment variables from a .env file
# Load environment variables from .env file
load_dotenv()  
import streamlit as st  # Import Streamlit for building the web app
import os  # Import OS module to access environment variables
import textwrap  # Import textwrap for formatting text output
import google.generativeai as genai  # Import Google's Generative AI SDK (Gemini)

# Function to format text as Markdown (not used in Streamlit, but useful in Jupyter Notebooks)
def to_markdown(text):
    text = text.replace('•', '  *')  # Convert bullet points to Markdown format
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))  # Indent text as blockquote
# Load the Google API key from environment variables

os.getenv("GEMINI_API_KEY")  
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Configure Generative AI with API key


# Function to interact with the Gemini AI model
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')  # Initialize the Gemini Pro model
    response = model.generate_content(question)  # Generate a response based on input question
    return response.text  # Return the generated text response

# Initialize the Streamlit web application
st.set_page_config(page_title="Q&A Demo")  # Set the webpage title

st.header("Bot Application")  # Display the main heading

# Create an input text box where users can enter their questions
input = st.text_input("Input: ", key="input")

# Create a button labeled "Ask the question"
submit = st.button("Ask the question")

# If the button is clicked
if submit:
    response = get_gemini_response(input)  # Get AI-generated response for the input question
    st.subheader("The Response is")  # Display a subheading for the response
    st.write(response)  # Display the AI-generated response

