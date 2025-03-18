# Import necessary libraries
from dotenv import load_dotenv  # To load environment variables from a .env file
import streamlit as st  # For building the web interface
import os  # To interact with the operating system (e.g., reading environment variables)
import sqlite3  # To interact with SQLite databases
import google.generativeai as genai  # Google's Generative AI library for interacting with Gemini models

# Load environment variables from the .env file
load_dotenv()

# Configure the Gemini API using the API key from the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to interact with the Gemini Pro model and generate SQL queries
def get_gemini_response(question, prompt):
    """
    This function sends a question and a predefined prompt to the Gemini Pro model
    and returns the generated SQL query as a response.
    
    Parameters:
    - question: The English question provided by the user.
    - prompt: A predefined instruction or context for the model.
    
    Returns:
    - The generated SQL query as text.
    """
    model = genai.GenerativeModel('gemini-1.5-pro')  # Load the Gemini Pro model
    response = model.generate_content([prompt[0], question])  # Generate content based on the inputs
    return response.text  # Return the generated SQL query

## Function to execute a SQL query on the database and retrieve results
def read_sql_query(sql, db):
    """
    This function executes a SQL query on the specified SQLite database and retrieves the results.
    
    Parameters:
    - sql: The SQL query to execute.
    - db: The name of the SQLite database file.
    
    Returns:
    - The rows fetched from the database as a list of tuples.
    """
    conn = sqlite3.connect(db)  # Connect to the SQLite database
    cur = conn.cursor()  # Create a cursor object to execute queries
    cur.execute(sql)  # Execute the SQL query
    rows = cur.fetchall()  # Fetch all rows returned by the query
    conn.commit()  # Commit any changes made to the database
    conn.close()  # Close the database connection
    for row in rows:  # Print the rows to the console (for debugging purposes)
        print(row)
    return rows  # Return the fetched rows

## Define the prompt for the Gemini model
prompt = [
    """
    You are an expert in converting English questions to SQL queries!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION.
    
    For example:
    Example 1 - How many entries of records are present?
    The SQL command will be something like this: SELECT COUNT(*) FROM STUDENT;
    
    Example 2 - Tell me all the students studying in Data Science class?
    The SQL command will be something like this: SELECT * FROM STUDENT WHERE CLASS="Data Science";
    
    Note: The SQL code should not have ``` in the beginning or end, and the word "sql" should not appear in the output.
    """
]

## Initialize the Streamlit app
st.set_page_config(page_title="I can Retrieve Any SQL query")  # Set the page title for the Streamlit app
st.header("Gemini App To Retrieve SQL Data")  # Display a header for the app

# Create a text input field for the user to provide their question
question = st.text_input("Input: ", key="input")

# Create a button for the user to submit their question
submit = st.button("Ask the question")

# If the "Ask the question" button is clicked
if submit:
    # Generate the SQL query using the Gemini Pro model
    response = get_gemini_response(question, prompt)
    print(response)  # Print the generated SQL query to the console (for debugging purposes)
    
    # Execute the SQL query on the database and retrieve the results
    response = read_sql_query(response, "student.db")
    
    # Display the results in the Streamlit app
    st.subheader("The Response is")
    for row in response:  # Iterate through the rows of the result
        print(row)  # Print the row to the console (for debugging purposes)
        st.header(row)  # Display the row as a header in the Streamlit app
