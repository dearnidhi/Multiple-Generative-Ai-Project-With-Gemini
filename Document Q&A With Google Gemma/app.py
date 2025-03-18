import streamlit as st  # Import Streamlit for creating a web-based UI
import os  # Import OS module to manage environment variables and file paths
from langchain_groq import ChatGroq  # Import ChatGroq for using Groq-based LLMs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import text splitter for chunking documents
from langchain.chains.combine_documents import create_stuff_documents_chain  # Import document processing chain
from langchain_core.prompts import ChatPromptTemplate  # Import prompt template for LLM
from langchain.chains import create_retrieval_chain  # Import retrieval chain for search and response
from langchain_community.vectorstores import FAISS  # Import FAISS for storing and retrieving vector embeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Import loader to process PDF files
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Import Google embeddings for vectorization
from dotenv import load_dotenv  # Import dotenv to load API keys from environment variables

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')  # Groq API key for ChatGroq model
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Google API key for embeddings

# Set the title of the Streamlit app
st.title("Gemma Model Document Q&A")

# Initialize the LLM (Language Model) using ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template for the language model
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
"""
)

# Function to create vector embeddings for document retrieval
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load PDF documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Load PDFs from 'us_census' folder
        st.session_state.docs = st.session_state.loader.load()  # Load PDF content

        # Check if documents are loaded
        if not st.session_state.docs:
            st.error("No documents were loaded. Please check the file path and ensure PDFs exist in './us_census'.")
            return

        # Split documents into smaller chunks for better processing
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Ensure that documents have valid content before vectorization
        if not st.session_state.final_documents:
            st.error("No document chunks were created. Please check the text extraction process.")
            return

        # Convert the text chunks into vector embeddings using FAISS
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


# Input box to take user queries
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to trigger document embedding process
if st.button("Documents Embedding"):
    vector_embedding()  # Call the function to process and store document embeddings
    st.write("Vector Store DB Is Ready")  # Notify the user that the embedding process is complete

import time  # Import time module to measure response time

# If a user has entered a query, process the request
if prompt1:
    # Create a document processing chain using LLM and the prompt template
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieve relevant documents from the vector store
    retriever = st.session_state.vectors.as_retriever()
    
    # Create a retrieval chain that integrates the retriever with document processing
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Measure the processing time of the query
    start = time.process_time()
    
    # Invoke the retrieval chain with the user's question
    response = retrieval_chain.invoke({'input': prompt1})
    
    # Print the response time in the terminal (for debugging)
    print("Response time:", time.process_time() - start)
    
    # Display the model's answer in Streamlit
    st.write(response['answer'])

    # Display similar documents retrieved from the vector store for transparency
    with st.expander("Document Similarity Search"):  # Use an expander to keep UI clean
        for i, doc in enumerate(response["context"]):  # Loop through retrieved documents
            st.write(doc.page_content)  # Display document content
            st.write("--------------------------------")  # Add a separator for clarity
