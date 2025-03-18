import streamlit as st  
from PyPDF2 import PdfReader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
import os  
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  
from langchain_community.vectorstores import FAISS  
from langchain.chains.question_answering import load_qa_chain  
from langchain.prompts import PromptTemplate  
from dotenv import load_dotenv  

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Google API key is missing! Set it in your .env file.")
else:
    from google.generativeai import configure
    configure(api_key=API_KEY)


def get_pdf_text(pdf_docs):
    """
    Extracts text from uploaded PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits extracted text into smaller chunks for efficient processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """
    Converts text chunks into embeddings and stores them using FAISS.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def load_vector_store():
    """
    Loads FAISS vector store if it exists.
    """
    if os.path.exists("faiss_index"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # return FAISS.load_local("faiss_index", embeddings)
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    return None


def get_conversational_chain():
    """
    Creates a conversational chain using Gemini.
    """
    prompt_template = PromptTemplate(
        template="""
        Answer the question as accurately as possible using the provided context. 
        If the answer is not available, say "Answer not found in the context."
        
        Context: {context}
        Question: {question}
        Answer:
        """,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt_template)


def handle_user_input(user_question):
    """
    Processes user input, retrieves relevant text from FAISS, and generates a response.
    """
    vector_store = load_vector_store()
    if not vector_store:
        st.error("No processed PDF found. Please upload and process a PDF first.")
        return
    
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.run({"input_documents": docs, "question": user_question})
    st.write("**Reply:**", response)


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Chat with PDFs - Gemini AI")
    st.header("Chat with PDFs using Gemini AI 🤖")

    user_question = st.text_input("Ask a question about your PDF files:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
