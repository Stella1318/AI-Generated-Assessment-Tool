import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from gemini_quizzify3 import PDFProcessor
from gemini_quizzify4b import EmbeddingClient
from gemini_quizzify5 import ChromaCollectionCreator

if __name__ == "__main__":
    st.header("Stella's Quizz Generator")

    # Configuration for EmbeddingClient
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-quizzify-428422",
        "location": "us-central1"
    }

    # Screen 1: Ingest Documents
    screen = st.empty()
    document = None  # Initialize the document variable
    with screen.container():
        st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
        
        # Initialize DocumentProcessor and Ingest Documents
        processor = PDFProcessor()
        processor.ingest_documents()
        
        # Initialize the EmbeddingClient with embed config
        try:
            embed_client = EmbeddingClient(**embed_config)
        except Exception as auth_error:
            st.error(f"Google Cloud authentication failed: {auth_error}", icon="🚨")
            st.stop()
        
        # Initialize the ChromaCollectionCreator
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
        
        with st.form("Load Data to Chroma"):
            topic_input = st.text_input("Enter the Quiz Topic")
            num_questions = st.slider("Number of Questions", min_value=1, max_value=20, value=5)
            submitted = st.form_submit_button("Generate a Quiz!")
            if submitted:
                if topic_input.strip() == "":
                    st.error("Quiz topic cannot be empty!", icon="🚨")
                else:
                    # Create a Chroma collection from the processed documents
                    chroma_creator.create_chroma_collection()
                    
                    # Query the Chroma collection for the topic input
                    document = chroma_creator.query_chroma_collection(topic_input)
                    
                    # Display the result
                    if document:
                        st.success("Successfully retrieved the top document for the quiz topic!")
                        st.write(document)
                    else:
                        st.error("Failed to retrieve any document for the quiz topic!", icon="🚨")

    # Screen 2: Display the result
    if document:
        screen.empty()  # Clear the initial screen
        with st.container():
            st.header("Query Chroma for Topic, top Document:")
            st.write(document)
