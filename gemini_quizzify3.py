import streamlit as st
from langchain.document_loaders import PyPDFLoader

# Step 1: Upload the PDF file
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")
        # Step 2: Process the file using PyPDFLoader
        pdf_loader = PyPDFLoader(uploaded_file)
        extracted_pages = pdf_loader.load_and_split()

        # Add extracted pages to the class variable
        PDFProcessor.add_pages(extracted_pages)

        st.success(f"Successfully processed {uploaded_file.name}")

    # Display all extracted pages
    st.write("Extracted Pages:")
    for i, page in enumerate(PDFProcessor.pages):
        st.write(f"Page {i+1}:")
        st.write(page)

class PDFProcessor:
    pages = []

    @classmethod
    def add_pages(cls, new_pages):
        cls.pages.extend(new_pages)
