import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from gemini_quizzify3 import PDFProcessor
from gemini_quizzify4b import EmbeddingClient
from gemini_quizzify5 import ChromaCollectionCreator

from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = []  # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Context: {context}
        """
    
    def init_llm(self):
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.12,  # Increased for less deterministic questions
            max_output_tokens=500
        )

    def generate_question_with_vectorstore(self):
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")
        
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        retriever = self.vectorstore.as_retriever()
        prompt = PromptTemplate.from_template(self.system_template)
        
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        chain = setup_and_retrieval | prompt | self.llm 

        response = chain.invoke(self.topic)
        return response

    def generate_quiz(self) -> list:
        self.question_bank = []

        for i in range(self.num_questions):
            question_str = self.generate_question_with_vectorstore()
            
            try:
                question = json.loads(question_str)
            except json.JSONDecodeError:
                print(f"Failed to decode question JSON at index {i}. Response: {question_str}")
                continue  # Skip this iteration if JSON decoding fails
            
            if self.validate_question(question):
                print(f"Successfully generated unique question at index {i}")
                self.question_bank.append(question)
            else:
                print(f"Duplicate or invalid question detected at index {i}.")
        
        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        if "question" not in question:
            return False
        
        for q in self.question_bank:
            if q.get("question") == question.get("question"):
                return False
        
        return True

if __name__ == "__main__":
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-quizzify-428422",
        "location": "us-central1"
    }

    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = PDFProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)

        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()

                st.write(topic_input)

                generator = QuizGenerator(topic_input, num_questions, chroma_creator)
                question_bank = generator.generate_quiz()

                st.write("Generated questions:")
                st.write(question_bank)

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Questions: ")
            for question in question_bank:
                st.write(question)
