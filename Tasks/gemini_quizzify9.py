import streamlit as st
import os
import sys

# Add the necessary paths for imports
sys.path.append(os.path.abspath('../../'))

# Importing required classes from tasks modules
from gemini_quizzify3 import PDFProcessor
from gemini_quizzify4b import EmbeddingClient
from gemini_quizzify5 import ChromaCollectionCreator
from gemini_quizzify8 import QuizGenerator

class QuizManager:
    def __init__(self, questions: list):
        self.questions = questions
        self.total_questions = len(questions)

    def get_question_at_index(self, index: int):
        valid_index = index % self.total_questions
        return self.questions[valid_index]

    def next_question_index(self, direction=1):
        if "question_index" not in st.session_state:
            st.session_state["question_index"] = 0

        current_index = st.session_state["question_index"]
        new_index = (current_index + direction) % self.total_questions
        st.session_state["question_index"] = new_index

def main():
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-quizzify-428422",
        "location": "us-central1"
    }

    # Initialize session state
    if 'question_bank' not in st.session_state:
        st.session_state.question_bank = None
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
    if 'score' not in st.session_state:
        st.session_state.score = 0

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

                # Generate the quiz questions
                generator = QuizGenerator(topic_input, num_questions, chroma_creator)
                st.session_state.question_bank = generator.generate_quiz()
                st.session_state.question_index = 0
                st.session_state.score = 0

    if st.session_state.question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")

            # Initialize the QuizManager with the generated questions
            quiz_manager = QuizManager(st.session_state.question_bank)

            # Get the current question based on the session state index
            current_question = quiz_manager.get_question_at_index(st.session_state.question_index)
            choices = [f"{choice['key']}) {choice['value']}" for choice in current_question['choices']]

            with st.form("Multiple Choice Question"):
                st.write(current_question['question'])
                answer = st.radio('Choose the correct answer', choices, key="current_answer")
                submit_answer = st.form_submit_button("Submit Answer")

                if submit_answer:
                    correct_answer_key = current_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                        st.session_state.score += 1
                    else:
                        st.error("Incorrect!")

                col1, col2 = st.columns([1, 1])
                with col1:
                    prev_clicked = st.form_submit_button("Previous")
                with col2:
                    next_clicked = st.form_submit_button("Next")

                if prev_clicked:
                    quiz_manager.next_question_index(direction=-1)
                    st.experimental_rerun()
                if next_clicked:
                    quiz_manager.next_question_index(direction=1)
                    st.experimental_rerun()

            st.write(f"Score: {st.session_state.score}/{len(st.session_state.question_bank)}")

if __name__ == "__main__":
    main()
