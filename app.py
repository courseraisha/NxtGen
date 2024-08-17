import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Initialize Streamlit page
st.set_page_config(page_title="NxtGen ChatBot", page_icon="small_logo.jpeg", layout='wide')

st.image("big_logo.jpeg", width=600)
st.markdown("")
st.markdown("**NxtGen Chatbot** is here to assist you with your queries.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Intent classification based on user input
def classify_intent(user_input):
    user_input = user_input.lower().strip()
    
    if any(greeting in user_input for greeting in ["hi", "hello", "hey"]):
        return "greeting"
    elif "name" in user_input or "who are you" in user_input:
        return "introduction"
    elif "thank you" in user_input or "thanks" in user_input:
        return "gratitude"
    elif "bye" in user_input or "goodbye" in user_input:
        return "farewell"
    return "other"

# Map intents to responses
intent_responses = {
    "greeting": "Hello! How can I assist you today?",
    "introduction": "I'm the NxtGen AI-Powered Chatbot, here to assist you with your queries.",
    "gratitude": "You're welcome! Feel free to ask anything else.",
    "farewell": "Goodbye! Have a great day.",
    "other": None  # Continue with normal processing
}

# Default fallback response
fallback_response = "I'm not sure how to respond to that. Could you please rephrase or ask something else?"

# Input area for user's question
user_input = st.chat_input("Type your question here...")

# Process input when the button is clicked

if user_input:
    # Classify the intent
    intent = classify_intent(user_input)
    bot_response = intent_responses[intent]
    
    if bot_response is None:
        # Perform vector embedding if necessary
        if "vectors" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFLoader("Introduction to  NxtGen Innovation (1).pdf")  # Single PDF File
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

        # Create document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Show loading spinner
        with st.spinner('Processing your request...'):
            try:
                # Process the user input and generate a response
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_input})
                response_time = time.process_time() - start
                bot_response = response.get('answer', fallback_response)  # Use fallback if no answer

            except Exception as e:
                st.error(f"An error occurred: {e}")
                bot_response = "Sorry, I couldn't process your request. Please try again later."

    # Add user input and bot response to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

    # Display chat history using st.chat_message
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"You: {chat['user']}")
        with st.chat_message("assistant"):
            st.markdown(f"Bot: {chat['bot']}")
