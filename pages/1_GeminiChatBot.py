import os

import streamlit as st
from langchain.llms import OpenAI

from src.model.llms import load_llm


def generate_response(input_text):
    """
    Generate a response from the language model based on the input text.

    Args:
        input_text (str): The input text for which the response is to be generated.

    Returns:
        response: The response generated by the language model.
    """
    llm = load_llm("gemini-pro")
    response = llm.invoke(input_text)
    return response

# Streamlit page configuration
st.set_page_config(
    page_title="Gemini Assistant Chatbot",
    page_icon=":speech_balloon:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
)

st.title("Gemini Assistant Chatbot💬")
st.caption("🤖 Your AI-powered assistant for any questions.")

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }
    ]

# User input prompt
prompt = st.chat_input("Your question: ")

# If the user provides a prompt, add it to the session state messages
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate and display the assistant's response if the last message is from the user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            response = generate_response(prompt)
            st.write(response.content)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )

# Sidebar content
with st.sidebar:
    st.image("./icons/chatbot.png", width=100)
    st.markdown("## Gemini Assistant Chatbot")
    st.markdown("Ask any question and receive answers from our powerful AI.")
    st.markdown("---")
    st.markdown("[Open in GitHub](https://github.com/acn-thaihanguyen/gemini_chatbot)")
    st.markdown("[Author: Nguyen Thai Ha]")
