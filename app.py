import streamlit as st
from agent import MemoryAgent
from datastore import PineconeDatastore
from prompts import INFO_AGENT_PROMPT
import openai
import os 
from dotenv import load_dotenv

load_dotenv(override=True)
openai.api_key = os.environ.get('OPENAI_API_KEY')

data = [
    "Row 1: Lorem ipsum dolor sit amet",
    "Row 2: Consectetur adipiscing elit",
    "Row 3: Sed do eiusmod tempor incididunt",
    "Row 4: Ut labore et dolore magna aliqua",
    "Row 5: Ut enim ad minim veniam",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    selected_tab = st.radio("Select a Tab", ["Chat", "Memory"])

# Instantiating the agent:
datastore = PineconeDatastore("Brad")
agent = MemoryAgent(datastore, INFO_AGENT_PROMPT, model='gpt-4-1106-preview', metadata="The current user's name is Brad.")

if selected_tab == "Chat":
    st.title("Chat")

    if prompt := st.chat_input("Copy problem here!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating model response..."):
                response = agent.run(prompt)
            st.markdown(response)

elif selected_tab == "Memory":
    st.title("Memory")
    for row in data:
        st.text(row)


# TODO: Case where the array is empty
# Case where the input does not really require any input to memory (make sure the agent can handle that)

