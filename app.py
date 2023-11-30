import streamlit as st
from agent import MemoryAgent
from datastore import PineconeDatastore
from prompts import INFO_AGENT_PROMPT
import openai
import os 
from dotenv import load_dotenv

load_dotenv(override=True)
openai.api_key = os.environ.get('OPENAI_API_KEY')

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    selected_tab = st.radio("Select a window:", ["Chat", "Memory"])
    user = st.radio("Select a user:", ["Alice", "Brad"])
    display_thoughts = st.radio("Display internal thoughts:", ["Off", "On"])

# Instantiating the agent:
datastore = PineconeDatastore(user)
agent = MemoryAgent(datastore, INFO_AGENT_PROMPT, model='gpt-4-1106-preview', metadata=f"The current user's name is {user}.")

if selected_tab == "Chat":
    st.title("Chat")

    if prompt := st.chat_input("Copy problem here!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating model response..."):
                if display_thoughts == 'Off':
                    response = agent.run(prompt)
                elif display_thoughts == 'On':
                    response = agent.run(prompt, return_thoughts=True)
            st.markdown(response)

elif selected_tab == "Memory":
    st.title("Memory")
    st.button("Erase Memories", on_click=datastore.delete_vectors, type="primary")
    data = datastore.get_all_memories()
    for row in data:
        st.text(row)


# TODO: Case where the array is empty
# Case where the input does not really require any input to memory (make sure the agent can handle that)


