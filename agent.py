import openai
from prompts import INFO_AGENT_PROMPT
import os
from datastore import PineconeDatastore
import streamlit as st

openai.api_key = os.environ.get('OPENAI_API_KEY')

class MemoryAgent:
    def __init__(self, datastore, init_prompt, model='gpt-4', metadata=None):
        self.datastore = datastore
        self.model = model
        self.init_prompt = init_prompt
        # All messages
        self.messages = [{'role': 'user', 'content': init_prompt}]
        # The messages that the user can see
        self.con_messages = self.messages.copy()
        if metadata:
            self.messages.append({'role': 'user', 'content': '[METADATA]: ' + metadata})
              
    def run(self, text, return_thoughts=False):
        self.messages.append({'role': 'user', 'content': 'User: ' + text})
        self.con_messages.append({'role': 'user', 'content': text})
        counter = 0
        while counter < 5:
            content = self._loop(return_thoughts=return_thoughts)
            if content:
                return content
            counter += 1
        return None

    # One LLM generation (ending with a function call)
    def _loop(self, return_thoughts):
        stop = ['\nObservation:']
        thought = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            stop=stop,
        )
        self.messages.append(thought["choices"][0]["message"])
        thought_content = thought["choices"][0]["message"]["content"]
        if return_thoughts:
            st.markdown("MODEL THOUGHT")
            st.markdown(thought_content)
        observation = None
        try:
            thought, action = thought_content.strip().split('\nAction:')
            function_call, args = action.split('[')
            function_call = function_call.strip().lower()
            if len(args) > 0:
                args = args.strip()
                args = args[:-1]
            if function_call == "read":
                print("Reading...")
                observation = self.datastore.read(args)
            elif function_call == "write": 
                print("Writing...")
                observation = self.datastore.write(args)
            elif function_call == "finish":
                print("Finishing...")
                self.con_messages.append({'role': 'assistant', 'content': args})
                self.messages.append({'role': 'assistant', 'content': args})
                return args
            else: 
                print(f"Bad function call: {function_call}")
                return None
        except Exception as e:
            print(e)
            print("Could not derive function call")
            return "String Parse Error"
        
        if observation:
            if return_thoughts:
                st.markdown("FUNCTION OUTPUT")
                st.markdown(observation)
            self.messages.append({'role': 'function', "name": function_call, 'content': 'Observation: ' + observation})
            print('---------')
            print("Memory Output: ")
            print(observation)
            print('---------')
        return None

def main():
    datastore = PineconeDatastore("Brad")
    datastore.write("Brad and Jenny broke up 5 years ago.")
    datastore.write("Brad and Jenny don't like each other very much")
    datastore.write("Jenny has three dogs.")
    agent = MemoryAgent(datastore, INFO_AGENT_PROMPT, model='gpt-4-1106-preview', metadata="The current user's name is Brad.")

    while True:
        user_input = input("Talk to the agent: ")
        output = agent.run(user_input)
        print(output)

if __name__ == '__main__': 
    main()