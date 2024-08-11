import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api"]
os.environ["SERPAPI_API_KEY"] = st.secrets["Serpapi_API_Key"]
llm = OpenAI()


def google(question):
    tools = load_tools(["serpapi"])
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    response = agent.run(question)
    return response