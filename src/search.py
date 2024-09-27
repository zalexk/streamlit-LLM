from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
import streamlit as st
import os

llm = ChatOpenAI(openai_api_key = st.secrets["openai_api"],
                    model = "gpt-4o",
                    temperature = 0.2,
                    base_url = st.secrets["base_url"])

os.environ["SERPAPI_API_KEY"] = st.secrets["Serpapi_API_Key"]

def google(question):
    tools = load_tools(["serpapi"])
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
    response = agent.run(question)
    return response
