import os
from openai import OpenAI
import streamlit as st

client = OpenAI(
    api_key = st.secrets["openai_api"],
)

def call(*prompt): # [system_prompt, user_prompt]
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt[0], 
            },
            {
                "role": "user",
                "content": prompt[1],
            }
        ],
        model = "gpt-4o-mini",
        temperature =  0.1
        
    )
    return chat_completion.choices[0].message.content