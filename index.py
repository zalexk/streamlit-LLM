import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Index | iAsk",
    page_icon="👋",
    initial_sidebar_state = "collapsed"
)

st.title("👋 iAsk 你的數碼小助手")

st.session_state["login_status"] = True # Remove password protection

if "login_status" not in st.session_state or st.session_state["login_status"] == False:
    with st.form("login"):
        st.markdown("## Login 登錄")
        password = st.text_input("**Password 密碼**", type = "password", autocomplete ="password")
        submit = st.form_submit_button()
        
        if submit == True and password.strip() == st.secrets["login"]:
            st.session_state["login_status"] = True
            st.rerun()
            
        else:
            st.session_state["login_status"] = False 
            
            if submit == True and password != st.secrets["login"]: 
                st.write(":red[Wrong password.Please try again]")
if st.session_state["login_status"] == True:
    st.toast('Successfully logged in', icon="✅")
    st.markdown("""
                ## 簡單介紹
                大家好，歡迎來到我們的網站。 👏
                
                我們希望透過這個項目，讓你們在數碼世界中遇到任何挑戰，都求助有門。
                
                遇到不懂的歡迎前來詢問，我們的親切、耐心的人工智能机器人會努力解答你的一切問題的 😀
                
                """)

    st.markdown("## 實力評估")
    level = st.radio(
        "**評估一下自己使用 ChatGPT 的實力吧，好讓我們可以更好地幫助你 🤗**", 
        # Radio form's title + Bold the title
        ["🟡 不懂","🔵 熟練",],
        captions = ["只能用日常語言與 ChatGPT 溝通",
                    "懂得使用提示詞以獲得想要的回應"],
        index = None) # Set preselected option be None
    
    level_index = {"🟡 不懂":"begin", "🔵 熟練": "expert"}
    
    st.session_state["level"] = [] # Initiate
    
    if level is not None:
        st.session_state["level"] = level_index[level]
        st.switch_page("pages/chatbot.py")
        
    st.page_link("pages/quiz.py", label="❓ 不知道自己水平？請點擊這裡")

