import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_message

load_dotenv()

st.set_page_config(page_title="rag")

st.title("Kr Income Tax ChatBot🤖")
st.caption("Have any questions about income tax? Just ask!")


if 'message_list' not in st.session_state:
    st.session_state.message_list=[]


for messages in st.session_state.message_list:
    with st.chat_message(messages["role"]):
        st.write(messages["content"])


if user_input := st.chat_input(placeholder="입력칸"):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.message_list.append({"role": "user", "content": user_input})

    with st.spinner(text="AI가 답변을 생성중입니다"):
        ai_message = get_ai_message(user_input)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})