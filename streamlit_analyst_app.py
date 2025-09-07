from __future__ import annotations

import os
from typing import List, Dict, Any
import json
import uuid

import streamlit as st
from dotenv import load_dotenv

from enhanced_econ_agent import build_enhanced_graph


load_dotenv()


st.set_page_config(page_title="경제 애널리스트 에이전트", layout="wide")
st.title("경제 애널리스트 에이전트 (고도화)")


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_enhanced_graph()


DATA_DIR = os.path.join(os.getcwd(), "conversations")


def _conv_path(conv_id: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    # sanitize filename
    safe = "".join(c for c in conv_id if c.isalnum() or c in ("-", "_"))
    return os.path.join(DATA_DIR, f"{safe}.json")


def load_messages(conv_id: str) -> List[Dict[str, Any]]:
    path = _conv_path(conv_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
        except Exception:
            pass
    return []


def save_messages(conv_id: str, messages: List[Dict[str, Any]]):
    path = _conv_path(conv_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _new_conv_id() -> str:
    return f"conv-{uuid.uuid4().hex[:8]}"


if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = _new_conv_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_from_disk" not in st.session_state:
    st.session_state.loaded_from_disk = False

with st.sidebar:
    st.header("설정")
    ticker = st.text_input("Ticker", value="SNOW")
    st.caption("예: AAPL, MSFT, NVDA, SNOW, ^GSPC ...")
    st.subheader("대화")
    conv_col1, conv_col2 = st.columns([3, 1])
    with conv_col1:
        conv_id_input = st.text_input("대화 ID", value=st.session_state.conversation_id)
    with conv_col2:
        new_chat = st.button("새 대화")
    if new_chat:
        st.session_state.conversation_id = _new_conv_id()
        st.session_state.messages = []
        st.session_state.loaded_from_disk = True  # prevent immediate reload
        st.rerun()
    # If user edited the id, attempt a one-time load
    if conv_id_input != st.session_state.conversation_id:
        st.session_state.conversation_id = conv_id_input
        st.session_state.messages = load_messages(st.session_state.conversation_id)
        st.session_state.loaded_from_disk = True
        st.rerun()

    focus_market = st.checkbox("시장 뉴스/시세", True)
    focus_technical = st.checkbox("기술적 분석", True)
    focus_macro = st.checkbox("거시 지표", True)
    focus_company = st.checkbox("기업 리서치", True)
    st.divider()
    st.caption("Azure OpenAI 환경변수는 .env에서 로드됩니다")

# First-load: try to restore messages for the current conversation id
if not st.session_state.loaded_from_disk:
    loaded = load_messages(st.session_state.conversation_id)
    if loaded:
        st.session_state.messages = loaded
    st.session_state.loaded_from_disk = True


def make_directive() -> str:
    focuses = []
    if focus_market:
        focuses.append("market_research")
    if focus_technical:
        focuses.append("technical_analysis")
    if focus_macro:
        focuses.append("macro_context")
    if focus_company:
        focuses.append("company_research")
    if not focuses:
        focuses = ["market_research", "technical_analysis", "macro_context", "company_research"]
    return (
        "Focus on the following workers first (if relevant): " + ", ".join(focuses)
    )


# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


user_input = st.chat_input("질문을 입력하세요 (예: 이 종목 투자해도 될까요?)")

if user_input:
    # Augment the user request with ticker + focus directive
    directive = make_directive()
    prompt = f"{user_input}\n\nTicker: {ticker}\n{directive}"

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_messages(st.session_state.conversation_id, st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        trace_log = st.empty()
        out_text = ""
        trace_lines: List[str] = []

        graph = get_graph()
        last_assistant: str | None = None

        # Build history for the agent: pass prior turns
        history = [(m["role"], m["content"]) for m in st.session_state.messages]
        for chunk in graph.stream({"messages": history}, stream_mode="values"):
            msg = chunk["messages"][-1]
            name = getattr(msg, "name", None) or getattr(msg, "type", "")
            content = str(msg.content)

            # The final analyst node returns an AI message; intermediate nodes add HumanMessages with names
            if getattr(msg, "type", None) == "ai":
                last_assistant = content
                out_text = content
                placeholder.markdown(out_text)
            else:
                trace_lines.append(f"- {name}: {content}")
                trace_log.markdown("\n".join(trace_lines))

        if last_assistant is None:
            # Fallback to last content if analyst message wasn't labeled
            last_assistant = out_text or (trace_lines[-1] if trace_lines else "")
            placeholder.markdown(last_assistant)

    st.session_state.messages.append({"role": "assistant", "content": last_assistant or ""})
    save_messages(st.session_state.conversation_id, st.session_state.messages)

st.caption("© Enhanced agent runs without modifying original files.")
