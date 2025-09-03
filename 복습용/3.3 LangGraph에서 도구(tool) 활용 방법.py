# %% [markdown]
# # 3.3 LangGraph에서 도구(tool) 활용 방법

# %% [markdown]
# - 도구(tool)를 활용한 에이전트 개발 방법을 알아봅니다
# - workflow를 직접 선언하지 않고, 사용가능한 도구들을 전달하면, 에이전트가 적합한 도구를 판단해서 사용합니다
#     - 이번 회차에서는 `ToolNode`를 통해 도구를 활용하는 방법을 알아봅니다

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-2024-11-20",
    api_version="2024-08-01-preview",
)

small_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini-2024-07-18",
    api_version="2024-08-01-preview",
)

# %%
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """숫자 a와 b를 더합니다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """숫자 a와 b를 곱합니다."""
    return a * b

# %% [markdown]
# - LangGraph는 `ToolNode`를 통해 도구를 활용합니다
# - `ToolNode`의 `invoke()`결과는 도구의 `invoke()` 결과와 유사합니다
#     - 도구는 도구의 실행 결과를 리턴하고, `ToolNode`는 도구의 실행 결과를 포함한 `ToolMessage`를 리턴합니다

# %%
from langgraph.prebuilt import ToolNode

tool_list = [add, multiply]
llm_with_tools = small_llm.bind_tools(tool_list)
tool_node = ToolNode(tool_list)

# %%
multiply.invoke({"a": 3, "b": 5})

# %%
ai_message = llm_with_tools.invoke("What is 3 plus 5?")

# %%
ai_message

# %% [markdown]
# - `ToolNode`를 `invoke()`하려면 `tool_calls` 속성을 포함한 `AIMessage`를 전달해야 합니다

# %%
tool_node.invoke({"messages": [ai_message]})

# %% [markdown]
# - 간단한 에이전트를 만들기 위해 LangGraph에서 제공하는 [`StateGraph`](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate)를 사용합니다

# %%
from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)

# %%
def agent(state: MessagesState) -> MessagesState:
    """
    에이전트 함수는 주어진 상태에서 메시지를 가져와
    LLM과 도구를 사용하여 응답 메시지를 생성합니다.

    Args:
        state (MessagesState): 메시지 상태를 포함하는 state.

    Returns:
        MessagesState: 응답 메시지를 포함하는 새로운 state.
    """
    # 상태에서 메시지를 추출합니다.
    messages = state["messages"]

    # LLM과 도구를 사용하여 메시지를 처리하고 응답을 생성합니다.
    response = llm_with_tools.invoke(messages)

    # 응답 메시지를 새로운 상태로 반환합니다.
    return {"messages": [response]}

# %%
from typing import Literal
from langgraph.graph import END


def should_continue(state: MessagesState) -> Literal["tools", END]:
    """
    주어진 메시지 상태를 기반으로 에이전트가 계속 진행할지 여부를 결정합니다.

    Args:
        state (MessagesState): `state`를 포함하는 객체.

    Returns:
        Literal['tools', END]: 도구를 사용해야 하면 `tools`를 리턴하고,
        답변할 준비가 되었다면 END를 반환해서 프로세스를 종료합니다.
    """
    # 상태에서 메시지를 추출합니다.
    messages = state["messages"]

    # 마지막 AI 메시지를 가져옵니다.
    last_ai_message = messages[-1]

    # 마지막 AI 메시지가 도구 호출을 포함하고 있는지 확인합니다.
    if last_ai_message.tool_calls:
        # 도구 호출이 있으면 'tools'를 반환합니다.
        return "tools"

    # 도구 호출이 없으면 END를 반환하여 프로세스를 종료합니다.
    return END

# %% [markdown]
# - `node`를 추가하고 `edge`로 연결합니다

# %%
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
graph_builder.add_edge("tools", "agent")

# %%
graph = graph_builder.compile()

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# - `graph.stream()`을 활용하면 에이전트가 답변을 생성하는 과정을 모니터링 할 수 있습니다

# %%
from langchain_core.messages import HumanMessage

for chunk in graph.stream(
    {"messages": [HumanMessage("3에다 5를 더하고 거기에 8을 곱하면?")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

# %%



