# %% [markdown]
# # 2.8 Multi-Agent 시스템과 RouteLLM

# %% [markdown]
# - 앞에서 개발한 `소득세 에이전트`와 `종합부동산세 에이전트`를 활용해서 다중 에이전트 시스템을 구현합니다

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
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str
    context: list
    answer: str


graph_builder = StateGraph(AgentState)

# %%
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal


class Route(BaseModel):
    target: Literal["income_tax", "llm", "real_estate_tax"] = Field(
        description="The target for the query to answer"
    )


router_system_prompt = """
You are an expert at routing a user's question to 'income_tax', 'llm', or 'real_estate_tax'.
'income_tax' contains information about income tax up to December 2024.
'real_estate_tax' contains information about real estate tax up to December 2024.
if you think the question is not related to either 'income_tax' or 'real_estate_tax';
you can route it to 'llm'."""


router_prompt = ChatPromptTemplate.from_messages(
    [("system", router_system_prompt), ("user", "{query}")]
)

structured_router_llm = small_llm.with_structured_output(Route)


def router(state: AgentState) -> Literal["income_tax", "real_estate_tax", "llm"]:
    """
    주어진 state에서 쿼리를 기반으로 적절한 경로를 결정합니다.

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.

    Returns:
        Literal['income_tax', 'real_estate_tax', 'llm']: 쿼리에 따라 선택된 경로를 반환합니다.
    """
    query = state["query"]
    router_chain = router_prompt | structured_router_llm
    route = router_chain.invoke({"query": query})

    return route.target

# %%
from langchain_core.output_parsers import StrOutputParser


def call_llm(state: AgentState) -> AgentState:
    """
    주어진 state에서 쿼리를 LLM에 전달하여 응답을 얻습니다.

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.

    Returns:
        AgentState: 'answer' 키를 포함하는 새로운 state를 반환합니다.
    """
    query = state["query"]
    llm_chain = small_llm | StrOutputParser()
    llm_answer = llm_chain.invoke(query)
    return {"answer": llm_answer}

# %% [markdown]
# - `node`를 추가하고 `edge`로 연결합니다
#     - 앞에서 개발한 `agent`들을 `node`로 활용할 수 있습니다

# %%
from income_tax_graph import graph as income_tax_agent
from real_estate_tax_graph import graph as real_estate_tax_agent

graph_builder.add_node("income_tax", income_tax_agent)
graph_builder.add_node("real_estate_tax", real_estate_tax_agent)
graph_builder.add_node("llm", call_llm)

# %%
from langgraph.graph import START, END

graph_builder.add_conditional_edges(
    START,
    router,
    {"income_tax": "income_tax", "real_estate_tax": "real_estate_tax", "llm": "llm"},
)
graph_builder.add_edge("income_tax", END)
graph_builder.add_edge("real_estate_tax", END)
graph_builder.add_edge("llm", END)

# %%
graph = graph_builder.compile()

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %%
initial_state = {"query": "소득세란 무엇인가요?"}
graph.invoke(initial_state)

# %%
initial_state = {"query": "집 15억은 세금을 얼마나 내나요?"}

graph.invoke(initial_state)

# %%
initial_state = {"query": "떡볶이는 어디가 맛있나요?"}
graph.invoke(initial_state)

# %%



