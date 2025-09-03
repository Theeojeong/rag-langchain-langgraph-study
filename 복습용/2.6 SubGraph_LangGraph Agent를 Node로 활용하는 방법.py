# %% [markdown]
# # 2.6 SubGraph: LangGraph Agent를 Node로 활용하는 방법

# %% [markdown]
# - [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)논문을 구현합니다
# - LangGraph 공식문서의 흐름을 따라갑니다
#     - 공식문서의 흐름은 간소화된 버전입니다
#     - 실제 논문과 유사한 구현은 3.3강을 참고해주세요
# 
# ![adaptive-rag](https://i.imgur.com/tbICSxY.png)

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str
    context: list
    answer: str


graph_builder = StateGraph(AgentState)

# %%
from langchain_community.tools import TavilySearchResults

tavily_search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)


def web_search(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 웹 검색을 수행합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 웹 검색 결과가 추가된 state를 반환합니다.
    """
    query = state["query"]
    results = tavily_search_tool.invoke(query)

    return {"context": results}

# %%
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LangChain 허브에서 프롬프트를 가져옵니다
generate_prompt = hub.pull("rlm/rag-prompt")
# OpenAI의 GPT-4o 모델을 사용합니다
generate_llm = ChatOpenAI(model="gpt-4o")


def web_generate(state: AgentState) -> AgentState:
    """
    주어진 문맥과 질문을 기반으로 답변을 생성합니다.

    Args:
        state (AgentState): 문맥과 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 답변을 포함한 state를 반환합니다.
    """
    # state에서 문맥과 질문을 추출합니다
    context = state["context"]
    query = state["query"]

    # 프롬프트와 모델, 출력 파서를 연결하여 체인을 생성합니다
    rag_chain = generate_prompt | generate_llm | StrOutputParser()

    # 체인을 사용하여 답변을 생성합니다
    response = rag_chain.invoke({"question": query, "context": context})

    # 생성된 답변을 'answer'로 반환합니다
    return {"answer": response}

# %% [markdown]
# - 간단한 질문에 답변을 하는 경우 작은 모델을 활용해서 비용을 저감하고, 답변 생성 속도를 향상시킬 수 있습니다

# %%
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# OpenAI의 GPT-4o-mini 모델을 사용합니다
basic_llm = ChatOpenAI(model="gpt-4o-mini")


def basic_generate(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 기본 답변을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 답변을 포함한 state를 반환합니다.
    """
    # state에서 질문을 추출합니다
    query = state["query"]

    # 기본 LLM 체인을 생성합니다
    basic_llm_chain = basic_llm | StrOutputParser()

    # 체인을 사용하여 답변을 생성합니다
    llm_response = basic_llm_chain.invoke(query)

    # 생성된 답변을 'answer'로 반환합니다
    return {"answer": llm_response}

# %% [markdown]
# - 사용자의 질문이 들어오면 `router` 노드에서 사용자의 질문을 분석해서 적절한 노드로 이동합니다
#     - 사용자의 질문에 관한 내용이 vector store에 있는 경우 `income_tax_agent` 노드로 이동합니다
#     - 사용자의 질문이 간단한 경우 `basic_generate` 노드로 이동합니다
#     - 사용자의 질문이 웹 검색을 통해 답변을 얻을 수 있는 경우 `web_search` 노드로 이동합니다

# %%
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal


class Route(BaseModel):
    target: Literal["vector_store", "llm", "web_search"] = Field(
        description="The target for the query to answer"
    )


router_system_prompt = """
You are an expert at routing a user's question to 'vector_store', 'llm', or 'web_search'.
'vector_store' contains information about income tax up to December 2024.
if you think the question is simple enough use 'llm'
if you think you need to search the web to answer the question use 'web_search'
"""


router_prompt = ChatPromptTemplate.from_messages(
    [("system", router_system_prompt), ("user", "{query}")]
)

router_llm = ChatOpenAI(model="gpt-4o-mini")
structured_router_llm = router_llm.with_structured_output(Route)


def router(state: AgentState) -> Literal["vector_store", "llm", "web_search"]:
    """
    사용자의 질문에 기반하여 적절한 경로를 결정합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        Literal['vector_store', 'llm', 'web_search']: 질문을 처리하기 위한 적절한 경로를 나타내는 문자열.
    """
    # state에서 질문을 추출합니다
    query = state["query"]

    # 프롬프트와 구조화된 라우터 LLM을 연결하여 체인을 생성합니다
    router_chain = router_prompt | structured_router_llm

    # 체인을 사용하여 경로를 결정합니다
    route = router_chain.invoke({"query": query})

    # 결정된 경로의 타겟을 반환합니다
    return route.target

# %% [markdown]
# - `node`를 추가하고 `edge`로 연결합니다

# %%
from income_tax_graph import graph as income_tax_subgraph

graph_builder.add_node("income_tax_agent", income_tax_subgraph)
graph_builder.add_node("web_search", web_search)
graph_builder.add_node("web_generate", web_generate)
graph_builder.add_node("basic_generate", basic_generate)

# %%
from langgraph.graph import START, END

graph_builder.add_conditional_edges(
    START,
    router,
    {
        "vector_store": "income_tax_agent",
        "llm": "basic_generate",
        "web_search": "web_search",
    },
)

graph_builder.add_edge("web_search", "web_generate")
graph_builder.add_edge("web_generate", END)
graph_builder.add_edge("basic_generate", END)
graph_builder.add_edge("income_tax_agent", END)

# %%
graph = graph_builder.compile()

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %%
initial_state = {"query": "대한민국의 수도는 어디인가요?"}
graph.invoke(initial_state)

# %%
initial_state = {"query": "연봉 5천만원인 거주자의 소득세는 얼마인가요?"}
graph.invoke(initial_state)

# %%
initial_state = {"query": "역삼 맛집을 추천해주세요"}
graph.invoke(initial_state)

# %%



