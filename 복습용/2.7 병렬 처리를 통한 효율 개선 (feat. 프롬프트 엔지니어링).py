# %% [markdown]
# # 2.7 병렬 처리를 통한 효율 개선 (feat. 프롬프트 엔지니어링)

# %% [markdown]
# - 아주 specific한 에이전트를 개발하는 경우 유리합니다 
# - 답변을 생성할 때 다양한 정보가 필요하다면 병렬 처리를 통해 시간을 절약할 수 있습니다

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str  # 사용자 질문
    answer: str  # 세율
    tax_base_equation: str  # 과세표준 계산 수식
    tax_deduction: str  # 공제액
    market_ratio: str  # 공정시장가액비율
    tax_base: str  # 과세표준 계산


graph_builder = StateGraph(AgentState)

# %%
import os

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
embedding_function = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
)
vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name="real_estate_tax",
    persist_directory="./real_estate_tax_collection",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
query = "5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?"

# %%
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = hub.pull("rlm/rag-prompt")

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-2024-11-20",
    api_version="2024-08-01-preview",
)

# %%
tax_base_retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 리턴해주세요",
        ),
        ("human", "{tax_base_equation_information}"),
    ]
)

tax_base_equation_chain = (
    {"tax_base_equation_information": RunnablePassthrough()}
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()
)

tax_base_chain = {
    "tax_base_equation_information": tax_base_retrieval_chain
} | tax_base_equation_chain


def get_tax_base_equation(state: AgentState) -> AgentState:
    """
    종합부동산세 과세표준을 계산하는 수식을 가져옵니다.
    `node`로 활용되기 때문에 `state`를 인자로 받지만,
    고정된 기능을 수행하기 때문에 `state`를 활용하지는 않습니다.

    Args:
        state (AgentState): 현재 에이전트의 상태를 나타내는 객체입니다.

    Returns:
        AgentState: 'tax_base_equation' 키를 포함하는 새로운 `state`를 반환합니다.
    """
    # 과세표준을 계산하는 방법을 묻는 질문을 정의합니다.
    tax_base_equation_question = "주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요"

    # tax_base_chain을 사용하여 질문을 실행하고 결과를 얻습니다.
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)

    # state에서 'tax_base_equation' 키에 대한 값을 반환합니다.
    return {"tax_base_equation": tax_base_equation}

# %%
tax_deduction_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


def get_tax_deduction(state: AgentState) -> AgentState:
    """
    종합부동산세 공제금액에 관한 정보를 가져옵니다.
    `node`로 활용되기 때문에 `state`를 인자로 받지만,
    고정된 기능을 수행하기 때문에 `state`를 활용하지는 않습니다.

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.

    Returns:
        AgentState: 'tax_deduction' 키를 포함하는 새로운 state를 반환합니다.
    """
    # 공제금액을 묻는 질문을 정의합니다.
    tax_deduction_question = "주택에 대한 종합부동산세 계산시 공제금액을 알려주세요"

    # tax_deduction_chain을 사용하여 질문을 실행하고 결과를 얻습니다.
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)

    # state에서 'tax_deduction' 키에 대한 값을 반환합니다.
    return {"tax_deduction": tax_deduction}

# %%
from langchain_community.tools import TavilySearchResults
from datetime import date

tavily_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}",
        ),
        ("human", "{query}"),
    ]
)


def get_market_ratio(state: AgentState) -> AgentState:
    """
    web 검색을 통해 주택 공시가격에 대한 공정시장가액비율을 가져옵니다.
    `node`로 활용되기 때문에 `state`를 인자로 받지만,
    고정된 기능을 수행하기 때문에 `state`를 활용하지는 않습니다.

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.

    Returns:
        AgentState: 'market_ratio' 키를 포함하는 새로운 state를 반환합니다.
    """
    # 오늘 날짜에 해당하는 공정시장가액비율을 묻는 쿼리를 정의합니다.
    query = f"오늘 날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇%인가요?"

    # tavily_search_tool을 사용하여 쿼리를 실행하고 컨텍스트를 얻습니다.
    context = tavily_search_tool.invoke(query)

    # tax_market_ratio_chain을 구성하여 쿼리와 컨텍스트를 처리합니다.
    tax_market_ratio_chain = tax_market_ratio_prompt | llm | StrOutputParser()

    # tax_market_ratio_chain을 사용하여 시장 비율을 계산합니다.
    market_ratio = tax_market_ratio_chain.invoke({"context": context, "query": query})

    # state에서 'market_ratio' 키에 대한 값을 반환합니다.
    return {"market_ratio": market_ratio}

# %%
tax_base_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
주어진 내용을 기반으로 과세표준을 계산해주세요

과세표준 계산 공식: {tax_base_equation}
공제금액: {tax_deduction}
공정시장가액비율: {market_ratio}""",
        ),
        ("human", "사용자 주택 공시가격 정보: {query}"),
    ]
)


def calculate_tax_base(state: AgentState) -> AgentState:
    """
    주어진 state에서 과세표준을 계산합니다.

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.

    Returns:
        AgentState: 'tax_base' 키를 포함하는 새로운 state를 반환합니다.
    """
    # state에서 필요한 정보를 추출합니다.
    tax_base_equation = state["tax_base_equation"]
    tax_deduction = state["tax_deduction"]
    market_ratio = state["market_ratio"]
    query = state["query"]

    # tax_base_calculation_chain을 구성하여 과세표준을 계산합니다.
    tax_base_calculation_chain = tax_base_calculation_prompt | llm | StrOutputParser()

    # tax_base_calculation_chain을 사용하여 과세표준을 계산합니다.
    tax_base = tax_base_calculation_chain.invoke(
        {
            "tax_base_equation": tax_base_equation,
            "tax_deduction": tax_deduction,
            "market_ratio": market_ratio,
            "query": query,
        }
    )

    # state에서 'tax_base' 키에 대한 값을 반환합니다.
    return {"tax_base": tax_base}

# %%
tax_rate_calculation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요

종합부동산세 세율:{context}""",
        ),
        (
            "human",
            """과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요

과세표준: {tax_base}
주택 수:{query}""",
        ),
    ]
)


def calculate_tax_rate(state: AgentState):
    """
    주어진 state에서 세율을 계산합니다.

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.

    Returns:
        dict: 'answer' 키를 포함하는 새로운 state를 반환합니다.
    """
    # state에서 필요한 정보를 추출합니다.
    query = state["query"]
    tax_base = state["tax_base"]

    # retriever를 사용하여 쿼리를 실행하고 컨텍스트를 얻습니다.
    context = retriever.invoke(query)

    # tax_rate_chain을 구성하여 세율을 계산합니다.
    tax_rate_chain = tax_rate_calculation_prompt | llm | StrOutputParser()

    # tax_rate_chain을 사용하여 세율을 계산합니다.
    tax_rate = tax_rate_chain.invoke(
        {"context": context, "tax_base": tax_base, "query": query}
    )

    # state에서 'answer' 키에 대한 값을 반환합니다.
    return {"answer": tax_rate}

# %% [markdown]
# - `node`를 추가하고 `edge`로 연결합니다
# - 하나의 `node`에서 `edge`를 활영해서 다양한 `node`들을 연결하면 병렬로 작업이 가능합니다

# %%
graph_builder.add_node("get_tax_base_equation", get_tax_base_equation)
graph_builder.add_node("get_tax_deduction", get_tax_deduction)
graph_builder.add_node("get_market_ratio", get_market_ratio)
graph_builder.add_node("calculate_tax_base", calculate_tax_base)
graph_builder.add_node("calculate_tax_rate", calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "get_tax_base_equation")
graph_builder.add_edge(START, "get_tax_deduction")
graph_builder.add_edge(START, "get_market_ratio")
graph_builder.add_edge("get_tax_base_equation", "calculate_tax_base")
graph_builder.add_edge("get_tax_deduction", "calculate_tax_base")
graph_builder.add_edge("get_market_ratio", "calculate_tax_base")
graph_builder.add_edge("calculate_tax_base", "calculate_tax_rate")
graph_builder.add_edge("calculate_tax_rate", END)

# %%
graph = graph_builder.compile()

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %%
initial_state = {"query": query}
graph.invoke(initial_state)

# %% [markdown]
# {'query': '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?',
# 
# 
# 
#  'answer': '주어진 정보를 바탕으로 사용자의 종합부동산세를 계산해드리겠습니다. 사용자가 소유한 주택은 총 **3채**이며, 각각 5억 원, 10억 원, 20억 원짜리 주택으로 총 공시가격 합계는 35억 원입니다. 이를 기준으로 계산을 진행하겠습니다.\n\n---\n\n### 1. 주택 소유 수에 따른 구분\n사용자는 **3주택 이상 소유자**로 간주됩니다. 따라서 "납세의무자가 3주택 이상을 소유한 경우"에 해당되는 세율을 적용합니다.\n\n---\n\n### 2. 과세표준 계산\n**공제금액**과 **공정시장가액비율**을 적용하여 과세표준을 계산합니다.\n\n- **공시가격 합계**: 35억 원\n- **공제금액** (3주택 이상 소유자): 9억 원\n- **공정시장가액비율**: 60%\n\n\\[\n\\text{과세표준} = (\\text{공시가격 합계} - \\text{공제금액}) \\times \\text{공정시장가액비율}\n\\]\n\n\\[\n\\text{과세표준} = (35억 - 9억) \\times 60\\% = 26억 \\times 0.6 = 15.6억 원\n\\]\n\n**결과: 과세표준 = 15.6억 원**\n\n---\n\n### 3. 세율에 따른 종합부동산세 계산\n사용자가 3주택 이상을 소유했으므로, 아래 세율표를 적용합니다.\n\n| **과세표준**              | **세율**                                  |\n|---------------------------|-------------------------------------------|\n| 3억 원 이하               | 1천분의 5                                 |\n| 3억 원 초과 6억 원 이하    | 150만 원 + (3억 원 초과분 × 1천분의 7)    |\n| 6억 원 초과 12억 원 이하   | 360만 원 + (6억 원 초과분 × 1천분의 10)   |\n| 12억 원 초과 25억 원 이하  | 960만 원 + (12억 원 초과분 × 1천분의 20)  |\n| 25억 원 초과 50억 원 이하  | 3,560만 원 + (25억 원 초과분 × 1천분의 30)|\n| 50억 원 초과 94억 원 이하  | 1억 1,600만 원 + (50억 원 초과분 × 1천분의 40)|\n| 94억 원 초과              | 2억 8,660만 원 + (94억 원 초과분 × 1천분의 50)|\n\n#### **과세표준 15.6억 원에 해당하는 구간: 12억 원 초과 25억 원 이하**\n- 기본세액: 960만 원\n- 초과분: \\((15.6억 - 12억) = 3.6억 원\\)\n- 추가세액: \\(3.6억 × 1천분의 20 = 720만 원\\)\n\n\\[\n\\text{종합부동산세} = \\text{기본세액} + \\text{추가세액} = 960만 원 + 720만 원 = 1,680만 원\n\\]\n\n---\n\n### 4. 최종 결과\n사용자가 소유한 3채의 주택(5억 원, 10억 원, 20억 원)의 공시가격 합계 35억 원을 기준으로 계산된 **종합부동산세는 1,680만 원**입니다.',
# 
# 
# 
#  'tax_base_equation': '과세표준 = (주택의 공시가격 합계 - 공제금액) × 공정시장가액비율',
# 
# 
# 
#  'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제금액은 다음과 같습니다:  \n1세대 1주택자는 12억 원, 법인이나 법인으로 보는 단체는 6억 원, 그 외의 경우는 9억 원입니다.',
# 
# 
# 
#  'market_ratio': '주택 공시가격의 공정시장가액비율은 **60%**입니다. 이는 제공된 정보에서 지방세법 시행령에 따라 주택은 60%, 토지와 건축물은 70%로 정해져 있다고 명시되어 있습니다.',
# 
# 
# 
#  'tax_base': '과세표준을 계산하기 위해 주어진 정보를 바탕으로 단계를 나누어 계산해보겠습니다.\n\n---\n\n### 1. **주택 공시가격 합계 계산**\n사용자가 소유한 주택의 공시가격:\n- 5억 원짜리 주택 1채\n- 10억 원짜리 주택 1채\n- 20억 원짜리 주택 1채\n\n총 공시가격 합계:\n\\[\n5억 + 10억 + 20억 = 35억 원\n\\]\n\n---\n\n### 2. **공제금액 적용**\n공제금액은 주택 소유자의 유형에 따라 다릅니다. 주어진 정보에서는 사용자가 **1세대 1주택자**인지, **법인이나 단체**인지, 아니면 **그 외의 경우**인지를 명시하지 않았으므로, 각각의 경우를 계산해 보겠습니다.\n\n- **1세대 1주택자 공제금액**: 12억 원\n- **법인/법인으로 보는 단체 공제금액**: 6억 원\n- **그 외의 경우 공제금액**: 9억 원\n\n---\n\n### 3. **과세표준 계산**\n과세표준 공식:\n\\[\n\\text{과세표준} = (\\text{공시가격 합계} - \\text{공제금액}) \\times \\text{공정시장가액비율}\n\\]\n\n공정시장가액비율: **60%**\n\n#### (1) 1세대 1주택자일 경우:\n\\[\n\\text{과세표준} = (35억 - 12억) \\times 60\\%\n\\]\n\\[\n\\text{과세표준} = 23억 \\times 0.6 = 13.8억 원\n\\]\n\n#### (2) 법인/법인으로 보는 단체일 경우:\n\\[\n\\text{과세표준} = (35억 - 6억) \\times 60\\%\n\\]\n\\[\n\\text{과세표준} = 29억 \\times 0.6 = 17.4억 원\n\\]\n\n#### (3) 그 외의 경우:\n\\[\n\\text{과세표준} = (35억 - 9억) \\times 60\\%\n\\]\n\\[\n\\text{과세표준} = 26억 \\times 0.6 = 15.6억 원\n\\]\n\n---\n\n### 4. **결론**\n- **1세대 1주택자**: 과세표준 = **13.8억 원**\n- **법인/법인으로 보는 단체**: 과세표준 = **17.4억 원**\n- **그 외의 경우**: 과세표준 = **15.6억 원**\n\n위의 과세표준을 기준으로 종합부동산세를 계산하게 됩니다. 세율은 과세표준의 구간에 따라 달라지므로, 추가적인 세율 정보가 필요하다면 알려주세요!'}


