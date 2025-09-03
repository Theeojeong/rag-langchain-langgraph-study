# %% [markdown]
# # 2.1 LangChain vs LangGraph (feat. LangGraph 개념 설명)
# 
# - LangChain을 활용한 간단한 `llm.invoke()` 예제를 살펴보고, 이를 LangGraph로 구현해보는 과정을 진행합니다.
# - LangGraph의 개념과 주요 기능을 이해하고, 두 프레임워크의 차이점을 비교합니다.

# %% [markdown]
# ## 환경설정
# 
# - `LangChain` 활용을 위해 필요한 패키지들을 설치합니다
# - 최신 버전을 설치해도 정상적으로 동작해야 하지만, 버전 명시가 필요하다면 `requirements.txt`를 참고해주세요

# %%
%pip install -q python-dotenv langchain-openai

# %% [markdown]
# - 먼저 `.env` 파일의 환경변수를 불러옵니다
# - `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` 등과 같이 환경변수를 설정하면 편하게 사용할 수 있습니다

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_openai import ChatOpenAI

query = "인프런에는 어떤 강의가 있나요?"

llm = ChatOpenAI(model="gpt-4o-mini")  # 테스트의 경우에는 작은 모델을 사용합니다
llm.invoke(query)

# %% [markdown]
# - `LangGraph` 활용을 위해 필요한 패키지를 설치합니다
# - 최신 버전을 설치해도 정상적으로 동작해야 하지만, 버전 명시가 필요하다면 `requirements.txt`를 참고해주세요

# %%
%pip install -q langgraph

# %% [markdown]
# - `state`는 LangGraph 에이전트의 state를 나타내는 데이터 구조입니다.
# - `state`는 `TypedDict`를 사용하여 정의되며, 이는 Python의 타입 힌팅을 통해 구조를 명확히 합니다.
#     - 지금 예제에서는 간단하게 `messages`라는 필드만 있습니다.
#     - 필요에 따라 다양한 값들을 활용할 수 있습니다.
#         - 2.2 회차에서 다룰 예정입니다.
# - `state`는 에이전트의 동작을 결정하는 데 사용되며, 각 노드에서 state를 업데이트하거나 참조할 수 있습니다.
# - `state`는 LangGraph의 노드 간에 전달되며, 에이전트의 state 전이를 관리합니다.

# %%
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    messages: list[Annotated[AnyMessage, add_messages]]

# %% [markdown]
# - 위에 선언한 `AgentState`를 활용하여 `StateGraph`를 생성합니다.

# %%
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)

# %% [markdown]
# - `graph`에 추가할 `node`를 생성합니다
# -  `node`는 LangGraph에서 실행되는 개별적인 작업 단위를 의미합니다. 
#     - 각 노드는 특정 기능을 수행하는 독립적인 컴포넌트로, 예를 들어 텍스트 생성, 데이터 처리, 또는 의사 결정과 같은 작업을 담당할 수 있습니다.
#     - `node`는 기본적으로 함수(function)로 정의되고, 뒤에서 다루지만 다른 에이전트(agent)를 활용할 수도 있습니다

# %%
def generate(state: AgentState) -> AgentState:
    """
    `generate` 노드는 사용자의 질문을 받아서 응답을 생성하는 노드입니다.
    """
    messages = state["messages"]
    ai_message = llm.invoke(messages)
    return {"messages": [ai_message]}

# %% [markdown]
# - `node`를 생성한 후에 `edge`로 연결합니다
# - `edge`는 노드들 사이의 연결을 나타내며, 데이터와 제어 흐름의 경로를 정의합니다. 
#     - 엣지를 통해 한 노드의 출력이 다음 노드의 입력으로 전달되어, 전체적인 워크플로우가 형성됩니다.
#     - `node`와 `edge`의 조합은 방향성 그래프(Directed Graph)를 형성하며, 이를 통해 복잡한 AI 에이전트의 행동 흐름을 구조화할 수 있습니다

# %%
graph_builder.add_node("generate", generate)

# %% [markdown]
# - 모든 그래프는 `START(시작)`와 `END(종료)`가 있습니다
#     - `END`를 explicit하게 선언하지 않는 경우도 종종 있지만, 가독성을 위해 작성해주는 것을 권장합니다

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

# %% [markdown]
# - `node`를 생성하고 `edge`로 연결한 후에 `compile` 메서드를 호출하여 `Graph`를 생성합니다

# %%
graph = graph_builder.compile()

# %% [markdown]
# - `compile` 후에는 그래프를 시각화하여 확인할 수 있습니다
# - 의도한대로 그래프가 생성됐는지 확인하는 습관을 기르는 것이 좋습니다
#     - `git`에서 코드 작업물을 commit하기 전에 `git diff`를 통해 변경사항을 확인하는 것과 같습니다

# %%
from IPython.display import display, Image

display(Image(graph.get_graph().draw_mermaid_png()))

# %%
from langchain_core.messages import HumanMessage

initial_state = {"messages": [HumanMessage(query)]}
graph.invoke(initial_state)


