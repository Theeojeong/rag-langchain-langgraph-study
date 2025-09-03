# %% [markdown]
# # 2.2 간단한 Retrieval 에이전트 (feat. PDF 전처리 꿀팁)

# %% [markdown]
# - RAG 에이전트를 만들어봅니다
# - [zerox](https://zerox.ai/)를 통해 PDF 파일을 전처리하는 방법을 알아봅니다
# 

# %% [markdown]
# ## 환경설정
# 
# - RAG 파이프라인을 위해 필요한 패키지들을 설치합니다
# - 최신 버전을 설치해도 정상적으로 동작해야 하지만, 버전 명시가 필요하다면 `requirements.txt`를 참고해주세요

# %%
%pip install -qU pypdf langchain-community langchain-text-splitters

# %% [markdown]
# - `PyPDFLoader`를 사용해 전처리된 데이터를 확인합니다

# %%
from langchain_community.document_loaders import PyPDFLoader

pdf_file_path = "./documents/income_tax.pdf"
loader = PyPDFLoader(pdf_file_path)
pages = []
async for page in loader.alazy_load():
    pages.append(page)

# %%
pages[35]

# %% [markdown]
# - 데이터 전처리를 위한 [py-zerox](https://www.piwheels.org/project/py-zerox/) 패키지를 설치합니다

# %%
%pip install -q py-zerox

# %%
from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# - 노트북에서 `asyncio`를 사용하기 위해 `nest_asyncio`를 설치합니다

# %%
%pip install -q nest_asyncio

# %%
import nest_asyncio

nest_asyncio.apply()

# %% [markdown]
# - `py-zerox`를 통해 pdf파일을 전처리합니다
# - 강의에서는 `OpenAI`를 사용하지만, 아래 예제는 `AzureOpenAI`를 사용합니다

# %%
from pyzerox import zerox
import os
import json
import asyncio

### 모델 설정 (Vision 모델만 사용) 참고: https://docs.litellm.ai/docs/providers ###

## 일부 모델에 필요할 수 있는 추가 모델 kwargs의 자리 표시자
kwargs = {}

## Vision 모델에 사용할 시스템 프롬프트
custom_system_prompt = None

model = "azure/gpt-4o-2024-11-20"
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = "2024-08-01-preview"  # "2023-05-15"


# 메인 비동기 진입점을 정의합니다
async def main():
    file_path = "./documents/income_tax.pdf"  ## 로컬 파일 경로 및 파일 URL 지원

    ## 일부 페이지 또는 전체 페이지를 처리
    select_pages = (
        None  ## 전체는 None, 특정 페이지는 int 또는 list(int) 페이지 번호 (1부터 시작)
    )

    output_dir = "./documents"  ## 통합된 마크다운 파일을 저장할 디렉토리
    result = await zerox(
        file_path=file_path,
        model=model,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
        select_pages=select_pages,
        **kwargs
    )
    return result


# 메인 함수를 실행합니다:
result = asyncio.run(main())

# 마크다운 결과를 출력합니다
print(result)

# %% [markdown]
# - zerox를 활용한 전처리 후 생성된 마크다운 파일을 LangGraph에서 활용하기 위해 [unstructured](https://unstructured.io/) 패키지를 설치합니다
# - `UnstructuredMarkdownLoader`를 사용해 전처리된 데이터를 확인합니다
#     - `loader`활용 시 테이블 구조가 사라지는 것을 확인할 수 있습니다

# %%
%pip install -q "unstructured[md]" nltk

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n"]
)

# %%
from langchain_community.document_loaders import UnstructuredMarkdownLoader

markdown_path = "./documents/income_tax.md"
loader = UnstructuredMarkdownLoader(markdown_path)
document_list = loader.load_and_split(text_splitter)

# %%
document_list[43]

# %% [markdown]
# - 마크다운 테이블을 활용하기 위해 `.md` -> `.txt`로 변환합니다

# %%
%pip install -q markdown html2text beautifulsoup4

# %%
import markdown
from bs4 import BeautifulSoup

text_path = "./documents/income_tax.txt"

# 마크다운 파일을 읽어옵니다
with open(markdown_path, "r", encoding="utf-8") as md_file:
    md_content = md_file.read()

# 마크다운 콘텐츠를 HTML로 변환합니다
html_content = markdown.markdown(md_content)

# HTML 콘텐츠를 파싱하여 텍스트만 추출합니다
soup = BeautifulSoup(html_content, "html.parser")
text_content = soup.get_text()

# 추출한 텍스트를 텍스트 파일로 저장합니다
with open(text_path, "w", encoding="utf-8") as txt_file:
    txt_file.write(text_content)

print("Markdown converted to plain text successfully!")

# %% [markdown]
# - `TextLoader`를 사용해 전처리된 데이터를 확인합니다

# %%
from langchain_community.document_loaders import TextLoader

loader = TextLoader(text_path)
document_list = loader.load_and_split(text_splitter)

# %%
document_list[39]

# %% [markdown]
# - 전처리된 데이터를 벡터화하기 위해 [Chroma](https://docs.trychroma.com/getting-started)를 활용합니다
# - LangChain과의 호환을 위해 [langchain-chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/)를 설치합니다

# %%
%pip install -q langchain-chroma

# %%
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# %%
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=document_list,
    embedding=embeddings,
    collection_name="income_tax_collection",
    persist_directory="./income_tax_collection",
)

# %%
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
query = "연봉 5천만원 직장인의 소득세는?"

# %%
retriever.invoke(query)

# %% [markdown]
# - `state`를 선언하고 에이전트를 생성합니다
# - 2.1강에서 진행한 것과 다르게 `messages` 커스텀 변수들을 선언합니다
#     - `query`는 사용자의 질문을 저장하는 용도로 사용합니다
#     - `context`는 벡터 스토어에서 추출한 데이터를 저장하는 용도로 사용합니다
#     - `answer`는 최종 응답을 저장하는 용도로 사용합니다

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

# %%
from langgraph.graph import StateGraph

graph_builder = StateGraph(AgentState)

# %% [markdown]
# - `retrieve` 노드는 사용자의 질문을 받아 벡터 스토어에서 추출한 데이터를 반환합니다

# %%
def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.
    """
    query = state["query"]  # state에서 사용자의 질문을 추출합니다.
    docs = retriever.invoke(query)  # 질문과 관련된 문서를 검색합니다.
    return {"context": docs}  # 검색된 문서를 포함한 state를 반환합니다.

# %% [markdown]
# - `LangChain`의 `hub`를 통해 미리 정의된 RAG 프롬프트를 활용합니다
#     - `hub`에는 이미 검증된 프롬프트들이 많기 때문에 프로젝트 진행 시 좋은 시작점이 됩니다
#     - `hub`에서 프롬프트를 찾아보고, 동작을 확인한 후 커스텀 하는 것을 권장합니다

# %%
from langchain import hub
from langchain_openai import ChatOpenAI

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-4o")

# %%
def generate(state: AgentState) -> AgentState:
    """
    사용자의 질문과 검색된 문서를 기반으로 응답을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문과 검색된 문서를 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 응답이 추가된 state를 반환합니다.
    """
    context = state["context"]  # state에서 검색된 문서를 추출합니다.
    query = state["query"]  # state에서 사용자의 질문을 추출합니다.
    rag_chain = prompt | llm  # RAG 프롬프트와 LLM을 연결하여 체인을 만듭니다.
    response = rag_chain.invoke(
        {"question": query, "context": context}
    )  # 질문과 문맥을 사용하여 응답을 생성합니다.
    return {"answer": response}  # 생성된 응답을 포함한 state를 반환합니다.

# %% [markdown]
# - `node`를 추가하고 `edge`로 연결합니다

# %%
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

# %%
graph = graph_builder.compile()

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# - 병렬처리나 `conditional_edge`가 없는 경우 `add_sequence()`를 통해 순차적으로 동작하는 그래프를 생성할 수 있습니다

# %%
sequence_graph_builder = StateGraph(AgentState).add_sequence([retrieve, generate])

# %%
sequence_graph_builder.add_edge(START, "retrieve")
sequence_graph_builder.add_edge("generate", END)

# %%
sequence_graph = sequence_graph_builder.compile()

# %%
display(Image(sequence_graph.get_graph().draw_mermaid_png()))

# %%
initial_state = {"query": query}
graph.invoke(initial_state)

# %%



