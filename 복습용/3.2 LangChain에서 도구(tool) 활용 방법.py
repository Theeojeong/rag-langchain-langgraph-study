# %% [markdown]
# # 3.2 LangChain에서 도구(tool) 활용 방법

# %% [markdown]
# - LangChain에서 도구(tool)을 활용하는 방법을 알아봅니다
# - 이 강의에서는 도구를 활용하는 방법을 중점적으로 다루며, 도구를 활용한 에이전트 개발 방법은 3.3 강의에서 다룹니다
#     - LangGraph에서의 도구 활용 방법은 LangChain의 문법을 따르니 주의깊게 봐주세요

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

# %% [markdown]
# - [tool decorator](https://python.langchain.com/docs/how_to/custom_tools/#tool-decorator)를 사용하면 쉽게 도구를 만들 수 있습니다

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
# - LLM을 호출했을 때와 도구를 사용했을 때의 차이를 알아봅니다

# %%
query = "3 곱하기 5는?"

# %%
small_llm.invoke(query)

# %% [markdown]
# - 도구 리스트는 LLM에 해당하는 `BaseModel` 클래스에 `bind_tools` 메서드를 통해 전달합니다

# %%
llm_with_tools = small_llm.bind_tools([add, multiply])

# %% [markdown]
# - `AIMessage`의 `additional_kwargs` 속성은 `tool_calls`를 포함합니다
# - `tool_calls`는 도구를 호출하는 메시지를 포함합니다

# %%
result = llm_with_tools.invoke(query)
result

# %% [markdown]
# - 여기서 `tool_calls`의 형태를 기억해두시면 남은 강의를 이해하시는데 도움이 됩니다

# %%
result.tool_calls

# %%
from typing import Sequence

from langchain_core.messages import AnyMessage, HumanMessage

human_message = HumanMessage(query)
message_list: Sequence[AnyMessage] = [human_message]

# %% [markdown]
# - `tool_calls` 속성은 도구를 호출하는 메시지를 포함합니다
# - `tool_calls`를 가진 `AIMessage`의 형태를 기억해두시면 남은 강의를 이해하시는데 도움이 됩니다

# %%
ai_message = llm_with_tools.invoke(message_list)
ai_message

# %%
ai_message.tool_calls

# %%
message_list.append(ai_message)

# %% [markdown]
# - `AIMessage`의 `tool_calls`를 활용해서 도구를 직접 호출할 수도 있습니다

# %%
tool_message = multiply.invoke(ai_message.tool_calls[0])

# %%
tool_message

# %% [markdown]
# - 하지만 에이전트의 경우 도구를 직접 호출하는 것이 아니라 도구를 호출하는 메시지를 만들어서 전달합니다

# %%
message_list.append(tool_message)

# %%
llm_with_tools.invoke(message_list)

# %% [markdown]
# - `message_list`의 순서를 기억해두시면 남은 강의를 이해하시는데 도움이 됩니다

# %%
message_list


