# AI Agent Study Project (AI_Agent)

이 저장소는 LangChain·LangGraph·Streamlit을 중심으로 “에이전트” 개념을 공부하고 실습하기 위한 개인 학습용 프로젝트입니다. 단일 에이전트부터 ReAct 기반 도구 호출, 멀티 에이전트(슈퍼바이저-워커 구조), RAG(검색 증강), 간단한 UI까지 단계적으로 시도합니다.

본 프로젝트의 결과물은 교육/연구 목적이며, 특히 금융 관련 예시는 투자 조언이 아닙니다.

## 주요 학습 주제

- 에이전트 기본기: LangChain 도구 호출(ReAct)과 상태 관리
- 멀티 에이전트: LangGraph 기반 Supervisor → Worker 라우팅
- RAG: 소득세법 문서 기반 질의응답(대화 이력 반영, Few-shot)
- UI: Streamlit 채팅 인터페이스 및 대화 저장·복원

## 구성 요소

- 경제 애널리스트 멀티 에이전트
  - 원본 예제: `real_agent_for_analyst.py`
  - 고도화 버전(기존 파일 미변경): `enhanced_econ_agent.py`
    - 추가 도구: 기술적 지표(RSI/MACD/SMA), 거시 스냅샷(^GSPC/^VIX/^TNX), 기업 프로필, 시세 히스토리
  - Streamlit UI: `streamlit_analyst_app.py` (채팅/에이전트 트레이스, 대화 ID별 저장)

- RAG(소득세법) 챗봇
  - 앱: `chat.py`
  - 체인/리트리버: `llm.py`, 프롬프트 예시: `config.py`

- 노트북 예제
  - `practice.ipynb`
  - `inflearn-langgraph-agent/3.7 찐 Multi-Agent System (feat. create_react_agent).ipynb`

## 설치

사전 요구사항: Python 3.10+ 권장, pip, 가상환경 사용 권장

1) 저장소 클론

```bash
git clone <this-repo>
cd AI_Agent
```

2) 가상환경 생성/활성화

- Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

- macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) 패키지 설치

```bash
pip install -r requirements.txt
```

참고: `requirements.txt` 인코딩 문제로 설치가 실패한다면, 에디터에서 UTF-8로 저장하거나 필요한 패키지만 개별 설치하세요(예: langchain, langgraph, streamlit, yfinance 등).

## 환경 변수(.env)

프로젝트는 `.env`를 로드합니다. 사용 중인 기능에 따라 다음을 설정하세요.

- Azure OpenAI (경제 애널리스트 에이전트)
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - 모델 배포명은 코드에 하드코딩되어 있으므로(예: `gpt-4o-2024-11-20`) 포털에서 동일한 이름으로 배포되어 있어야 합니다.

- Polygon(선택): `POLYGON_API_KEY` (설정 시 Polygon 도구 사용)

## 실행 방법

- 고도화 경제 애널리스트 Streamlit 앱 실행

```bash
streamlit run streamlit_analyst_app.py
```

사이드바에서 Ticker/포커스(시장/기술/거시/기업)를 설정하고 질문을 입력하세요. 대화는 `conversations/<대화ID>.json`에 자동 저장/복원됩니다.

- 소득세법 RAG 챗봇 실행

```bash
streamlit run chat.py
```

- 파이썬에서 단일 실행(예시)

```python
from enhanced_econ_agent import run_once
print(run_once("Would you invest in Snowflake? Ticker: SNOW"))
```

## 아키텍처 개요(경제 애널리스트)

- Supervisor가 `market_research`, `stock_research`, `company_research`, `technical_analysis`, `macro_context`를 순차/선택적으로 호출 → `analyst` 노드가 최종 요약 및 추천(BUY/HOLD/SELL) 생성
- Streamlit UI는 에이전트 스트림을 받아 중간 워커 결과를 트레이스로 보여주고, 최종 응답을 챗 메시지로 출력

## 면책 조항

- 본 저장소의 코드는 교육/연구 목적입니다.
- 제공되는 분석·추천은 실제 투자 조언이 아니며, 정확성/적시성을 보장하지 않습니다.

## 기여

실습/학습 관점의 개선 아이디어, 버그 리포트, 문서 보완 PR을 환영합니다.
