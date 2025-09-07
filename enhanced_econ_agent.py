from __future__ import annotations

"""
Enhanced economic analyst multi-agent graph based on real_agent_for_analyst.py

This module does NOT modify existing files. It recreates a compatible graph
with added tools and nodes (technical analysis, macro context, valuation)
and exposes build_enhanced_graph() for use from apps (e.g., Streamlit).
"""

from typing import Literal, Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np
import yfinance as yf

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


def _get_llms() -> tuple[AzureChatOpenAI, AzureChatOpenAI]:
    """Construct primary and small LLMs identical to the reference file."""
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-2024-11-20",
        api_version="2024-08-01-preview",
    )
    small_llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini-2024-07-18",
        api_version="2024-08-01-preview",
    )
    return llm, small_llm


# -----------------------------
# Tools (added without touching existing code)
# -----------------------------


@tool
def get_stock_history(
    ticker: str, period: str = "6mo", interval: str = "1d"
) -> Dict[str, Any]:
    """Return OHLCV history for the given ticker and period."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return {"error": f"No data for {ticker}"}
    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "rows": len(df),
        "data": df.reset_index().to_dict(orient="list"),
    }


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


@tool
def compute_technical_indicators(
    ticker: str, period: str = "6mo", interval: str = "1d"
) -> Dict[str, Any]:
    """Compute SMA, EMA, RSI, and MACD for a ticker. Returns a concise summary."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return {"error": f"No data for {ticker}"}

    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    rsi14 = _rsi(close, 14)

    latest = {
        "price": float(close.iloc[-1]),
        "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None,
        "sma50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else None,
        "signal": float(signal.iloc[-1]) if not np.isnan(signal.iloc[-1]) else None,
        "rsi14": float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else None,
    }

    crossover = None
    if latest["sma20"] and latest["sma50"]:
        prev = sma20.iloc[-2] - sma50.iloc[-2]
        now = sma20.iloc[-1] - sma50.iloc[-1]
        if prev < 0 and now > 0:
            crossover = "golden_cross"
        elif prev > 0 and now < 0:
            crossover = "death_cross"

    momentum = None
    if latest["rsi14"] is not None:
        if latest["rsi14"] > 70:
            momentum = "overbought"
        elif latest["rsi14"] < 30:
            momentum = "oversold"
        else:
            momentum = "neutral"

    macd_signal = None
    if latest["macd"] is not None and latest["signal"] is not None:
        macd_signal = (
            "bullish" if latest["macd"] > latest["signal"] else "bearish"
        )

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "latest": latest,
        "signals": {
            "sma_crossover": crossover,
            "rsi_momentum": momentum,
            "macd_signal": macd_signal,
        },
    }


@tool
def market_macro_snapshot(
    index_ticker: str = "^GSPC", vix_ticker: str = "^VIX",
    yield_ticker: str = "^TNX", period: str = "6mo"
) -> Dict[str, Any]:
    """Return a macro snapshot using broad index (^GSPC), VIX, and 10Y yield (^TNX)."""
    def last_change(ticker: str) -> Dict[str, Any]:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
        if df.empty:
            return {"ticker": ticker, "error": "no_data"}
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        close = float(last["Close"])
        pct = float(((last["Close"] - prev["Close"]) / prev["Close"]) * 100.0) if prev["Close"] else 0.0
        return {"ticker": ticker, "close": close, "daily_pct": pct}

    return {
        "index": last_change(index_ticker),
        "vix": last_change(vix_ticker),
        "us10y": last_change(yield_ticker),
        "period": period,
    }


@tool
def company_profile_tool(ticker: str) -> Dict[str, Any]:
    """Return basic company profile (sector, industry, marketCap, beta)."""
    tkr = yf.Ticker(ticker)
    info = {}
    try:
        info = tkr.fast_info.__dict__ if hasattr(tkr, "fast_info") else {}
    except Exception:
        pass
    fallback = {}
    try:
        fallback = tkr.info
    except Exception:
        # yfinance may restrict info; ignore failures
        pass

    wanted = {
        k: fallback.get(k)
        for k in [
            "longName",
            "sector",
            "industry",
            "marketCap",
            "beta",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "enterpriseToEbitda",
        ]
    }

    return {"ticker": ticker, "fast_info": info, "profile": wanted}


# -----------------------------
# Agents / Nodes
# -----------------------------


def _market_research_agent(llm: AzureChatOpenAI):
    # Prefer YahooFinanceNewsTool if available; Polygon tools if configured
    tools = []
    try:
        from langchain_community.tools.yahoo_finance_news import (
            YahooFinanceNewsTool,
        )

        tools.append(YahooFinanceNewsTool())
    except Exception:
        pass

    try:
        from langchain_community.agent_toolkits.polygon.toolkit import (
            PolygonToolkit,
        )
        from langchain_community.utilities.polygon import PolygonAPIWrapper

        polygon = PolygonAPIWrapper()
        toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon)
        tools.extend(toolkit.get_tools())
    except Exception:
        # Polygon might not be configured; continue without it
        pass

    return create_react_agent(
        llm,
        tools=tools,
        state_modifier="You are a market researcher. Provide facts only, cite sources when possible.",
    )


def _stock_research_agent(llm: AzureChatOpenAI):
    return create_react_agent(
        llm,
        tools=[get_stock_history],
        state_modifier="You are a stock researcher. Provide factual recent price/volume context.",
    )


def _company_research_agent(llm: AzureChatOpenAI):
    return create_react_agent(
        llm,
        tools=[company_profile_tool],
        state_modifier="You are a company researcher. Provide concise fundamentals only.",
    )


def _technical_agent(llm: AzureChatOpenAI):
    return create_react_agent(
        llm,
        tools=[compute_technical_indicators],
        state_modifier=(
            "You are a technical analyst. Compute indicators and summarize signals."
            " No opinions; describe momentum, crossovers, and risk neutrally."
        ),
    )


def _macro_agent(llm: AzureChatOpenAI):
    return create_react_agent(
        llm,
        tools=[market_macro_snapshot],
        state_modifier=(
            "You are a macro strategist. Summarize index, volatility (VIX), and 10Y yield"
            " regime neutrally and briefly."
        ),
    )


def market_research_node(agent, state: MessagesState) -> Command[Literal["supervisor"]]:
    result = agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="market_research"
                )
            ]
        },
        goto="supervisor",
    )


def stock_research_node(agent, state: MessagesState) -> Command[Literal["supervisor"]]:
    result = agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="stock_research"
                )
            ]
        },
        goto="supervisor",
    )


def company_research_node(agent, state: MessagesState) -> Command[Literal["supervisor"]]:
    result = agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="company_research"
                )
            ]
        },
        goto="supervisor",
    )


def technical_analysis_node(agent, state: MessagesState) -> Command[Literal["supervisor"]]:
    result = agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="technical_analysis"
                )
            ]
        },
        goto="supervisor",
    )


def macro_context_node(agent, state: MessagesState) -> Command[Literal["supervisor"]]:
    result = agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="macro_context"
                )
            ]
        },
        goto="supervisor",
    )


def _analyst_chain(llm: AzureChatOpenAI):
    prompt = PromptTemplate.from_template(
        (
            "You are a professional stock market analyst. Using the multi-agent findings\n"
            "(market_research, stock_research, company_research, technical_analysis, macro_context)\n"
            "provide a final recommendation for the given ticker.\n\n"
            "Requirements:\n"
            "- Output sections: Summary, Key Drivers, Risks, Time Horizon, Recommendation.\n"
            "- Recommendation must be one of: BUY, HOLD, or SELL.\n"
            "- Keep it factual and concise (10-15 lines).\n\n"
            "Information:\n{messages}"
        )
    )
    return prompt | llm


def analyst_node(analyst_chain, state: MessagesState):
    result = analyst_chain.invoke({"messages": state["messages"][1:]})
    return {"messages": [result]}


# -----------------------------
# Supervisor and Graph
# -----------------------------


def _supervisor_node(llm: AzureChatOpenAI, members: list[str]):
    options = members + ["FINISH"]

    system_prompt = (
        "You are a supervisor managing workers: "
        f"{members}. Given the user request and prior messages, "
        "respond with the worker to act next. When done, respond FINISH."
    )

    from typing_extensions import TypedDict

    class Router(TypedDict):
        next: Literal[*options]

    def node(state: MessagesState) -> Command[Literal[*members, "analyst"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = "analyst"
        return Command(goto=goto)

    return node


def build_enhanced_graph():
    """Build and return the enhanced multi-agent graph."""
    llm, small_llm = _get_llms()

    # Agents
    market_agent = _market_research_agent(llm)
    stock_agent = _stock_research_agent(llm)
    company_agent = _company_research_agent(llm)
    tech_agent = _technical_agent(llm)
    macro_agent = _macro_agent(llm)

    # Nodes bound with their agents
    def _market_node(state: MessagesState):
        return market_research_node(market_agent, state)

    def _stock_node(state: MessagesState):
        return stock_research_node(stock_agent, state)

    def _company_node(state: MessagesState):
        return company_research_node(company_agent, state)

    def _tech_node(state: MessagesState):
        return technical_analysis_node(tech_agent, state)

    def _macro_node(state: MessagesState):
        return macro_context_node(macro_agent, state)

    # Analyst
    analyst_chain = _analyst_chain(llm)

    # Graph
    members = [
        "market_research",
        "stock_research",
        "company_research",
        "technical_analysis",
        "macro_context",
    ]
    supervisor_node = _supervisor_node(llm, members)

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("market_research", _market_node)
    graph_builder.add_node("stock_research", _stock_node)
    graph_builder.add_node("company_research", _company_node)
    graph_builder.add_node("technical_analysis", _tech_node)
    graph_builder.add_node("macro_context", _macro_node)
    graph_builder.add_node("analyst", lambda state: analyst_node(analyst_chain, state))

    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_edge("analyst", END)
    return graph_builder.compile()


def run_once(question: str) -> str:
    """Utility: run the enhanced graph once for a plain question."""
    graph = build_enhanced_graph()
    last = None
    for chunk in graph.stream({"messages": [("user", question)]}, stream_mode="values"):
        last = chunk
    if not last:
            return ""
    return str(last["messages"][-1].content)

