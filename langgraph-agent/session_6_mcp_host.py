#!/usr/bin/env python3
"""
MCP-based Trading Bot with Web UI Human-in-the-Loop System (STDIO Host)

구성
- MCP Server(trade): session_6_mcp_server.py 를 STDIO로 자식 프로세스로 실행
- LangGraph Agent: ReAct prebuilt + MCP tools + HITL(request_human_approval)
- Web UI(FastAPI): 승인(HITL) 인터페이스 + 간단 API (POST /api/trade)
- In-memory 상태: 승인요청/승인결과/WebSocket 연결
"""

import os
import json
import sys
import time
import uuid
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any, Literal
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient

# =============================================================================
# GLOBAL STATE (In-Memory)
# =============================================================================

pending_approvals: Dict[str, Dict[str, Any]] = {}
completed_approvals: Dict[str, Dict[str, Any]] = {}
active_connections: Dict[str, WebSocket] = {}

broadcast_q: SimpleQueue[dict] = SimpleQueue()

agent_ready = asyncio.Event()
global_agent = None
global_mcp_client: MultiServerMCPClient | None = None

BASE_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = str(BASE_DIR / "session_6_mcp_server.py")

# =============================================================================
# HITL TOOL
# =============================================================================


@tool("request_human_approval")
async def request_human_approval(ticker: str,
                                 action: Literal["BUY", "SELL"],
                                 reason: str,
                                 market_data: str = "") -> str:
    """
    BUY/SELL 의사결정에 대한 인간 승인을 요청하고 5분간 대기한다.
    """
    try:
        if action not in ("BUY", "SELL"):
            return json.dumps(
                {
                    "approved": False,
                    "error": "승인은 BUY/SELL 액션에만 필요합니다",
                    "timestamp": time.time()
                },
                ensure_ascii=False)

        request_id = f"approval_{ticker}_{action}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        approval_request = {
            "request_id": request_id,
            "ticker": ticker,
            "action": action,
            "reason": reason,
            "market_data": market_data,
            "timestamp": time.time(),
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        pending_approvals[request_id] = approval_request

        broadcast_q.put({"type": "approval_request", "data": approval_request})

        print(f"🔔 승인 요청 생성: {ticker} {action} (ID: {request_id})")

        max_wait_time = 300  # 5분
        check_interval = 2
        waited = 0
        while waited < max_wait_time:
            if request_id not in pending_approvals:
                break
            await asyncio.sleep(check_interval)
            waited += check_interval

        if request_id in pending_approvals:
            del pending_approvals[request_id]
            return json.dumps(
                {
                    "approved": False,
                    "error": "승인 요청 시간 초과 (5분)",
                    "timeout": True,
                    "timestamp": time.time()
                },
                ensure_ascii=False)

        result = completed_approvals.pop(request_id, None)
        if result:
            return json.dumps(result, ensure_ascii=False)

        return json.dumps(
            {
                "approved": False,
                "error": "승인 상태를 확인할 수 없습니다",
                "timestamp": time.time()
            },
            ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {
                "approved": False,
                "error": f"승인 요청 중 오류 발생: {str(e)}",
                "timestamp": time.time()
            },
            ensure_ascii=False)


# =============================================================================
# WEB UI & API
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    async def broadcast_loop():
        while True:
            sent_any = False
            try:
                while True:
                    msg = broadcast_q.get_nowait()
                    await broadcast_to_all_connections(msg)
                    sent_any = True
            except Exception:
                # 큐가 비었거나 기타 예외 시 잠깐 쉼
                await asyncio.sleep(0.2 if not sent_any else 0)

    app.state.broadcast_task = asyncio.create_task(broadcast_loop())
    try:
        yield
    finally:
        # shutdown
        task = getattr(app.state, "broadcast_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="Trading Bot HITL Interface (STDIO)",
    lifespan=lifespan,
)


def load_html_template() -> str:
    """업로드된 ui_template.html을 우선 읽고, 없으면 로컬 상대경로로 시도."""
    candidates = [
        str(BASE_DIR / "ui_template.html"),
        "/mnt/data/ui_template.html",
    ]
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            continue
    return "<h1>UI Template not found</h1>"


@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    return load_html_template()


@app.post("/api/trade")
async def request_trade(request: Request):
    try:
        data = await request.json()
        ticker = (data.get("ticker") or "").upper()

        await asyncio.wait_for(agent_ready.wait(), timeout=20)
        if not ticker:
            return {"status": "error", "error": "Ticker symbol is required"}

        if global_agent is None:
            return {"status": "error", "error": "Agent not initialized"}

        print(f"\n🚀 API 거래 요청 수신: {ticker}")
        response = await run_trading_demo(global_agent, ticker)
        tail = response['messages'][-1].content if response else "응답 없음"
        return {
            "status": "success",
            "ticker": ticker,
            "message": "거래 분석 완료",
            "response": tail
        }

    except asyncio.TimeoutError:
        return {"status": "error", "error": "Agent not initialized (timeout)"}
    except Exception as e:
        print(f"❌ API 거래 요청 실패: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/api/stocks-by-mcp")
async def get_stock_list():
    try:
        blobs = await global_mcp_client.get_resources("trade",
                                                      uris="trade://stocks")
        if not blobs:
            return {"stocks": [], "error": "리소스가 비어 있습니다."}

        blob = blobs[0]

        # 문자열이 있으면 우선 사용
        try:
            text = blob.as_string()
            return json.loads(text)
        # except Exception:
        #     pass
        #
        # # 문자열이 없으면 바이트로 파싱
        # try:
        #     data = blob.as_bytes()
        #     return json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            # 그래도 안 되면 실패 처리
            return {"stocks": [], "error": "리소스 파싱 실패"}

    except Exception as e:
        print(f"❌ MCP stocks 리소스 조회 실패: {e}")
        return {"stocks": [], "error": "MCP 서버에서 종목 데이터를 가져올 수 없습니다."}


# @app.get("/api/stocks")
# async def get_stock_list():
#     # Tech giants
#     tech_stocks = [
#         {
#             "ticker": "AAPL",
#             "name": "Apple Inc.",
#             "category": "Technology"
#         },
#         {
#             "ticker": "MSFT",
#             "name": "Microsoft Corporation",
#             "category": "Technology"
#         },
#         {
#             "ticker": "GOOGL",
#             "name": "Alphabet Inc. (Class A)",
#             "category": "Technology"
#         },
#         {
#             "ticker": "GOOG",
#             "name": "Alphabet Inc. (Class C)",
#             "category": "Technology"
#         },
#         {
#             "ticker": "AMZN",
#             "name": "Amazon.com Inc.",
#             "category": "Technology"
#         },
#         {
#             "ticker": "META",
#             "name": "Meta Platforms Inc.",
#             "category": "Technology"
#         },
#         {
#             "ticker": "TSLA",
#             "name": "Tesla Inc.",
#             "category": "Technology"
#         },
#         {
#             "ticker": "NVDA",
#             "name": "NVIDIA Corporation",
#             "category": "Technology"
#         },
#         {
#             "ticker": "NFLX",
#             "name": "Netflix Inc.",
#             "category": "Technology"
#         },
#         {
#             "ticker": "ADBE",
#             "name": "Adobe Inc.",
#             "category": "Technology"
#         },
#     ]

#     # Financial
#     financial_stocks = [
#         {
#             "ticker": "JPM",
#             "name": "JPMorgan Chase & Co.",
#             "category": "Financial"
#         },
#         {
#             "ticker": "BAC",
#             "name": "Bank of America Corp.",
#             "category": "Financial"
#         },
#         {
#             "ticker": "WFC",
#             "name": "Wells Fargo & Company",
#             "category": "Financial"
#         },
#         {
#             "ticker": "GS",
#             "name": "Goldman Sachs Group Inc.",
#             "category": "Financial"
#         },
#         {
#             "ticker": "MS",
#             "name": "Morgan Stanley",
#             "category": "Financial"
#         },
#         {
#             "ticker": "C",
#             "name": "Citigroup Inc.",
#             "category": "Financial"
#         },
#         {
#             "ticker": "AXP",
#             "name": "American Express Company",
#             "category": "Financial"
#         },
#         {
#             "ticker": "BLK",
#             "name": "BlackRock Inc.",
#             "category": "Financial"
#         },
#         {
#             "ticker": "SCHW",
#             "name": "Charles Schwab Corporation",
#             "category": "Financial"
#         },
#         {
#             "ticker": "USB",
#             "name": "U.S. Bancorp",
#             "category": "Financial"
#         },
#     ]

#     # Healthcare
#     healthcare_stocks = [
#         {
#             "ticker": "JNJ",
#             "name": "Johnson & Johnson",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "PFE",
#             "name": "Pfizer Inc.",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "UNH",
#             "name": "UnitedHealth Group Inc.",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "ABBV",
#             "name": "AbbVie Inc.",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "TMO",
#             "name": "Thermo Fisher Scientific Inc.",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "DHR",
#             "name": "Danaher Corporation",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "BMY",
#             "name": "Bristol Myers Squibb Company",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "MRK",
#             "name": "Merck & Co. Inc.",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "CVS",
#             "name": "CVS Health Corporation",
#             "category": "Healthcare"
#         },
#         {
#             "ticker": "GILD",
#             "name": "Gilead Sciences Inc.",
#             "category": "Healthcare"
#         },
#     ]

#     # Consumer
#     consumer_stocks = [
#         {
#             "ticker": "KO",
#             "name": "Coca-Cola Company",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "PEP",
#             "name": "PepsiCo Inc.",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "WMT",
#             "name": "Walmart Inc.",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "HD",
#             "name": "Home Depot Inc.",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "MCD",
#             "name": "McDonald's Corporation",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "NKE",
#             "name": "Nike Inc.",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "SBUX",
#             "name": "Starbucks Corporation",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "TGT",
#             "name": "Target Corporation",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "LOW",
#             "name": "Lowe's Companies Inc.",
#             "category": "Consumer"
#         },
#         {
#             "ticker": "COST",
#             "name": "Costco Wholesale Corporation",
#             "category": "Consumer"
#         },
#     ]

#     # Industrial
#     industrial_stocks = [
#         {
#             "ticker": "BA",
#             "name": "Boeing Company",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "CAT",
#             "name": "Caterpillar Inc.",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "GE",
#             "name": "General Electric Company",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "MMM",
#             "name": "3M Company",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "HON",
#             "name": "Honeywell International Inc.",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "UPS",
#             "name": "United Parcel Service Inc.",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "FDX",
#             "name": "FedEx Corporation",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "LMT",
#             "name": "Lockheed Martin Corporation",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "RTX",
#             "name": "RTX Corporation",
#             "category": "Industrial"
#         },
#         {
#             "ticker": "NOC",
#             "name": "Northrop Grumman Corporation",
#             "category": "Industrial"
#         },
#     ]

#     # Energy
#     energy_stocks = [
#         {
#             "ticker": "XOM",
#             "name": "Exxon Mobil Corporation",
#             "category": "Energy"
#         },
#         {
#             "ticker": "CVX",
#             "name": "Chevron Corporation",
#             "category": "Energy"
#         },
#         {
#             "ticker": "COP",
#             "name": "ConocoPhillips",
#             "category": "Energy"
#         },
#         {
#             "ticker": "SLB",
#             "name": "Schlumberger Limited",
#             "category": "Energy"
#         },
#         {
#             "ticker": "EOG",
#             "name": "EOG Resources Inc.",
#             "category": "Energy"
#         },
#         {
#             "ticker": "PXD",
#             "name": "Pioneer Natural Resources Company",
#             "category": "Energy"
#         },
#         {
#             "ticker": "KMI",
#             "name": "Kinder Morgan Inc.",
#             "category": "Energy"
#         },
#         {
#             "ticker": "OXY",
#             "name": "Occidental Petroleum Corporation",
#             "category": "Energy"
#         },
#         {
#             "ticker": "VLO",
#             "name": "Valero Energy Corporation",
#             "category": "Energy"
#         },
#         {
#             "ticker": "PSX",
#             "name": "Phillips 66",
#             "category": "Energy"
#         },
#     ]

#     # Korean stocks
#     korean_stocks = [
#         {
#             "ticker": "005930.KS",
#             "name": "Samsung Electronics Co., Ltd.",
#             "category": "Korean Tech"
#         },
#         {
#             "ticker": "018260.KS",
#             "name": "Samsung SDS Co., Ltd.",
#             "category": "Korean Tech"
#         },
#         {
#             "ticker": "000660.KS",
#             "name": "SK Hynix Inc.",
#             "category": "Korean Tech"
#         },
#         {
#             "ticker": "035420.KS",
#             "name": "NAVER Corporation",
#             "category": "Korean Tech"
#         },
#         {
#             "ticker": "207940.KS",
#             "name": "Samsung Biologics Co., Ltd.",
#             "category": "Korean Healthcare"
#         },
#         {
#             "ticker": "051910.KS",
#             "name": "LG Chem Ltd.",
#             "category": "Korean Industrial"
#         },
#     ]

#     # Combine all stocks
#     all_stocks = tech_stocks + financial_stocks + healthcare_stocks + consumer_stocks + industrial_stocks + energy_stocks + korean_stocks

#     return {"stocks": all_stocks}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    conn_id = uuid.uuid4().hex[:8]
    await websocket.accept()
    active_connections[conn_id] = websocket
    print(f"🌐 웹소켓 연결: {conn_id}")

    try:
        await websocket.send_text(
            json.dumps({
                "type": "system",
                "message": "connected"
            }))
        last_ping = time.time()
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(),
                                              timeout=2.0)
                msg = json.loads(data)
                if msg.get("type") == "approval_response":
                    await handle_approval_response(msg)
            except asyncio.TimeoutError:
                if time.time() - last_ping > 20:
                    await websocket.send_text(
                        json.dumps({
                            "type": "ping",
                            "t": time.time()
                        }))
                    last_ping = time.time()
                continue
    except WebSocketDisconnect:
        print(f"🔌 웹소켓 종료: {conn_id}")
    except Exception as e:
        print(f"❌ WebSocket 오류({conn_id}): {e}")
    finally:
        active_connections.pop(conn_id, None)


async def handle_approval_response(message: Dict[str, Any]):
    req_id = message["request_id"]
    approved = bool(message["approved"])

    if req_id in pending_approvals:
        req = pending_approvals[req_id]
        print(
            f"\n✅ 승인 응답: {req['ticker']} {req['action']} - {'승인' if approved else '거부'}"
        )

        if approved:
            token = f"token_{req_id}_{int(time.time())}"
            result = {
                "approved": True,
                "message": "승인 완료",
                "approval_token": token,
                "timestamp": time.time()
            }
            print(f"🎫 승인 토큰: {token}")
        else:
            result = {
                "approved": False,
                "message": "거래가 거부되었습니다",
                "timestamp": time.time()
            }

        completed_approvals[req_id] = result
        del pending_approvals[req_id]

        await broadcast_to_all_connections({
            "type": "approval_processed",
            "request_id": req_id,
            "approved": approved
        })


async def broadcast_to_all_connections(message: Dict[str, Any]):
    if not active_connections:
        return
    payload = json.dumps(message, ensure_ascii=False)
    disconnects = []
    for cid, ws in active_connections.items():
        try:
            await ws.send_text(payload)
        except Exception as e:
            print(f"❌ WS 전송 실패({cid}): {e}")
            disconnects.append(cid)
    for cid in disconnects:
        active_connections.pop(cid, None)


# =============================================================================
# MCP CLIENT + LangGraph Agent (STDIO)
# =============================================================================


async def build_agent() -> tuple[MultiServerMCPClient, any]:
    """
    STDIO로 trade MCP 서버(session_6_mcp_server.py)를 자식 프로세스로 실행하고,
    노출된 MCP tools를 LangGraph ReAct agent에 바인딩한다.
    """
    PYTHON = os.environ.get("PYTHON_EXECUTABLE") or sys.executable

    client = MultiServerMCPClient({
        "trade": {
            "transport": "stdio",
            "command": PYTHON,
            "args": [MCP_SERVER_PATH],
        }
    })

    tools = await client.get_tools()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    memory = MemorySaver()
    agent = create_react_agent(model=model,
                               tools=[request_human_approval] + tools,
                               checkpointer=memory)

    agent_ready.set()
    return client, agent


async def run_trading_demo(agent, ticker: str = "NVDA"):
    msg = f"""
{ticker} 종목에 대해 거래 분석을 수행해주세요.

다음 단계를 따라주세요:
1. analyze_market_trend 도구를 사용해서 {ticker}의 시장 동향을 분석하세요
2. 분석 결과를 바탕으로 BUY/SELL/HOLD 결정을 내리세요
3. BUY 또는 SELL 추천 시에는 request_human_approval 도구로 사용자 승인을 요청하세요
4. 승인 후 execute_trade 도구로 거래를 실행하세요

모든 응답은 한국어로 해주세요.
""".strip()

    config = {"configurable": {"thread_id": f"demo_{int(time.time())}"}}
    res = await agent.ainvoke({"messages": [HumanMessage(content=msg)]},
                              config)
    print("\n📋 에이전트 응답(요약):")
    print(res['messages'][-1].content if res else "")
    return res


# =============================================================================
# MAIN
# =============================================================================


def run_web_server():
    # Replit 호환: PORT 환경변수 우선 사용, 없으면 8080
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


async def main():
    print("🤖 MCP 기반 Trading Bot with HITL (STDIO) 시작")
    print("=" * 60)

    try:
        mcp_client, agent = await build_agent()

        global global_agent, global_mcp_client
        global_agent = agent
        global_mcp_client = mcp_client

        print("\n🎯 시스템 준비 완료!")
        print("💡 사용법: POST /api/trade  {'ticker':'AAPL'}")

        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 종료 요청")
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 시스템 종료됨")


if __name__ == "__main__":
    asyncio.run(main())
