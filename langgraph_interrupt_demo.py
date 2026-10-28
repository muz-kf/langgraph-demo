"""
Dynamic routing + human-in-the-loop interrupt with LangGraph.

    pip install -U langgraph langchain-core langchain-openai
    python langgraph_interrupt_demo.py

LLM integration via OpenRouter (https://openrouter.ai, OpenAI-compatible API).
Set the OPENROUTER_API_KEY environment variable to your OpenRouter API key.
"""
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()  # load OPENROUTER_API_KEY from .env file if present

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------- state
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # reducer: appends, never overwrites
    intent: str
    pending_action: dict
    result: str


# ---------------------------------------------------------------- llm
class Intent(BaseModel):
    intent: Literal["weather", "stocks", "transfer", "chat"] = Field(
        description="The user intent: weather, stocks, transfer, or chat."
    )


def _get_llm() -> ChatOpenAI:
    """Lazy-initialize the OpenRouter LLM; raises if OPENROUTER_API_KEY is missing."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Get a key at https://openrouter.ai/keys"
        )
    return ChatOpenAI(
        model="openrouter/free",  # any OpenRouter model ID works with the OpenAI-compatible base URL
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# NOTE: intent_llm is lazy; call _get_intent_llm() when you need it.
def _get_intent_llm():
    return _get_llm().with_structured_output(Intent)


# ---------------------------------------------------------------- nodes
def guardrails(state: State) -> Command[Literal["router", "refuse"]]:
    text = state["messages"][-1].content.lower()
    return Command(goto="refuse" if "hack" in text else "router")


def refuse(state: State):
    return {"messages": [AIMessage("Sorry, I can't help with that.")]}


def router(state: State):
    # real app: intent = llm.with_structured_output(Intent).invoke(...)
    response = _get_intent_llm().invoke(state["messages"])
    return {"intent": response.intent}


def route_by_intent(state: State) -> str:
    # used by add_conditional_edges: returns a key of the mapping below
    return state["intent"]


def weather_node(state: State):
    return {"result": "Chennai: 31C, humid (stub data)"}


def stock_node(state: State):
    return {"result": "TSLA: 250.00 (stub data)"}


def chat_node(state: State):
    return {"result": "Hi! Ask me about weather, stocks, or transfers."}


def plan_transfer(state: State):
    # real app: the LLM produces this tool call
    action = {"tool": "transfer_money", "args": {"to": "Alex", "amount": 500}}
    return {"pending_action": action}


def approval_node(state: State) -> Command[Literal["execute_tool", "reject_tool"]]:
    # 1st run  : interrupt() saves state, pauses the graph, and surfaces this payload
    # resume   : the node re-runs from the top; interrupt() now RETURNS the resume value
    # NOTE: keep code above interrupt() free of side effects (it runs twice).
    answer = interrupt({
        "question": "Approve this action?",
        "action": state["pending_action"],
    })
    if answer == "yes":
        return Command(goto="execute_tool")
    return Command(goto="reject_tool")


def execute_tool(state: State):
    a = state["pending_action"]["args"]
    return {"result": f"Transferred {a['amount']} to {a['to']}."}


def reject_tool(state: State):
    return {"result": "Okay, I cancelled the transfer."}


def respond(state: State):
    return {"messages": [AIMessage(state["result"])]}


# ---------------------------------------------------------------- wiring
b = StateGraph(State)
for name, fn in [
    ("guardrails", guardrails), ("refuse", refuse), ("router", router),
    ("weather_node", weather_node), ("stock_node", stock_node),
    ("chat_node", chat_node), ("plan_transfer", plan_transfer),
    ("approval_node", approval_node), ("execute_tool", execute_tool),
    ("reject_tool", reject_tool), ("respond", respond),
]:
    b.add_node(name, fn)

b.add_edge(START, "guardrails")            # guardrails -> router/refuse via Command(goto=)
b.add_conditional_edges("router", route_by_intent, {
    "weather":  "weather_node",
    "stocks":   "stock_node",
    "transfer": "plan_transfer",
    "chat":     "chat_node",
})
b.add_edge("plan_transfer", "approval_node")   # approval_node -> execute/reject via Command(goto=)
for n in ["weather_node", "stock_node", "chat_node", "execute_tool", "reject_tool"]:
    b.add_edge(n, "respond")
b.add_edge("respond", END)
b.add_edge("refuse", END)

# a checkpointer is REQUIRED for interrupt(); use PostgresSaver etc. in production
graph = b.compile(checkpointer=MemorySaver())


def _print_bot(result: dict) -> None:
    """Print the latest bot message from a state snapshot."""
    messages = result.get("messages", [])
    if messages and getattr(messages[-1], "content", None):
        print(f"Bot: {messages[-1].content}\n")


def _resolve_interrupt(config: dict) -> bool:
    """If the graph is paused at a human-in-the-loop interrupt, ask for approval
    and resume. Returns True if an interrupt was handled."""
    state = graph.get_state(config)
    interrupts = getattr(state, "interrupts", None) or {}
    if not interrupts:
        return False

    # interrupts: dict of node_name -> list of Interrupt objects
    # Each Interrupt has a .value attribute which is the dict passed to interrupt()
    first_node_interrupts = next(iter(interrupts.values()), None)
    if not first_node_interrupts:
        return False

    interrupt_obj = first_node_interrupts[0] if isinstance(first_node_interrupts, list) else first_node_interrupts
    value = getattr(interrupt_obj, "value", interrupt_obj)

    if isinstance(value, dict):
        question = value.get("question", "Approval required")
        action = value.get("action")
    else:
        question, action = "Approval required", value

    print(f"\n[System] {question}")
    if action:
        print(f"  Action: {action}")
    answer = input("Approve? (yes/no): ").strip().lower()
    try:
        result = graph.invoke(Command(resume=answer), config)
        _print_bot(result)
    except Exception as e:
        print(f"\n[Error] While resuming: {type(e).__name__}: {e}\n")
    return True


# ---------------------------------------------------------------- interactive demo
if __name__ == "__main__":
    print("=" * 60)
    print("  LangGraph AI Assistant  (LLM via OpenRouter)")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the session.\n")

    config = {"configurable": {"thread_id": "interactive-session"}}

    while True:
        # 0. Resolve any pending interrupt first
        if _resolve_interrupt(config):
            continue

        # 1. Read user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        if not user_input:
            continue

        # 2. Run the graph
        try:
            result = graph.invoke({"messages": [HumanMessage(user_input)]}, config)
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}\n")
            continue

        # 3. If the graph paused at an interrupt, the loop top will handle it
        if result.get("__interrupt__"):
            continue

        # 4. Print the bot response
        _print_bot(result)
