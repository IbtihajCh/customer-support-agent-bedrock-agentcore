"""
Customer Support AI Agent
==========================
An AgentCore-hosted support assistant for the storefront.

Capabilities are deliberately layered, and each layer fails independently:
  * retrieval   - product/policy answers, grounded in a Bedrock Knowledge Base
  * computation - loyalty arithmetic, run in the Code Interpreter sandbox
  * live data   - orders and refunds, reached over MCP through the gateway
  * browsing    - outbound page reads when a customer pastes a URL
  * recall      - cross-session customer facts and preferences

Design stance: the model never sources a fact it could have looked up, and
never performs arithmetic it could have delegated. Every numeric or policy
answer is traceable to a tool result.

Run one turn locally:
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy:
  agentcore configure --entrypoint main.py --name csai
  agentcore deploy
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
import shutil, tempfile
from pathlib import Path
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict, List, Optional
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser

from contextlib import ExitStack
from dataclasses import dataclass


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")


# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# One ASGI app per deployment; AgentCore discovers the entrypoint through it.

app = BedrockAgentCoreApp()


# Headless runtime: there is nobody to answer a tool-consent prompt.
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Four values, all produced by Part 1 of the setup. Left as sentinels so that an
# unconfigured checkout reports precisely what is missing rather than failing
# somewhere deep inside the first tool call.

GATEWAY_URL = "<your-gateway-url>"
KB_ID       = "<your-knowledge-base-id>"
REGION      = "<your-aws-region>"
MEMORY_ID   = "<your-memory-id>"

_SENTINEL_CHARS = "<>"


def _unfilled(value: Optional[str]) -> bool:
    """True while a setting is empty or still holds its placeholder."""
    if not value:
        return True
    return any(ch in value for ch in _SENTINEL_CHARS)


def _missing_settings() -> List[str]:
    """Names of the settings still awaiting real values."""
    settings = {
        "GATEWAY_URL": GATEWAY_URL,
        "KB_ID": KB_ID,
        "REGION": REGION,
        "MEMORY_ID": MEMORY_ID,
    }
    return [name for name, value in settings.items() if _unfilled(value)]


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Nova 2 Lite through the global cross-region profile.

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)

# Boto3 validates the region eagerly and raises InvalidRegionError on a
# placeholder. Since both `agentcore configure` and `agentcore deploy` import
# this module before any of our code runs, an eager client would make an
# unconfigured checkout impossible to even inspect. So construction is deferred
# until the settings are known to be real.
memory_client: Optional[MemoryClient] = None
_bedrock_runtime = None


def _connect_clients() -> None:
    """Build the two AWS clients the tools need, once, on first use."""
    global memory_client, _bedrock_runtime
    if memory_client is None:
        memory_client = MemoryClient(region_name=REGION)
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


if not _unfilled(REGION):
    _connect_clients()


SYSTEM_PROMPT = """You are the customer support assistant for our store.

You can resolve four kinds of request: order tracking, returns and refunds,
product and policy questions, and loyalty rewards.

How to use your tools, without exception:
- search_knowledge_base holds the product catalogue, the returns policy, the
  warranty terms, the loyalty tiers, and the order status glossary. Call it
  before you answer anything in those areas. Your own recollection of what a
  policy "usually" says is not evidence.
- The order and refund tools come from the store gateway. They are the only
  source of truth about a specific order, customer, or refund.
- calculate_loyalty_discount runs the points and tier arithmetic in a sandbox.
  Call it for any discount, total, or points figure. Do not compute it
  yourself, not even for a single multiplication.
- browser reads live pages when a customer refers to a URL.

How to write your replies:
- Answer first, then the detail that supports it. Two or three sentences is
  usually plenty.
- Reproduce figures exactly as your tools returned them. Never round, re-derive,
  or restate a number from memory.
- If the knowledge base has nothing useful, say that plainly and offer to pass
  the customer to a human. Do not fill the gap with a plausible policy.
- If a customer mentions a preference or a personal detail, acknowledge it so it
  is remembered for next time.
"""


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    try:
        strategies = mem_client.get_memory_strategies(memory_id) or []
    except Exception:
        logger.warning("No memory strategies available for %s", memory_id,
                       exc_info=True)
        return {}

    templates: Dict[str, str] = {}
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue
        # The SDK mirrors the raw API keys onto friendlier ones; accept either
        # spelling so a SDK-side rename cannot silently empty this map.
        kind = strategy.get("type") or strategy.get("memoryStrategyType")
        raw = strategy.get("namespaces") or strategy.get("namespaceTemplates") or []
        if isinstance(raw, str):
            raw = [raw]
        if kind and raw:
            templates[kind] = raw[0]
    return templates


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────

class MemoryHook(HookProvider):
    """Long-term memory for the support agent.

    Two callbacks run per turn: context is pulled in before the model sees the
    customer's message, and the finished exchange is written back afterwards.
    Both are best-effort - a memory outage must degrade recall, never break the
    reply.
    """

    CONTEXT_HEADER = "Customer Context:"
    MAX_MEMORIES = 8          # more than this crowds out the actual question
    MAX_CONTEXT_CHARS = 1200

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)
        # Remembers exactly what we prepended, so it can be removed precisely
        # rather than by pattern-matching on the way out.
        self._preambles: Dict[str, str] = {}

    # ── message utilities ────────────────────────────────────────────────────
    @staticmethod
    def _parts(message: dict) -> List[dict]:
        content = message.get("content", [])
        if isinstance(content, str):
            return [{"text": content}]
        return [part for part in content if isinstance(part, dict)]

    @classmethod
    def _text(cls, message: dict) -> str:
        return "".join(
            part.get("text", "") for part in cls._parts(message) if "text" in part
        ).strip()

    @classmethod
    def _is_tool_output(cls, message: dict) -> bool:
        """Tool results arrive as user-role messages and are not customer speech."""
        return any("toolResult" in part for part in cls._parts(message))

    @staticmethod
    def _snippet(record) -> str:
        """Pull the text out of a memory record, whatever shape it arrives in."""
        if isinstance(record, str):
            return record.strip()
        if not isinstance(record, dict):
            return ""
        content = record.get("content")
        if isinstance(content, dict):
            return (content.get("text") or "").strip()
        if isinstance(content, str):
            return content.strip()
        return (record.get("text") or "").strip()

    # ── hooks ────────────────────────────────────────────────────────────────
    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Prepend anything we already know about this customer."""
        try:
            history = getattr(event.agent, "messages", None) or []
            if not history:
                return
            message = history[-1]

            # This event fires for assistant turns and tool results too.
            if message.get("role") != "user" or self._is_tool_output(message):
                return

            question = self._text(message)
            if not question or not self.namespaces:
                return

            # Collect per strategy, dropping duplicates: the same fact often
            # surfaces from more than one namespace.
            grouped: Dict[str, List[str]] = {}
            seen = set()
            capped = False
            for kind, template in self.namespaces.items():
                if capped:
                    break
                namespace = template.replace("{actorId}", self.actor_id)
                try:
                    records = self.memory_client.retrieve_memories(
                        self.memory_id,
                        namespace=namespace,
                        query=question,
                        top_k=5,
                    )
                except Exception:
                    logger.warning("Recall failed for %s", namespace, exc_info=True)
                    continue

                if isinstance(records, dict):
                    records = records.get("memoryRecordSummaries", [])
                for record in records or []:
                    snippet = self._snippet(record)
                    fingerprint = snippet.casefold()
                    if not snippet or fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    grouped.setdefault(kind, []).append(snippet)
                    if len(seen) >= self.MAX_MEMORIES:
                        capped = True
                        break

            if not grouped:
                return

            lines: List[str] = []
            for kind, snippets in grouped.items():
                lines.append(f"[{kind}]")
                lines.extend(f"  - {s}" for s in snippets)
            preamble = f"{self.CONTEXT_HEADER}\n" + "\n".join(lines)
            if len(preamble) > self.MAX_CONTEXT_CHARS:
                preamble = preamble[: self.MAX_CONTEXT_CHARS].rstrip()

            self._preambles[question] = preamble
            message["content"] = [{"text": f"{preamble}\n\n{question}"}]
            logger.info("Recalled %d memory item(s) for %s", len(seen), self.actor_id)
        except Exception:
            logger.warning("Context retrieval skipped", exc_info=True)

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Write the finished exchange back so the next session can recall it."""
        try:
            history = getattr(event.agent, "messages", None) or []

            question: Optional[str] = None
            answer: Optional[str] = None
            for message in reversed(history):
                role = message.get("role")
                if role == "assistant" and answer is None:
                    answer = self._text(message) or None
                elif role == "user" and question is None and not self._is_tool_output(message):
                    question = self._text(message) or None
                if question and answer:
                    break

            if not question or not answer:
                return

            question = self._strip_preamble(question)
            self._preambles.pop(question, None)

            self.memory_client.create_event(
                self.memory_id,
                self.actor_id,
                self.session_id,
                messages=[(question, "USER"), (answer, "ASSISTANT")],
            )
            logger.info("Stored exchange for %s", self.actor_id)
        except Exception:
            logger.warning("Memory write skipped", exc_info=True)

    def _strip_preamble(self, text: str) -> str:
        """Recover the customer's own words from a context-augmented message.

        Without this, every successive turn would re-save the recalled memories
        as though the customer had spoken them, and the noise compounds.
        """
        # Preferred path: we know exactly what we inserted this session.
        for original, preamble in self._preambles.items():
            if text.startswith(preamble) and text[len(preamble):].lstrip("\n") == original:
                return original

        # Fallback for text we did not construct (e.g. replay, restart).
        if text.startswith(self.CONTEXT_HEADER):
            _, separator, remainder = text.partition("\n\n")
            if separator and remainder.strip():
                return remainder.strip()
        return text

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        """Wire both callbacks onto the agent lifecycle."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or not KB_ID.strip() or _unfilled(KB_ID):
        return (
            "Knowledge Base is not configured: KB_ID is empty or missing. "
            "Please configure KB_ID before attempting a knowledge-base search."
        )
    if _bedrock_runtime is None:
        return (
            "Knowledge Base client is unavailable: the Bedrock Agent Runtime "
            "client could not be created. Check the REGION setting and credentials."
        )

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as exc:
        logger.warning("Knowledge base query failed", exc_info=True)
        return f"Knowledge base lookup failed: {exc}"

    passages: List[str] = []
    seen = set()
    for hit in response.get("retrievalResults") or []:
        # Retrieval can return the same passage from several chunks; repeating
        # it just spends context.
        text = ((hit.get("content") or {}).get("text") or "").strip()
        fingerprint = text.casefold()
        if text and fingerprint not in seen:
            seen.add(fingerprint)
            passages.append(text)

    if not passages:
        return "No relevant information was found in the knowledge base."

    return "\n---\n".join(passages)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────


@dataclass(frozen=True)
class RedemptionPolicy:
    """The redemption rules, kept in one place and mirrored into the sandbox."""

    points_per_dollar: int = 100       # 100 points redeem for $1
    increment: int = 500               # redemption happens in whole blocks
    max_fraction_of_order: float = 0.50  # points may cover at most half an order


@dataclass(frozen=True)
class EarnRates:
    """Points earned per dollar, by product category."""

    standard: int = 1
    device: int = 2
    fresh: int = 5


POLICY = RedemptionPolicy()
EARN_RATES = {"standard": 1, "device": 2, "fresh": 5}
TIER_RATES = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}


def _build_discount_code(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str,
) -> str:
    """Return the Python the sandbox will execute.

    The sandbox is a separate process with no access to this module, so every
    input and every constant is interpolated into the source. Kept as its own
    function so the arithmetic can be exercised without a live Code Interpreter.
    """
    return f'''
import json

account_points   = {int(loyalty_points)}
tier             = {tier!r}
order_total      = {float(order_total)}
product_category = {product_category!r}

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

points_per_dollar = {POLICY.points_per_dollar}
increment         = {POLICY.increment}
max_fraction      = {POLICY.max_fraction_of_order}

earn_rate = earn_rates.get(str(product_category).lower(), 1)
tier_rate = tier_rates.get(str(tier).capitalize(), 0.00)

# Redemption is capped two ways: it happens in whole {POLICY.increment}-point blocks,
# and it may never cover more than {int(POLICY.max_fraction_of_order * 100)}% of the order.
ceiling = int(order_total * max_fraction * points_per_dollar)
in_blocks = int(account_points // increment) * increment
redeemed = max(0, min(in_blocks, ceiling, int(account_points)))

points_value  = redeemed / points_per_dollar
subtotal      = order_total - points_value
tier_discount = round(subtotal * tier_rate, 2)
final_total   = round(subtotal - tier_discount, 2)
points_earned = int(final_total * earn_rate)

print(json.dumps({{
    "loyalty_points":   int(account_points),
    "tier":             str(tier).capitalize(),
    "tier_rate":        tier_rate,
    "product_category": str(product_category).lower(),
    "order_total":      round(order_total, 2),
    "points_redeemed":  int(redeemed),
    "points_value":     round(points_value, 2),
    "subtotal":         round(subtotal, 2),
    "tier_discount":    tier_discount,
    "tier_discount_pct": round(tier_rate * 100, 2),
    "final_total":      final_total,
    "total_savings":    round(points_value + tier_discount, 2),
    "points_earned":    points_earned,
    "remaining_points": int(account_points - redeemed + points_earned),
}}))
'''


@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = _build_discount_code(
        loyalty_points, tier, order_total, product_category
    )

    try:
        with code_session(REGION) as interpreter:
            response = interpreter.invoke(
                "executeCode",
                {"code": code, "language": "python", "clearContext": True},
            )
            for event in response.get("stream", []):
                printed = (
                    (event.get("result") or {})
                    .get("structuredContent", {})
                    .get("stdout", "")
                )
                if printed and printed.strip():
                    return printed.strip()

        raise RuntimeError("the sandbox produced no output")

    except Exception as exc:
        # Degrade to the part that needs no sandbox. Points redemption is
        # deliberately withheld rather than approximated: an invented discount
        # is worse than an admitted one.
        logger.warning("Code Interpreter unavailable; tier discount only",
                       exc_info=True)

        earn_rate = EARN_RATES.get(str(product_category).lower(), 1)
        tier_rate = TIER_RATES.get(str(tier).capitalize(), 0.00)
        tier_discount = round(order_total * tier_rate, 2)
        final_total = round(order_total - tier_discount, 2)

        return json.dumps(
            {
                "loyalty_points": int(loyalty_points),
                "tier": str(tier).capitalize(),
                "tier_rate": tier_rate,
                "product_category": str(product_category).lower(),
                "order_total": round(order_total, 2),
                "points_redeemed": 0,
                "points_value": 0.0,
                "subtotal": round(order_total, 2),
                "tier_discount": tier_discount,
                "tier_discount_pct": round(tier_rate * 100, 2),
                "final_total": final_total,
                "total_savings": tier_discount,
                "points_earned": int(final_total * earn_rate),
                "remaining_points": int(loyalty_points),
                "fallback": True,
                "note": (
                    "Code Interpreter was unreachable, so points redemption was "
                    "not applied; only the tier discount is included. "
                    f"Reason: {exc}"
                ),
            }
        )


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────

def _ensure_browser_driver() -> None:
    """
    Restore the execute bit on Playwright's bundled Node driver.

    Playwright is a thin RPC client: even when it only connects over CDP to the
    remote AgentCore browser, its transport spawns its own Node driver binary.
    The deployment zip is assembled on Windows, where NTFS has no execute bit
    at all, so every entry is recorded as mode 0666 and the driver unpacks on
    Linux without +x. Spawning it then fails with PermissionError(13), which
    surfaces as "Task exception was never retrieved" and a browser tool that
    dies mid-call while the agent reports the tool as unavailable.

    /var/task is read-only, so relocating the binary to a writable directory
    and repointing Playwright at it is the only usable repair. Idempotent, and
    a no-op on any platform that already carries the bit.
    """
    try:
        import playwright

        driver = Path(playwright.__file__).parent / "driver" / "node"
        if not driver.exists():
            return

        if os.access(driver, os.X_OK):
            logger.warning("Browser driver ready: %s", driver)
            return

        logger.warning(
            "Browser driver not executable: %s (mode %s) - relocating",
            driver,
            oct(driver.stat().st_mode),
        )
        relocated = Path(tempfile.gettempdir()) / "pw-node-driver" / "node"
        relocated.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(driver, relocated)
        relocated.chmod(0o755)
        os.environ["PLAYWRIGHT_NODEJS_PATH"] = str(relocated)
        logger.warning(
            "Browser driver relocated to %s (executable=%s)",
            relocated,
            os.access(relocated, os.X_OK),
        )
    except Exception:
        logger.warning("Browser driver bootstrap failed", exc_info=True)


def _reply_text(result) -> str:
    """Take the assistant's prose out of a Strands result object."""
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("text"):
                return part["text"]
    return str(result)


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        payload = payload or {}
        prompt = payload.get("prompt") or payload.get("message") or ""
        if not prompt:
            return "No prompt was provided in the request."

        customer_id = payload.get("customer_id") or "anonymous"
        session_id = payload.get("session_id") or str(uuid.uuid4())

        missing = _missing_settings()
        if missing:
            return (
                "This agent is not configured yet. Still missing: "
                + ", ".join(missing)
            )

        _connect_clients()

        hook = MemoryHook(customer_id, session_id, memory_client, MEMORY_ID)
        _ensure_browser_driver()
        browser = AgentCoreBrowser(region=REGION)
        local_tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            browser.browser,
        ]

        # ExitStack keeps the MCP session open for the whole invocation while
        # still guaranteeing teardown. The gateway carries the order and refund
        # tools; if it is unreachable we would rather answer with reduced
        # capability than return nothing at all.
        with ExitStack() as stack:
            gateway_tools = []
            try:
                gateway = stack.enter_context(
                    MCPClient(lambda: streamable_http_client(GATEWAY_URL))
                )
                gateway_tools = list(gateway.list_tools_sync())
                logger.info("Gateway supplied %d tool(s)", len(gateway_tools))
            except Exception:
                logger.warning(
                    "Gateway unreachable; continuing with local tools only",
                    exc_info=True,
                )

            agent = Agent(
                model=model,
                tools=local_tools + gateway_tools,
                hooks=[hook],
                system_prompt=SYSTEM_PROMPT,
            )
            return _reply_text(agent(prompt))

    except Exception as exc:
        logger.exception("Invocation failed")
        return f"Sorry, I could not complete that request: {exc}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
