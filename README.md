# Customer Support Agent — Amazon Bedrock AgentCore

An AI customer support agent for a fictional e-commerce store, built with the
[Strands Agents SDK](https://strandsagents.com) and deployed to
[Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/).

It handles order tracking, refunds, product and policy questions, loyalty
calculations and live web lookups through a single conversational interface,
while remembering customers across separate sessions.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │   AgentCore Runtime          │
                        │   Nova 2 Lite + Strands      │
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
              Local tools        Gateway (MCP)      AgentCore Memory
                    │                  │                  │
      ┌─────────────┴───────┐   ┌──────┴───────┐   ┌──────┴───────┐
      │ search_knowledge_base│  │ managed-kb   │   │ SEMANTIC     │
      │ calculate_loyalty_…  │  │ refund-actions│  │ USER_PREFERENCE│
      │ browser              │  │ order-tracker │  └──────────────┘
      └──────────────────────┘   └──────────────┘
```

### Gateway targets (7 MCP tools across 3 target types)

| Target | Type | Tools |
|---|---|---|
| `managed-kb` | Connector → Bedrock Managed Knowledge Base | `Retrieve` |
| `refund-actions` | Lambda | `initiate_refund`, `check_refund_status`, `get_return_label` |
| `order-tracker` | OpenAPI schema → API Gateway | `track_order`, `list_customer_orders`, `get_customer_profile` |

Gateway tools are discovered at runtime via `MCPClient.list_tools_sync()` and
merged with the agent's local tools, so adding a backend means adding a target
rather than changing agent code.

### Local tools

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Retrieval-augmented answers from the product catalogue and store policies |
| `calculate_loyalty_discount` | Exact loyalty arithmetic executed in the AgentCore Code Interpreter sandbox |
| `browser` | Live web page retrieval via the AgentCore Browser |

---

## Design notes

**Least-privilege tools.** Each capability is exposed as a bounded tool rather
than handing the model broad access. Retrieval is scoped to one knowledge base,
arithmetic runs in a sandboxed interpreter, and the browser is a managed session.

**Failure isolation.** No single dependency should take down the conversation.
Tools return readable messages instead of raising: if the knowledge base is
unconfigured the agent says so, if the code interpreter is unreachable the
loyalty tool falls back to a tier-only calculation and flags it, and the agent
keeps its local tools if the Gateway connection fails. A support agent that goes
silent is worse than one that admits it cannot check something right now.

**Arithmetic is never done in the model.** All loyalty maths runs as generated
Python inside the Code Interpreter, so figures are exact and the business rules
live in one auditable place rather than in prompt text.

**Memory is keyed to the customer, not the session.** Memories persist under
`/users/{actorId}/...`, so a brand-new session can still recall a returning
customer's name and preferences.

---

## Configuration

Four values at the top of `main.py` must be set before deploying. The repository
ships with placeholder values, so an unconfigured checkout reports exactly which
settings are missing:

| Constant | Description |
|---|---|
| `REGION` | AWS region, e.g. `us-east-1` |
| `GATEWAY_URL` | AgentCore Gateway MCP endpoint |
| `KB_ID` | Bedrock Knowledge Base ID |
| `MEMORY_ID` | AgentCore Memory resource ID |

Client construction is lazy, so the module imports cleanly even before these are
filled in — which keeps `agentcore configure` and `agentcore deploy` working.
If a value is missing at invocation time, the agent returns a clear message
naming exactly what is unset instead of failing obscurely.

---

## Setup

Requires Python 3.10+ and the AgentCore starter toolkit.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Infrastructure to provision beforehand:

1. Two Lambda functions from `lambda/` — `order_tracker` behind API Gateway
   (REST proxy integration), `refund_processor` invoked directly.
2. An AgentCore Gateway with three targets: the Knowledge Base connector, the
   Lambda target registered from `lambda/lambda_schema`, and an OpenAPI target
   pointing at the API Gateway stage.
3. A Bedrock Knowledge Base ingested with `product_catalog.txt`.
4. An AgentCore Memory resource with SEMANTIC and USER_PREFERENCE strategies.

The Gateway execution role needs `bedrock:Retrieve` and `bedrock:GetKnowledgeBase`
on the knowledge base; the runtime execution role additionally needs the memory
and browser permissions, plus `bedrock-agent-runtime:Retrieve`.

## Deploy

```bash
agentcore configure --entrypoint main.py --name <agent-name> \
  --requirements-file requirements.txt --region us-east-1

agentcore deploy
```

Then invoke:

```bash
agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```

---

## Verified behaviour

| Scenario | Result |
|---|---|
| Order tracking | Status, carrier and tracking number returned from the API-backed Gateway target |
| Refund processing | Refund ID issued with `APPROVED` status and a 3–5 business day credit timeline |
| Knowledge base (RAG) | Loyalty tier benefits answered from the ingested catalogue |
| Cross-session memory | A second session, same customer, recalled the name and communication preference stored in the first |
| Loyalty calculation | 4,000 points redeemed, 10% tier discount, $99.00 final total, 349 points remaining |
| Browser | Live page title retrieved from the web |

---

## Notes and limitations

- **Deployed with `direct_code_deploy`**, which packages dependencies with `uv`
  and ships them to S3, skipping the container build path entirely.
- **A Windows build host needs care.** Windows filesystems have no execute bit,
  so archives created there store mode `0666`. Playwright launches a bundled
  helper binary even when only connecting to a remote browser, so that binary
  arrives without execute permission and fails on Linux. `_ensure_browser_driver()`
  in `main.py` detects this, relocates the binary to a writable location and
  marks it executable at start-up. Building on Linux or macOS avoids the issue.
- **Tool errors are reported to the model as text**, which means a tool failure
  is a conversational turn rather than an exception. This is deliberate — it
  keeps the agent responsive — but it does mean monitoring should watch for
  error text in responses, not just for stack traces.

---

## Attribution

The Lambda backends (`lambda/order_tracker.py`, `lambda/refund_processor.py`,
`lambda/lambda_schema`) and `product_catalog.txt` were supplied as part of the
Udacity *Developing AI Agents with Amazon Bedrock AgentCore* project scaffold and
are included here so the project runs end to end. The agent implementation in
`main.py` — the runtime entrypoint, tool integration, retrieval, memory hooks,
sandboxed calculation and deployment handling — is original work.

## License

Released under the MIT License — see [LICENSE](LICENSE). The MIT grant covers
the original work in this repository; the scaffold files listed above remain
subject to their original terms.

