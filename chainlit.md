# iPhone Guide Assistant

A grounded, tool-calling RAG agent that answers **strictly** from the official
**iPhone User Guide (iOS 7.1)** — and always shows its work.

- Every answer **cites the page and section** it came from.
- If something isn't in the guide, it says so instead of guessing.
- The conversation is multi-turn and remembers context within your session.

Ask away — for example: *"How do I connect to Wi-Fi?"*

---

## What the agent can do

It's a **`gpt-4o` tool-calling agent** (a **ReAct-style** reason-and-act loop
built on **LangGraph**). On each turn it reasons about your question and decides
whether to search the guide, answer directly, or refuse — then performs
**grounded, retrieval-augmented generation (RAG)** to synthesise one cohesive,
cited answer.

### Tools it can call

| Tool | When it's used |
| --- | --- |
| **`search_guide`** | A single iPhone topic — a setting, a how-to, setup, or troubleshooting. |
| **`search_guide_parallel`** | One message that bundles several distinct topics (e.g. *"take a screenshot, set up a hotspot, and save battery"*). Sub-queries are retrieved **in parallel** and grouped so the answer stays cohesive. |

The agent picks the tool itself; you don't have to. Greetings and off-topic
messages are answered without a search.

---

## How retrieval works (under the hood)

A **hierarchical (coarse-to-fine) hybrid retrieval** pipeline — visible live in
the **execution trace** under each answer and in the **Live dashboard**:

1. **Coarse stage (hierarchical retrieval)** — narrows the guide down to the
   most relevant *sections* before reading passages, the classic
   coarse-to-fine strategy.
2. **Fine stage (hybrid search)** — retrieves the best passages **within** those
   sections by combining:
   - **Dense / semantic retrieval** — embedding-based vector similarity, and
   - **Sparse retrieval** — **Okapi BM25** lexical keyword match,
   - fused with **Reciprocal Rank Fusion (RRF)** into one ranked list.
3. **Parent-document retrieval (contextual expansion)** — widens each hit with
   neighbouring text from the same section ("semantic contextual retrieval"),
   so multi-step instructions aren't cut off.
4. **Cross-encoder reranking (optional)** — a reranker reorders the final
   shortlist for precision; if it's unavailable or errors, the agent
   **gracefully degrades** and continues without it (no failed turn).
5. **Grounded synthesis with faithfulness checks** — the model answers only from
   the retrieved passages and attaches page + section citations.

For multi-topic questions the agent also performs **multi-query retrieval**
(`search_guide_parallel`), fanning out concurrent searches per sub-topic.

Chunks are built with **semantic, page-preserving chunking**, so citations map
back to real pages in the guide.

---

## Memory & context

- **Live window** — the most recent messages are kept verbatim.
- **Rolling summary** — once the conversation passes the fold threshold, older
  turns are summarised into long-term memory so context survives without
  blowing the token budget. The **Live dashboard** shows exactly how many
  messages remain until summarisation kicks in.

---

## Safety

- **Prompt-injection guard** screens each message before the model runs and
  refuses jailbreak / "ignore your instructions" attempts.
- **Strict grounding** — the agent won't answer from outside the guide, and
  unsupported claims are filtered out.

---

## Transparency in the UI

- **Steps** under each answer — expand to see the agent's reasoning, every tool
  call, and the **fused RRF scores** of retrieved passages.
- **Citations** — click any source to preview the exact passage (page + section).
- **Live dashboard** (header, top-right) — token usage vs the model limit,
  memory window & summarisation countdown, retrieval config, latency, and
  guardrail status.
- **⚙️ Settings** — tune `retrieval_top_k`, `coarse_sections_n`, the reranker
  toggle, and more, live.

---

## Techniques at a glance

For reviewers, the named methods used here:

- **Retrieval-Augmented Generation (RAG)** — grounded answers from a private corpus.
- **ReAct-style tool-calling agent** (LangGraph) — reason → act → observe loop.
- **Hierarchical retrieval** — coarse section selection → fine passage retrieval.
- **Hybrid search** — dense (semantic embeddings) + sparse (**Okapi BM25**).
- **Reciprocal Rank Fusion (RRF)** — rank-based fusion of the two retrievers.
- **Semantic chunking** — embedding-aware, page-preserving splitting.
- **Parent-document retrieval / semantic contextual retrieval** — context expansion.
- **Cross-encoder reranking** — precision reordering, with graceful degradation.
- **Multi-query retrieval** — parallel sub-query fan-out for multi-topic asks.
- **Conversational memory with rolling summarisation** — live window + long-term summary.
- **Prompt-injection guardrails & grounded faithfulness checks** — safety + no hallucination.
- **Vector store** — **Qdrant** (cloud) with a local **FAISS** fallback.
- **Observability** — **LangSmith** tracing + in-app execution trace and dashboard.
