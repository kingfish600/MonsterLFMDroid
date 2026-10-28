# Agentic capability: what LFM2.5-24B-A2B can actually do

_Measured on the RedMagic 10s Pro, 2026-10-28. This is the feasibility evidence for
building an agent harness into MonsterLFMDroid._

## Tool calling works end-to-end — verified, not assumed
`llama-server --jinja` + the OpenAI-compatible `tools` parameter, against the model
we already ship. Request defined one function (`get_weather(city, units?)`); the
model returned:

```
finish_reason : tool_calls
name          : get_weather
arguments     : {"city":"Chicago"}
decode        : 34.78 t/s
```

It also correctly **omitted the optional `units` parameter**, and its reasoning trace
shows it deciding to do so. So the whole loop is available: define tools → model
emits a structured call → harness executes → feed the result back as a `tool` role
message → model continues.

**The chat template is built for this.** Extracted from the GGUF, it contains
`tool_call` ×20 and `tool_calls` ×14, and renders calls as
`<|tool_call_start|>[func_name(arg=value)]<|tool_call_end|>`.

## Why this model in particular
Liquid's release notes state it was **post-trained inside agent harnesses** —
OpenClaw, Hermes Agent, Pi, OpenCode — through agentic RL. It is a
**reasoning-only** model (always emits a chain of thought first), which is exactly
what tool-use loops want.

Its published benchmark position is the interesting part. Against
Qwen3.6-35B-A3B (the model we scrapped), LFM2.5 wins most of the *agentic* set:

| benchmark | LFM2.5-24B-A2B | Qwen3.6-35B-A3B |
|---|---|---|
| AutomationBench | **27.69** | 8.72 |
| LiveClaw | **48.09** | 38.19 |
| ToolSandbox | **80.71** | 75.02 |
| IFEval | **96.75** | 93.06 |
| IFBench | **74.69** | 64.21 |
| Multi-IF | **84.38** | 81.53 |
| τ³-Bench Banking | **8.04** | 7.63 |
| BFCLv4 | **65.01** | 65.00 |
| GPQA Diamond | 74.42 | **83.69** |
| SWE-Bench Verified | 51.00 | **66.20** |
| SciCode | 32.71 | **37.35** |

So: better at tool use and instruction following, behind on knowledge and coding.
For a phone assistant, that is the right side of the trade.

## Answers to the specific asks

**Web search (DuckDuckGo)** — straightforward and the highest-value first tool. The
harness performs the HTTP request; the model only emits `web_search(query)`. Needs
a JSON tool schema, an executor, and result injection. No model changes.

**"Can it use phone apps?"** — the *model* can ask; the *harness* decides what is
executable. Exposing Android capabilities (open app, share, send SMS, read
clipboard, notifications) is Kotlin-side work with runtime permissions, and each
one is a security decision, not a model capability. Feasible, but should be
deliberately scoped — start read-only.

**"Is it a visual model we can send docs to?"** — **No, this one is text-only.**
LFM2.5-24B-A2B has no vision tower. Liquid ships separate VL variants:
`LFM2.5-VL-3B`, `LFM2.5-VL-1.6B`, `LFM2.5-VL-450M`. Text documents (PDF/txt →
text) work today with no model change; images/screenshots need a VL model, and the
app's catalog already has an `MMPROJ` asset kind so the plumbing exists in
principle.

**"A lightweight harness like LiveClaw / DeepSeek harness?"** — this is the right
shape, and the model was trained for it. The app already has a foreground service,
an HTTP server, and a streaming UI; what it lacks is a tool registry, an executor,
and a multi-turn loop. That is a contained addition rather than a rewrite.

## Suggested build order
1. **Tool registry + loop** — schemas, executor, `tool` role round-trip. Nothing
   user-visible until 2.
2. **`web_search` (DuckDuckGo)** — first real tool; makes the model's knowledge
   weakness a non-issue and plays directly to its tool-calling strength.
3. **`fetch_url`** — read a page the search returned. Cheap once 2 exists.
4. **Read-only phone context** — clipboard, current time/location, notifications.
   No permissions that can act on the user's behalf.
5. **Acting tools** (open app, share, SMS) — only after the above, and gated.
6. **Vision** — separate VL model + mmproj, if wanted. Independent track.

## Constraint to keep in mind
Decode falls off with context (measured): ~31 t/s at 500 tokens, ~25 at 4K,
~15-17 at 8K. An agent loop *accumulates* context fast — every tool result lands in
the window. So a harness needs either aggressive context trimming, or to keep
tool results small, or it will spend its speed budget on context. This is the main
engineering risk in the whole idea.
