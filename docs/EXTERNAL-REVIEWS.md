# External intelligence on LFM2.5 / LFM3 (from user-supplied reviews)

_Collected 2026-10-28 from five independent video reviews plus the Liquid AI release
notes. Kept because several items are directly actionable and some contradict the
vendor benchmarks._

## Actionable: samplers (now applied)
The GGUF declares its own recommended sampling, and **both independent reviewers used
exactly these values**:

```
general.sampling.temp           = 0.1
general.sampling.top_k          = 50
general.sampling.penalty_repeat = 1.05
```

The app had been shipping temp **1.0** / top_k 20 / repeat 1.0 — inherited from the
176B Flash-Next model this fork was split from. Fixed in commit `5f7b491`.

Honest caveat: an A/B on a simple factual question (temp 1.0 vs 0.1, 3 samples each)
gave identical, correct answers both ways. The difference will show up on *hard*
prompts, not easy ones — this change is about variance and hallucination rate, not
about the median answer.

## Actionable: tool-call format
From the quant/deployment review: the model **"emits Python-style function calls
inside dedicated markers by default. Structured JSON calls must be explicitly
configured in the system prompt."**

This matches what I extracted from the template —
`<|tool_call_start|>[func_name(arg=value)]<|tool_call_end|>` — and it matters for the
harness: don't assume JSON tool calls are the default; either accept the Python-style
form or state the JSON requirement in the system prompt.

**Tool calling worked in my own test** (`get_weather({"city":"Chicago"})`,
`finish_reason: tool_calls`, optional param correctly omitted). But one reviewer
reports it *"routinely dropped or missed required tool calls"* inside OpenCode. So:
verify per-harness, don't assume.

## Where the reviews agree with my measurements
- **Speed**: a Vulkan review measured LFM2.5 at **37–39 tok/s vs Qwen3.6-35B's
  25–28**, and prompt processing **447–647 vs 317–364**. That is the same shape and
  magnitude as my device numbers (~31–37 t/s decode, ~83 t/s prefill), so the phone
  is behaving like a normal llama.cpp deployment of this model.
- **Reasoning-only**: all reviewers note it always emits a chain of thought first.
  This is what causes empty `content` when the token budget is small — I hit that
  repeatedly and it is a real harness trap.
- **QAD_Q4_0** is the recommended low-memory quant, which is what we ship.

## Where the reviews disagree with the vendor
Liquid's benchmarks put LFM2.5 ahead of Qwen3.6 on agentic work. The independent
testing says the agentic *intent* is real but execution is weak:

| area | verdict |
|---|---|
| agentic diagnosis / investigation / instruction following | **good** — "respectable", avoids overthinking loops |
| heavy scientific arithmetic | **fails** — got mass wrong (607 vs 663 kg), nonsense result |
| procedural visual/frontend code (canvas, CSS animation) | **fails consistently** — blank canvases, grey blobs |
| low-resource multilingual | **weak** — gibberish outside high-resource languages |
| coding overall | **weak** — one leaderboard scored it 1 vs 5 |
| safety refusals | **good** |

A five-test physics-simulation head-to-head scored **Qwen 4–0** on *visual output*
while LFM was 35–45% faster. So: **fast and agentic, but not a code/visual
generator.** For a phone assistant that's an acceptable profile — it argues for
keeping generated-code ambitions small and leaning on tools.

## LFM3 — released since we started, and it changes the options
Verified present on Hugging Face (download counts real):

| repo | notable file | size |
|---|---|---|
| `LiquidAI/LFM3-1.7B-GGUF` | QAD-Q4_0 | **0.98 GiB** |
| `LiquidAI/LFM3-VL-2B-GGUF` | QAD-Q4_0 + mmproj-Q8_0 | **0.98 + 0.19 GiB** |
| `LiquidAI/LFM3-Omni-2B-GGUF` | any-to-any (text/image/audio/video) | — |
| `LiquidAI/LFM3-Audio-2B-GGUF` | ASR/TTS, 18 languages | — |

Architecture: **Tri-Hybrid** (gated short convolutions + sliding-window attention +
global attention), **256k context**, and at 4-bit the 1.7B base fits in ~1 GB.

Published claims put **LFM3-1.7B** well above LFM2.5-2.6B and above Qwen3.5 4B/9B on
the agentic sets (LiveClaw 45.60 vs 14.70; AutomationBench 41.29 vs 0.94; BFCLv4
71.39 vs 55.89; ToolSandbox 80.36). Treat as vendor claims, but the direction is
consistent with LFM2.5's own agentic strength.

**Why this matters for us:**
- **Vision becomes cheap.** `LFM3-VL-2B` QAD-Q4_0 + mmproj is **1.17 GiB** — it fits
  beside the 24B with room to spare, and the catalog already has an `MMPROJ` asset
  kind. This is the answer to "can we send docs to it": not with the 24B, but yes
  with this.
- **A tool router becomes cheap.** `LFM3-1.7B` at ~1 GiB with 256k context is a
  natural fast path for tool dispatch, leaving the 24B for answer quality.

## Standing risk
Agent loops accumulate context, and decode falls with context (measured: ~31 t/s at
500 tokens, ~25 at 4K, ~15–17 at 8K). Keep tool results small or trim aggressively.
LFM3's 256k window makes this *worse* if used naively — a big window is not free.
