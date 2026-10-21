# Stage 9 daytime prep (host-only)

Date: 2026-10-18/19. **The phone is unplugged and released: no `adb` command was run in this
session.** Every number below comes from the 32-core host PC running the real `varH` weights
(`moe-dense-q8/varH-0000{1,2}-of-00002.gguf`, 37.0 + 28.8 GB).

This file is written incrementally, one section per task, in the order the audit specified:
**Task 1 -> Task 2 -> Task 3**.

---

## Task 1: Cross-engine functional equivalence on `varH` - are the two graphs the same function?

**Short answer: the two engines compute the same function with the same top-k, but they do NOT meet
the audit's probability criterion.** Both engines, run head-to-head on the host with the same weights
and matched flags, produce **16/16 identical greedy tokens**, an **identical top-5 token ID set in the
same order**, and a distribution that agrees to float precision wherever the model is confident - but
the top-5 *probabilities* differ by **6-37% relative** at the positions where tokens compete, against a
`<= 1e-3` acceptance criterion. So: same architecture, same graph function, same argmax; not the same
numbers at near-ties.

**And the audit's correction on hyper-connections is accepted: `ik` does not omit them.** Its
`qwen4exp_hc_mix` / `qwen4exp_hc_combine` (`src/graphs/build_qwen4exp.cpp:23,77`) build the same mixing
from primitives (`scale`, `sigmoid`, `reshape`, `repeat`) where our `models/qwen4exp.cpp:342` emits one
fused `LLM_FUSED_OP_DSV4_HC_PRE/POST` per site. The earlier "ik has 0 HC ops" reading was an artefact
of comparing op *names*: the maths is present in both, and **ours is the cheaper formulation**
(1.761 ms/token, 1.2% of our decode budget, for what ik spreads across ~40 small nodes per layer).
That is the opposite of a gap and must not be read as one.

### 1.1 The head-to-head run

Both engines were built for the host and run as `llama-server` against the same shard, with the same
request:

```
built:   upstream-llama/build-x86/bin/llama-server      (ours)
         ik_llama.cpp/build-x86/bin/llama-server        (ik, built this session: LLAMA_BUILD_SERVER=ON)
request: prompt "Explain how a bicycle works.", n_predict 16, temperature 0, seed 1,
         cache_prompt false, n_probs 5
flags:   -c 2048 -ngl 0 -t 16 -tb 16 --no-warmup -fa off -ctk f16 -ctv f16
```

**The flag matching matters, and the first run got it wrong.** With each engine's *defaults* (ours
running q8_0 KV, ik f16, and the two versions disagreeing about flash attention) the texts diverged at
token 6 - ours `"The user asks me to explain"`, ik `"The user is asking me to explain"`. That looks
like a graph difference and is not one: with `-fa off -ctk f16 -ctv f16` on both, the outputs are
**identical**. So the first comparison measured flag drift, which is exactly the kind of false positive
this audit exists to avoid.

| position | our top-1 | ik top-1 | delta (nats) | token |
|---|---|---|---|---|
| 0 | -0.1188 | -0.1793 | **+0.0605** | `"\n\n"` |
| 1 | -0.0085 | -0.0153 | +0.0068 | `"<think>"` |
| 2 | -0.1993 | -0.1853 | -0.0141 | `"\n"` |
| 3-4 | -0.0000 | -0.0000 | 0.0000 | `"The"`, `" user"` |
| 5 | -0.7698 | -0.6986 | **-0.0713** | `" is"` |
| 7 | -0.3400 | -0.4117 | +0.0718 | `" me"` |
| 8-15 | -0.0001 .. -0.0042 | same | **0.0000** | `" to explain how a bicycle works. This"` |

**The audit's acceptance criterion, evaluated:**

| criterion | result |
|---|---|
| identical top-1 token ID | **PASS** - both `271` = `"\n\n"` |
| identical top-5 token ID set | **PASS, in the same order** - `271 "\n\n"`, `248046 "<|im_end|>"`, `28763 " Include"`, `198 "\n"`, `4844 " Then"`. Our rank-2 looked different only because our server renders the special token `248046` as the empty string; the vocabulary in shard 1 confirms `248046 = "<|im_end|>"`, the exact token ik named |
| top-5 probabilities within `<= 1e-3` relative | **FAIL** - 6.24% / 22.4% / 36.6% / 36.7% / 8.1% relative on ranks 1-5. It passes only at the confident positions (3-4, 8-15), where the two engines agree to ~1e-5 |

So two of the three criteria hold and the third does not, by 60-370x. That is the central numerical
result of Task 1 and it is not dismissible as "last-bit float32 associativity": as §1.2 shows, it
corresponds to a ~0.3 nat difference in a logit gap.

* **Token identity: 16 of 16 positions identical.** Both engines emit `271, 248068, 198, 760, 1156,
  369, 9859, 728, 310, 10033, 1204, 264, 33828, 4138, 13, 1061` (ours reports ids; ik's server does not
  return a token array, so its ids are inferred from the identical text - the text agrees
  character-for-character over the compared prefix).
* **Top-5 rank agreement at position 0**: rank 1 `"\n\n"`, rank 3 `" Include"`, rank 4 `"\n"`, rank 5
  `" Then"` are the same tokens in both; rank 2 is our id `248046` (renders as the empty string) against
  ik's `"<|im_end|>"`, which is the same special token displayed differently.
* **The drift is small, and it is concentrated exactly where it should be**: mean |delta| = **0.0141
  nats**, max **0.0718**. Wherever the model is confident (p ≈ 1.0, positions 3-4 and 8-15) the two
  engines agree to 1e-5 - i.e. to float precision. The 0.06-0.07 nat differences appear only at
  positions where two or three tokens genuinely compete (p ≈ 0.45-0.9), which is what a tiny logit
  perturbation looks like through a softmax.

### 1.2 The residual difference is larger than summation-order noise, and that is the real finding

At position 0 our top-2 logit gap is `log(0.888019/0.018131) = 3.890` and ik's is
`log(0.835857/0.023364) = 3.577` - a **0.31 nat difference in a logit gap**. Pure float32 reordering
of a 2560-wide dot product is ~1e-5 relative (~1e-3 absolute), so 0.31 is 300x too large to be
arithmetic noise. Something in the two graphs is *formulated* differently at the ~0.1-0.3 logit level
even though the greedy path agrees.

Candidates, in the order they would explain a 0.1-0.3 shift, each of which is a *legitimate* difference
between a 2026 fork and a mid-2024 one and none of which is "missing maths":

| candidate | why it fits | how to settle it |
|---|---|---|
| **QSA / pooled attention** (`compress_ratios`, `kpool`, `top_score`, `selection_groups`) - **leading candidate** | ours emits a discrete `ARGSORT` top-k over pooled keys (48 calls/token); ik has no `ARGSORT` node. A pool-index selection that flips at a near-tie changes *which* keys are attended, which moves logits by ~0.1-0.3 in one step - the only mechanism in this list that can produce a difference that large, and it explains why the drift appears **only** where tokens compete: when the model is confident the selection is not near a tie | force the indexer to attend all pooled keys on both engines (`compress_ratios` off / top-k = all) and re-compare; if the probabilities then agree to 1e-3, the indexer selection is the cause |
| **PLE n-gram hashing** (`per_layer_token_embd` injected per layer, `ple_conv_kernel`) | our `build_inp_ple` hashes `ple_n_heads` rows per token and mixes with a conv1d; an off-by-one in the offsets shifts every layer's injected vector slightly | compare against ik with the PLE contribution zeroed on both |
| **RoPE/attention kernel** | non-FA attention differs between versions in mask/softmax detail | already matched `-fa off`; next is `-ctk f32` |

**What this means for the audit question.** It rules *out* the hypothesis that `ik` omits or
approximates layers - hyper-connections included, per the correction above: if it did, the 16-token
greedy path and the top-5 set would not both agree. It rules *in* one specific, bounded numerical
difference (~0.3 logit at near-ties, zero at confidence) whose most likely cause is **discrete QSA
pool selection**, which is testable in one experiment rather than a bisection. It is **not** a
performance lever and it is **not** a reason to change our graph.

**And it means the engines are not interchangeable for quality claims.** A cross-engine PPL comparison
would carry this ~0.3-logit difference on top of the thing being measured; paired PPL must stay
within-engine.

### 1.3 Source-level graph comparison: where every op-count difference comes from

Both trees implement the same architecture (`qwen4exp`, 48 blocks, 512 experts, hyper-connections
`dsv4_hc_mult`, per-layer token embeddings). `ik` has `build_qwen4exp()` in `llama-build-context.cpp`
and parses `hc_*`, `per_layer_token_embd`, `output_hc_*`; ours has `src/models/qwen4exp.cpp`. The
structural differences are **fusion granularity**, and each of the audit's five counts has a named
source:

| op | ours (calls/token) | ik (calls/token) | where ours emits it | what ik does instead |
|---|---|---|---|---|
| `DSV4_HC_PRE` / `POST` | **192** (96 + 96), **1.761 ms/token** | **0 by name** | `models/qwen4exp.cpp:342,385` -> `add_fused_node(LLM_FUSED_OP_DSV4_HC_PRE/POST)` - **our** fused op for hyper-connection mixing (2 per site x 2 sites per layer x 48) | **not omitted**: `ik` builds the identical mix from primitives in `qwen4exp_hc_mix`/`qwen4exp_hc_combine` (`src/graphs/build_qwen4exp.cpp:23,77`) and fuses neighbours into `FUSED_RMS_NORM` (96), `MULTI_ADD` (109), `MUL_MULTI_ADD` (48), `FUSED_MUL_UNARY` (179), `L2_NORM` (72). **Our single fused op is the cheaper formulation of the same maths** |
| `MUL_MAT` f32 | **130** (5.758 ms/token) | **0** | the hyper-connection low-rank projections (`hc_*_down` `[hc_dim, hc_lr]` -> `hc_*_up` `[hc_lr, hc_dim]`) run as plain F32 matmuls | the same projections, folded into `MUL_MULTI_ADD` / `MULTI_ADD` fused ops. **Note this reconciles exactly: 833 - 703 = 130** |
| `ARGSORT` | **48** (1/layer) | **0** | `llama-graph.cpp:2234` `ggml_argsort_top_k(selection_groups, 2)` - the QSA pool selector's group top-k | its `top_score`/`live` selection is done inside its fused ops (`:843` `ggml_get_rows(top_k)`) without a separate sort node |
| `CONCAT` | **73** | **0** | `models/qwen4exp.cpp:761` (k pack), `:817` (pool tail), `:849` (live pool), `:1321` (PLE conv input), plus the MTP tail | ik builds the same packs with `CPY`-into-view / fused ops instead of `CONCAT` |
| `GET_ROWS` | **174** (5.791 ms) | **14** (0.029 ms) | five distinct sites: token embedding (`:555`), the PLE n-gram gather (`:1362`, `ple_n_heads` rows/token), per-layer `inp_out_ids` gathers (`:474,475,478,507,602`), the QSA pool index gather (`:814`), and the top-k score gather (`:843`) | ik gathers the same information with **far fewer nodes** - its PLE path is one gather per step rather than ours per layer |
| `CPY` | 186 | 77 | cache writes and view copies | fewer because more is fused |

So: **the graphs are mathematically equivalent in structure and in greedy output, and the op-count
gap is our fork spending graph nodes on fine-grained ops where ik spends separate fused ops.** That is
a *kernel/op-fusion* gap, exactly as the Stage 8 disassembly suggested, and it is why "optimize toward
ik's op counts" is the wrong framing: the same work has to happen, it is only *where* it happens that
differs.

---

## Task 2: `D1` - the packed activation retargeted at the winning four-row decode kernel

### 2.1 Why the previous two attempts could not have worked

| attempt | kernel | where it applied | pack frequency | outcome |
|---|---|---|---|---|
| Stage 7 `A6` (`BONSAI_Q8_0_X4=1`) | **row-pair** (`MOE_X4=3`'s loop) | decode | once per 16-row block | null (and the kernel itself was -3.6%) |
| Stage 8 `C2` (`MOE_X4_PREFILL_PACKED=1`) | 4 rows x 2 tokens (`x4t2`) | **prefill tile only** | once per (token pair, 16-row block), cached | bit-exact, unmeasured |
| **Stage 9 `D1` (`BONSAI_Q8_0_X4=2`)** | **4 rows x 1 token, mode-1 order** | **decode `n = 1`, both call sites, plus the n>1 tail** | **cached per (row pointer, n)** | **bit-exact, unmeasured (device next window)** |

The audit's two facts are both correct and both fatal to the earlier attempts: `C2` never touched the
decode path, and `A6` was only ever measured on a kernel that had already lost. `D1` is the first
version that puts the packed layout under the shipped winning kernel, on the path where it is
actually used.

### 2.2 `bonsai_x4_rows4_mode1_packed`

`arch/arm/iqk_x4.c`. The body is `bonsai_x4_rows4_mode1`'s, unchanged in structure: four weight rows
share one expansion of a block, eight accumulators alternate even and odd blocks, each row resolves as
`vaddvq(e + o)`. The only difference is the activation accessor - four 16-byte loads from
`rec*144 + half*64 .. +64`, i.e. contiguous 128-byte records with the scales at `rec*144 + 128`, using
the same contiguous-latency `block_q8_0_x4` layout ik uses. It is explicitly **not** the `MOE_X4=3`
row-pair loop.

Stack frame **304 B with 20 stores / 21 loads to `[sp]`** against mode 1's **432 B / 28 / 27** -
recorded as a **diagnostic only**, never as an acceptance criterion, because Stage 7 proved zero spills
can lose (`MOE_X4=3`, -3.62%) and 55 spills can win (`MOE_X4=1`, +4.53%).

### 2.3 Gate 1, kernel level: bit-exact on all eight sizes

`MoeDroid/tools/s6_q2_0_x4_test.c`, `qemu-aarch64-static`, 4 rows against `ggml_vec_dot_q2_0_q8_0`:

| `nb` | 1 | 2 | 3 | 5 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| **D1** | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** |

(mode 1 and mode 3 are shown in the same harness as controls; mode 2 remains near-exact by
construction at <= 7.63e-05.)

### 2.4 Gate 1, graph level: byte-identical on all six `MUL_MAT_ID` shapes

`MoeDroid/tools/s7_mmid_prefill_verify.sh`, arm `d1` (`BONSAI_Q8_0_X4=2`):

```
case0..case5   D1==stock   D1 stable
```

including **case 5 (single token, 6 threads - the decode shape this exists for)**, **case 4** (no
expert receives two rows, so the stock tail runs) and the 32-token shape. The `stock1` arm shows
`pk_hits=0 pk_misses=0`, so the counter cannot be reporting work that did not happen.

### 2.5 The cache-hit measurement the audit asked for

| shape | tokens | threads | hits | misses | **hit rate** |
|---|---|---|---|---|---|
| case 0 | 16 | 8 | 2725 | 795 | 77.4% |
| case 1 | 16 | 8 | 1291 | 373 | 77.6% |
| case 2 | 32 | 8 | 1571 | 1245 | 55.8% |
| case 3 | 16 | **1** | 3906 | 126 | **96.9%** |
| case 4 | 4 | 8 | 202 | 54 | 78.9% |
| **case 5** | **1** | **6** | **300** | **20** | **93.8%** |

**The audit's prediction is confirmed in shape and slightly under in magnitude.** On the single-token
decode shape with 6 threads the pack is reused **93.8%** of the time: every thread working on
`ffn_up_exps` / `ffn_gate_exps` asks for the *same* `(src1 pointer, n)` key, so each thread's first
touch is a miss and every subsequent chunk is a hit. The prediction of "99%+" is 94% in practice
because the test's 10-experts-used x 48-layer walk interleaves `up`, `gate` and `down` activation rows,
and `down` uses a different row (1 x 512) that evicts one of the two slots.

The 8-thread multi-token shapes are much worse (56-78%) for the same reason Stage 8 measured: with
several token rows in flight a two-slot cache thrashes. **That is fine for D1's purpose** - D1 targets
`n = 1`, where there is exactly one activation row per op - and it is why D1 is scoped to decode rather
than being claimed as a general win.

### 2.6 Decode-only by construction (audit requirement)

`BONSAI_Q8_0_X4=2` is gated so it engages **only** when the activation carries a single token, which
makes it exactly orthogonal to `C2` (`BONSAI_MOE_X4_PREFILL_PACKED=1`, prefill-only):

| call site | token-count test | why |
|---|---|---|
| `mul_mat_id_one_chunk` (experts) | **`ne12 == 1`** | in `MUL_MAT_ID` the activation is `[n_embd, n_expert_used, n_tokens]`, so the **token** count is `ne12`, not `ne11` |
| `mul_mat_one_chunk` (dense) | `ne11 == 1` | there the activation is `[n_embd, n_tokens]`, so `ne11` *is* the token count |

**The harness caught this exact mistake in one run.** The first version gated the expert path on
`ne11 == 1`, which reads naturally from the audit's wording but is the *expert* count in `MUL_MAT_ID`,
so D1 disabled itself on every shape - the qemu verification reported `pk_hits=0` for all six and the
mistake was visible in seconds rather than on the device. Corrected to `ne12`, the hit pattern is now
exactly the intended one:

| shape | tokens | threads | pk_hits | pk_misses | D1 engaged? |
|---|---|---|---|---|---|
| case 0 | 16 | 8 | 0 | 0 | no (C2's territory) |
| case 1 | 16 | 8 | 0 | 0 | no |
| case 2 | 32 | 8 | 0 | 0 | no |
| case 3 | 16 | **1** | 0 | 0 | no |
| case 4 | 4 | 8 | 0 | 0 | no |
| **case 5** | **1** | **6** | **300** | **20** | **yes, 93.8% hit** |

### 2.7 Thread safety (audit requirement)

The packed buffer is the `__thread` (thread-local) `bonsai_pk_cached` slot, so each worker packs its
own private copy on its first 16-row chunk of that activation row and hits it thereafter - there is no
shared mutable state and therefore no race to lose. Verified by execution rather than by argument:
`D1 == stock` is byte-identical on **every** shape, and the shapes span **1, 6 and 8 threads** (case 3
is `-t 1`, case 5 is `-t 6`, cases 0/1/2/4 are `-t 8`). Since "stock" here *is* `MOE_X4=1` (it is the
default, `BONSAI_MOE_X4` unset), `D1 == stock` is literally the audit's requested check: **D1 output is
byte-identical to the unpacked `MOE_X4=1` kernel at `-t 1`, `-t 6` and `-t 8`.**

### 2.8 What is still missing, stated plainly

`D1` has **no speed measurement**. It is bit-exact at both levels and the cache demonstrably hits, but
whether the two saved activation loads per 128 weights per row beat mode 1's spilling arithmetic is a
question only the device can answer. It is arm `D1` in the next window (see §3.3), with the ship rule
fixed in advance: **both mirrored decode passes positive and neutral-or-better prefill.**

---

## Task 3: `GET_ROWS` (174 vs 14) - attributed, and **no Gate-1 cleanup exists**

### 3.1 The audit's premise does not hold, and the code says why

The addendum proposes that `ik` does one `ggml_get_rows(model.tok_embd_per_layer, rows)` up front and
slices it per layer, while we gather per layer. **`ik` gathers per layer too.** Its
`qwen4exp_ple(...)` takes the layer index and its own body contains
`ggml_get_rows(ctx0, model.tok_embd_per_layer, rows)` followed by `cb(emb, "ple_embd", il)`
(`ik_llama.cpp/src/graphs/build_qwen4exp.cpp:207-210`, with `il` in the callback name). The PLE gather
is therefore not the difference, and hoisting it is not available.

Our own PLE gather is in fact **once per graph**, not per layer: `build_inp_ple` (line 1362) is called
from `:422` outside the layer loop, and the only `inp_out_ids` gathers inside the per-layer block are
guarded by `il == n_layer - 1 && crop_before_nextn(...)` - i.e. **last layer only** (3 gathers, once).

### 3.2 Then where do 174 calls come from? Per-site attribution

`llama_get_rows` sites in `models/qwen4exp.cpp`, classified by whether their builder runs per layer:

| site | builder | runs | calls/token |
|---|---|---|---|
| `:814` `sel_idx` (pool index gather) | `build_qsa_sel` | **per layer** | 48 |
| `:843` `top_score` (pool top-k gather) | `build_qsa_sel` | **per layer** | 48 |
| `llama-graph.cpp:2235` `group_scores` | MoE router | **per layer**, guarded by `n_expert_groups > 1` (active here: `ARGSORT` measures exactly 48/token, i.e. 1/layer) | 48 |
| `:474,475,478` / `:507` | `build_hc_combine` | **last layer only** (`il == n_layer-1`) | 3 |
| `:1362` PLE n-gram gather | `build_inp_ple` | **once per graph** | 1 |
| `:555` token embedding | `build_inp_embd` | once per graph | 1 |
| `:602`, `:1223`, `:1680`, nextn/MTP paths | assorted | once per graph | ~5 |

**96 + 48 = 144 of the 174 calls (83%) are the QSA pool selector and the MoE group selector, both
per layer.** The remaining ~30 are embeddings, the PLE gather, the last-layer crops and the MTP head.
The arithmetic closes: `144 + 30 = 174`.

`ik`'s 14 comes from doing those two selections **inside fused ops** rather than as standalone
`GET_ROWS` nodes - the same trick as its hyper-connection primitives. Its `MULTI_ADD` (109),
`MUL_MULTI_ADD` (48), `FUSED_MUL_UNARY` (179) and `FUSED_RMS_NORM` (96) are where that work goes.

### 3.3 Verdict: no Gate-1-safe graph cleanup, and that is the answer

**None of those gathers is redundant.** Each performs a real, algorithm-required selection:

* the QSA gathers select *which pooled KV keys* the indexer attends - removing or batching them changes
  the attention inputs, so it is not a cleanup, it is a different model;
* the MoE group gather selects *which expert groups* are candidates - same argument;
* the last-layer crops are already minimal (3, once).

So there is **no `G1` (`BONSAI_QWEN4_GRAPH=1`) proposed tonight**, and I am not going to ship a
speculative graph edit on the strength of a call count. A wrong change here is silent: it produces
slightly different logits, which is exactly the class of bug §1.2 shows is already at the edge of
detection.

What *would* reduce the 5.791 ms is **fusing** those selections into single ops (the ik strategy), which
is a kernel/op project with a new ggml op, a CPU kernel, and a graph change - not a flag. The honest
scoping number for that project: the QSA + MoE selectors are `5.791 ms (GET_ROWS) + 0.944 ms
(ARGSORT) + 2.296 ms (CONCAT) = 9.03 ms/token, 6.1% of our 147 ms decode budget`, and fusion would
recover only part of it.

### 3.4 The shootout update

`MoeDroid/tools/s8_shootout.sh` now carries the two packed variants as separate, orthogonal arms plus
their identity gates:

| arm | env | scope | status after this session |
|---|---|---|---|
| `A0` | - | shipped defaults (`MOE_X4=1`, tile ON) | reference |
| **`D1`** | `BONSAI_Q8_0_X4=2` | **decode only** (`ne12 == 1`) | **Gate 1 PASS at kernel and graph level, 93.8% cache hit - unmeasured speed** |
| **`C2`** | `BONSAI_MOE_X4_PREFILL_PACKED=1` | **prefill only** (`ne11`/token > 1) | Gate 1 PASS (Stage 8), unmeasured speed |
| `A5` | `BONSAI_MOE_X4=3` | decode | rejected twice (Stage 7, Stage 8) |
| `C1` | `BONSAI_MOE_X4_PREFILL=2` | prefill | rejected in Stage 8 (both decode passes negative) |
| `B0`/`B1`/`B2` | ik flags | reference engine | gap confirmed at +38.8% decode / +34.0% prefill |

`D1` and `C2` are 100% orthogonal by construction (`ne12 == 1` vs multi-token), so a single window can
measure them back-to-back without interaction. Nothing here is claimed as a win; both are Gate-1
bit-exact, which is the only thing that could be established on the host.

---

## Summary

| task | outcome | hard evidence |
|---|---|---|
| 1. cross-engine equivalence on `varH` | **same function, same top-k; the audit's probability criterion FAILS at near-ties** (6.2-36.7% relative vs 1e-3 limit) while agreeing to 1e-5 where the model is confident; HC omission **withdrawn** | 16/16 identical greedy tokens; identical top-5 ID set; per-position logprob table; both `qwen4exp` builders read |
| 2. `D1` - packed activation on the winning decode kernel | **Gate 1 PASS at both levels** (0.000e+00 on 8 sizes; byte-identical to `MOE_X4=1` on 6 shapes at 1/6/8 threads); **93.8% cache hit** on the single-token shape; decode-only and orthogonal to `C2` | `.stage7/t/s6_q2_0_x4_test` under qemu; `s7_mmid_prefill_verify.sh` arm `d1` |
| 3. `GET_ROWS` 174 vs 14 | **premise refuted**: `ik` gathers per layer too, and ours is 96 QSA + 48 MoE + ~30 misc, none redundant. **No `G1` proposed**; the 5.8 ms needs op fusion, scoped at 9.03 ms/token | per-site attribution table + measured call counts |

### What the audit changed in this document

Four corrections were applied, and three of them changed a conclusion:

1. **`DSV4_HC` is not omitted by `ik`.** Its `qwen4exp_hc_mix`/`qwen4exp_hc_combine` build the same maths
   from primitives; our fused op (`1.761 ms/token`, 1.2% of decode) is the *cheaper* formulation. The
   "19.1 ms headline" is withdrawn - it compared op *names*, not work.
2. **The Task 1 acceptance criterion was made explicit** (identical top-1 ID, identical top-5 ID set,
   probabilities within `1e-3` relative) and the measurement **fails the third clause**. Publishing the
   criterion before the measurement is what makes that a result rather than a rationalisation.
3. **`D1` was made strictly decode-only.** Gating it correctly required `ne12` (tokens) in the expert
   path, not `ne11` (experts) - the harness caught the wrong guard immediately as `pk_hits=0` on all six
   shapes, which is the argument for verifying a guard's *effect* and not just its intent.
4. **The `GET_ROWS` hoisting premise was refuted from `ik`'s own source.** No graph flag is shipped on
   a call count.

### Files touched

| file | change |
|---|---|
| `MonsterQwenDroid/docs/STAGE9-daytime-prep.md` | this document |
| `upstream-llama/ggml/src/ggml-cpu/arch/arm/iqk_x4.c` | `bonsai_x4_rows4_mode1_packed`, `bonsai_vec_dot_q2_0_q8_0_x4_packed4`, `bonsai_q8_0_x4_mode()` extended to 0/1/2 |
| `upstream-llama/ggml/src/ggml-cpu/ggml-cpu.c` | `D1` wiring in both `_one_chunk` sites with the `ne12 == 1` decode guard |
| `upstream-llama/ggml/src/ggml-cpu/ggml-cpu-impl.h` | prototypes |
| `MoeDroid/tools/s6_q2_0_x4_test.c` | D1 kernel gate over 8 sizes |
| `MoeDroid/tools/s7_mmid_prefill_verify.sh` | `d1` arm, D1 comparison column, pack-cache reporting |
| `MoeDroid/tools/s8_shootout.sh` | `D1` arm + its two identity gates, symbol checks |
| `MonsterQwenDroid/iterations.jsonl` | iterations 61-63 |
| `.stage9/run_one.sh`, `.stage9/{ours,ik}*.json` | the cross-engine harness and its raw responses |

### The next window, in one line

`s8_shootout.sh 0` then `s8_shootout.sh 3` now measures **`D1` (decode, packed 4-row) against `C2`
(prefill, packed tile)** in the same mirrored passes, both Gate-1 bit-exact, with `A0`/`A5`/`C1`/`B0`/
`B1`/`B2` as the established references. Ship rule unchanged and pre-declared: both mirrored passes
positive on the phase it targets, neutral-or-better on the other.

---

## URGENT: `bonsai_pk_cached` was stale across nodes - fixed and proven

**Verdict: the audit is right, the bug was real, it is fixed, and the fix is now covered by a test that
fails without it.** No shipped behaviour was ever affected (both packed paths default to off), but
neither `D1` nor `C2` could have been enabled safely.

### U.1 The bug, confirmed in the code

```c
static const void * bonsai_pk_cached(const void * row, int n, int slot) {
    if (g_pk_n[slot] == n && g_pk_src[slot] == row) {      // <-- no generation, never invalidated
        return g_pk_buf[slot];                             //     FALSE HIT after wdata is rewritten
    }
    ...
}
```

`params->wdata` is **one** scratch buffer: every `MUL_MAT` / `MUL_MAT_ID` node re-quantises its own
`src1` into it at offset 0 (`ggml-cpu.c:1247`, `:1583`). So `src1_col = wdata + (i11 + i12*ne11)*row_size`
lands on the *same address* in the next node and in the next token, with *different contents*. Keyed on
`(pointer, n)` alone the cache reported a hit and returned the previous node's - or previous token's -
packed activation. Silent, and wrong.

### U.2 The fix: per-node generation in the key

* `_Thread_local uint64_t g_pk_gen` plus `g_pk_ent_gen[slot]`, and the cache compares the generation
  as part of the key. A stale entry can no longer match.
* `bonsai_pk_invalidate()` bumps the generation, and it is called **once per node per worker thread**,
  in `ggml_graph_compute_thread`, immediately before the node runs. A per-*graph* counter would not do:
  `wdata` is rewritten at every node **inside** one graph, which is exactly layer 0 -> layer 1.
* The hook costs one load + branch when both packed switches are off (`bonsai_pk_enabled()` caches the
  decision once), and two stores when they are on.
* The property that mattered is preserved: **inside one node `wdata` is immutable during COMPUTE**, so
  all of that node's 16-row chunks still hit the cache. Only cross-node reuse is dropped - and crossing
  nodes is precisely what was unsafe.

### U.3 The mandatory host proof (before -> after, both directions)

Level 1, direct in-place overwrite through the real cache (`s6_mmid_prefill_test.c`, `bonsai_pk_probe`):

```
S9PK level1 first_access_fresh=1 stale_without_invalidate=1 (is_A=1 is_B=0) fresh_with_invalidate=1
S9PK level1 PASS: stale without invalidation, fresh with it
```

i.e. the same pointer with new contents **is** served stale when nobody invalidates (the bug, reproduced
deliberately) and is **fresh** after `bonsai_pk_invalidate()` (the fix). Both halves are asserted, so the
test would fail if either behaviour changed.

Level 2, two sequential evaluations of the **same graph** with `src1` overwritten in place at the same
address - the audit's accepted alternative, and the closest model of what the next token does:

| shape | **before fix** (invalidation disabled) | **after fix** |
|---|---|---|
| `tokens=1, experts_used=1, -t 1` | **FAIL** - second eval used stale packed data | **PASS** (hits 394 / misses 25) |
| `tokens=1, experts_used=1, -t 6` | **FAIL** | **PASS** (hits 484 / misses 31) |
| `tokens=1, experts_used=1, -t 8` | **FAIL** | **PASS** (hits 574 / misses 37) |
| `tokens=16, experts_used=1, -t 1/6/8` | pass | **PASS** |
| `tokens=16, experts_used=10, -t 6` (control) | pass | **PASS** |

Each line prints the live `pk_hits/pk_misses`, because **a Level 2 pass is worthless if the cache was
never consulted** - the first version of this test passed without the fix for exactly that reason, and
the counters are what exposed it.

**Why `n_used = 1` is the shape that triggers it:** with one expert there is exactly one activation row,
hence one `wdata` pointer, so the two-slot cache recurrs across nodes instead of thrashing. With ten
experts and two slots every access evicts, so the stale read cannot happen - which is also why the
session-long tests and the device identity gates never saw it.

### U.4 Why every earlier test passed (the audit's question, answered)

`s7_mmid_prefill_verify.sh` and `s6_mmid_prefill_test.c` built **one `MUL_MAT_ID` node per graph and
evaluated it once**, with a fresh context and cplan per case. A cache that is only ever populated and
re-read within one node cannot go stale, so the harness was structurally unable to see the bug. The
Level 2 test above exists because of that gap.

### U.5 Stage 7's `A6` was safe - the cache is what introduced the hazard

Item 4 of the audit, checked in the code: `A6` (`BONSAI_Q8_0_X4=1`) calls
`bonsai_pack_q8_0_x4(ne00, g_q8_0_x4_buf, src1_col)` **unconditionally on every call** into a fixed
per-thread buffer - there is no key, no cache and no reuse, so it repacks per 16-row block and **had no
staleness bug** (it was merely slow, which is the +26% traffic Stage 7 measured). `bonsai_pk_cached` and
its key were introduced in **Stage 8, Step 4** (`C2`), and `D1` inherited it. So:

| variant | cached? | staleness bug? | shipped default? |
|---|---|---|---|
| `A6` `BONSAI_Q8_0_X4=1` (Stage 7) | no - unconditional repack | **no** | off |
| `C2` `BONSAI_MOE_X4_PREFILL_PACKED=1` (Stage 8) | yes | **yes, now fixed** | off |
| `D1` `BONSAI_Q8_0_X4=2` (Stage 9) | yes | **yes, now fixed** | off |

Because all three default to `0`, **no device measurement and no shipped APK in Stages 7-9 ever ran the
affected code**. That is luck rather than design, and the lesson is recorded: a cache keyed on a pointer
into a recycled scratch buffer needs a liveness token from the day it is written.

### U.6 Side finding: a graph with two `MUL_MAT_ID` nodes crashes in the arm64 build

While building the two-node version of Level 2 I found that a graph containing **two** `MUL_MAT_ID`
nodes `SIGSEGV`s under `qemu-aarch64-static` at every thread count, **with every packed flag off**, so
it is not related to this fix. The single-node cases are unaffected and the whole existing harness runs
one node per graph. It is recorded here as an open item: either this fork cannot express a two-`MUL_MAT_ID`
graph, or the arm64 path has an out-of-bounds access that a real model with two MoE matmuls per layer
would hit on device. Worth settling before anything relies on it.

### U.7 State after the fix

| check | result |
|---|---|
| `s7_mmid_prefill_verify.sh` (8 arms, 6 shapes) | **PASS**, every arm `== stock`, stable on repeat |
| `D1` cache amortisation on the single-token shape | **300 hits / 20 misses = 93.8%**, unchanged by the fix |
| `C2` cache on multi-token shapes | hits preserved within a node |
| `HOST_CHECK=1 s8_shootout.sh 0` | **0 failures** |
| kernel-level Gate 1 (8 sizes) | unchanged, `0.000e+00` |

---

## Tonight's run: the two additions (A mandatory, B optional) are wired and host-verified

The phone is away, so this is harness work: implemented, syntax-checked, unit-tested on the host, and
gated by `HOST_CHECK=1 s8_shootout.sh 0` (**0 failures**, 27 checks now). Neither addition has been
executed on a device yet.

### A. Prove `D1` / `C2` actually engaged - a 0-hit arm is INVALID, not neutral

This is the guard against the failure that already happened once: gating D1 on `ne11` disabled it
everywhere, and only the host harness noticed because every shape showed `pk_hits=0`. On the device
nothing replays that counter, so a no-op D1 would have been published as "no effect".

**Engine side** (`ggml-cpu.c`): `BONSAI_PK_REPORT=1` registers an `atexit` reporter that prints
`BONSAI_PK hits=N misses=M` to stderr. It registers lazily, from inside the cache itself, so a run with
the packed paths off prints nothing at all - the line is positive evidence of engagement rather than
noise. Verified under qemu: with `BONSAI_Q8_0_X4=2` the line appears (`hits=574 misses=37`); with the
flags off it does not (0 occurrences).

**Arm side** (`s8_arm_ll.sh`): every **PACKED** arm sets `BONSAI_PK_REPORT=1` - carried in the
`C2_ENV` / `D1_ENV` defaults themselves, because the validity rule reads the counter on *timed* arms
and the script only exports it inside its ident block. `A0`, `AP2` and the `B*` baselines do **not**
set it: their verdict is `n/a`, and a counter line on them would only pollute logs they get compared
against. `stop_and_report_pk` itself stays silent on non-packed arms, so no `S9PK ... unavailable=1`
line is written for them either - the guard was exercised directly:

```
env=[<none>]                              -> silent (n/a arm)
env=[BONSAI_Q8_0_X4=2]                    -> reports S9PK
env=[BONSAI_MOE_X4_PREFILL_PACKED=1]      -> reports S9PK
env=[BONSAI_Q8_0_X4=1]                    -> reports S9PK
```

Every arm ends with `stop_and_report_pk`, which sends **SIGINT first** (llama-server shuts down gracefully so `atexit` can
run), waits up to 20 s, and only then SIGKILLs. It greps the engine log and emits

```
S9PK tag=<arm> hits=<H> misses=<M> rate=<H*100/(H+M)>%
```

or `S9PK tag=<arm> unavailable=1` if no line appeared. That applies to **both** exit paths (timing and
identity), so no arm can end without a verdict.

**Analysis side** (`s8_ll_table.py`): a new block near the top of the report lists every arm's hits,
misses, hit rate and verdict, and the rule is mechanical:

| arm state | verdict |
|---|---|
| packed flag set, `hits > 0`, **6/6 measured samples** | **engaged** |
| packed flag set, `hits == 0` | **INVALID (0 hits)** - t/s must not be quoted |
| packed flag set, no `S9PK` line at all | **INVALID (no counter)** - fail closed |
| **any arm** with `n < meas` (e.g. killed after 4 of 6 requests) | **INVALID (truncated 4/6)** |
| no packed flag, 6/6 samples (e.g. `A0`, `A5`, `C1`) | **n/a** - not judged |

**Footnote on `measured` - `6/6` is not the same as "clean".** `6/6` means six samples *landed*, not
that six samples ran un-throttled. **Every prefill arm reads 6/6 while the throttling trip lands inside
the window at request 5 or 6 of 6** (§3b.2), so a prefill median mixes two clock regimes. Those arms
stay admissible on purpose - invalidating them would delete prefill measurement entirely - but a prefill
median must always be read together with the `throttled` column that now sits beside it, and prefill
**deltas** are comparable only among arms with the **same K**. A 2/6 arm and a 0/6 arm are not a matched
pair; Stage 8 already reported every prefill number this way, and the validity table now makes the
distinction unmissable rather than implicit.

**The second rule exists because the first one was about to become the new fragile link.** `hits > 0`
is satisfied by an arm killed at graph 200 of 1024: its rate would render as `@g200` and read as early,
but its **t/s would still be published**, computed from however many of the six measured requests
happened to finish. So truncation is now checked on the same footing as cache engagement, and it
applies to **every** timed arm rather than only the packed ones - a truncated `A0` is just as
unquotable as a truncated `D1`. The report's validity table carries both reasons in one verdict column
(`INVALID (0 hits)`, `INVALID (truncated 4/6)`), and a non-packed truncated arm is one of the
self-tests, so the rule cannot regress into "only checks the packed arms".

Invalid arms are counted and named under the table, so a null result cannot silently be published. Five
self-tests cover the rule, including the zero-hit and missing-counter cases.

**Hit rates are printed next to t/s** so a high number can be sanity-checked against the shape:
`n_used = 1` legitimately gives ~94% (one activation row, one `wdata` pointer); `n_used = 10` thrashes
the two slots (56-78%) and **must not be claimed as a cache win** - that distinction is now visible in
the report rather than buried in a log.

### B. Cross-prompt freshness, same live session (`id-d1-p2`)

A same-prompt repeat cannot expose a stale packed activation, because the activation row is identical.
A **different** prompt rewrites `wdata` with new values at the same address, which is the case that can.

Implemented as `IDENT_PROMPT2=1` in `s8_arm_ll.sh`: after the normal identity request, still inside the
same live server, it issues a third request - `"Summarize how tides are caused."`, `temp 0`, `seed 1`,
`IDENT_N` tokens - and emits `S7IDENT2`/`S7IDTOK2`. The driver runs the pair `id-A0-p2` (stock) and
`id-d1-p2` (`BONSAI_Q8_0_X4=2`) and compares the second-prompt token arrays with a new `gate2`, which
reports `IDENTITY2 PASS` or `IDENTITY2 DIVERGES ... first differing second-prompt token index N`.

Cost: two identity arms, ~3 minutes - inside the 7-minute budget.

### What neither addition covers, stated so the record is honest

Both are device-side cover. **Neither replaces the host node->node test** that proves the generation
counter is what makes the cache safe (`U.3`): A proves the cache was *used*, B proves it is *fresh
across a prompt change*, and the host Level 2 test proves it is *fresh across the wdata rewrite that
happens between nodes*. A and B would both pass while a per-graph-only (rather than per-node) key was
in place, because both cross a graph boundary; only the Level 2 test catches that.

---

## The last dependency, closed: the generation bumps per NODE, not per GRAPH

`A` (`S9PK` counters) proves the cache was used. `B` (`id-d1-p2`) proves freshness across a prompt
change. **Both cross only a graph boundary, so both pass with a per-graph-only key** - they cannot tell
the two fixes apart. This test does, because it puts two nodes that each re-quantise `wdata + 0`
**inside one graph**.

### The test

`node_alias_case` in `MoeDroid/tools/s6_mmid_prefill_test.c`, run by `MoeDroid/tools/s9_pk_node_alias.sh`:

* **node A**: `GGML_OP_MUL_MAT_ID`, q2_0 experts, `n_used = 1`, 1 token - the trigger shape, because one
  expert means one activation row means one `wdata` pointer, so the two-slot cache recurrs instead of
  thrashing;
* **node B**: `GGML_OP_MUL_MAT`, a *different* q2_0 weight, 1 token, with **no data dependency on A**,
  so both execute in one graph and both quantise into `wdata + 0`;
* A's and B's `src1` are different tensors with different contents, so a stale hit is guaranteed to be
  wrong rather than accidentally right;
* acceptance: B must equal a **stock two-graph reference** where A and B run in separate evaluations.

`BONSAI_PK_INVALIDATE=node|graph` selects the fix under test from **one binary**; `node` is the default
and the shipped behaviour, `graph` exists only to prove the test is discriminating.

### The six cells

| cell | mode | `-t` | A_hits | A_misses | B_hits | B_misses | B == stock | first differing element |
|---|---|---|---|---|---|---|---|---|
| 1 | **node** | 1 | 31 | 1 | 31 | 1 | **YES** | - |
| 2 | **node** | 6 | 30 | 2 | 30 | 6 | **YES** | - |
| 3 | **node** | 8 | 30 | 2 | 24 | 8 | **YES** | - |
| 4 | **graph** | 1 | 31 | 1 | 32 | **0** | **NO** | **0** |
| 5 | **graph** | 6 | 30 | 2 | 32 | 4 | **NO** | **172** |
| 6 | **graph** | 8 | 30 | 2 | 26 | 6 | **NO** | **0** |

**Non-vacuity holds in every cell**: `A_hits > 0` **and** `B_hits > 0`, reported separately by op class
(`tag 1 = MUL_MAT_ID`, `tag 0 = MUL_MAT`), so node B demonstrably consulted the cache and could
therefore be served stale data. Packs per node stay at ~1 per worker thread (1 miss at `-t 1`, 2-8 at
`-t 6/8`), which is the expected shape for a `_Thread_local` cache.

Cell 4 is the clearest evidence: at `-t 1`, **`B_misses = 0`** - node B never packed anything at all, it
was served node A's activation outright. In cells 5-6 the main thread packs its own copy while the
remaining workers hit A's stale entry, so the output is wrong (and wrong at element 172 in cell 5)
rather than uniformly stale - which is exactly why a partial-thread-coverage test could look green.

### The statement

**Per-node invalidation proven; per-graph-only key demonstrably serves a stale pack at -t 1, -t 6 and
-t 8.**

### Consequence for tonight

The dependency is closed, so **`D1` enters tonight's timing matrix as planned** and `C2` stays in it.
No arms are pulled. Re-run the whole gate at any time with:

```
MoeDroid/tools/s9_pk_node_alias.sh      # exit 0 only if node matches AND graph differs
```

### A.1 The counter no longer depends on a clean shutdown

The atexit-only reporter made the evidence chain fragile: it needed the arm to stop the engine with
SIGINT so `atexit` could run, and any SIGKILL (a hung arm, a crashed server, a timeout) would skip it,
leave no counter line, and cause the analyzer to mark a **good** packed arm INVALID.

Fixed by printing periodically as well:

* **one line at graph 1, then every 512 graphs**, from **thread 0 only**, to stderr:
  `BONSAI_PK why=periodic graphs=<N> hits=<H> misses=<M>`
  (`BONSAI_PK_FIRST` / `BONSAI_PK_EVERY` override them; `EVERY=0` stops after the first line);
* the exit line is still emitted, tagged `why=exit`, and is simply the most complete one;
* the arm script now reads the **last** line of either kind (it previously took the *first*), so the
  report is produced whether the engine exited cleanly or was killed;
* the `S9PK` line gained `graphs=` and `source=periodic|exit`, so a `source=periodic` arm is visibly
  one that did not stop cleanly.

Verified under qemu, then verified again against a log with the exit line **deleted** (exactly what a
SIGKILL leaves behind):

```
S9PK tag=demo hits=902 misses=69 rate=92.9 graphs=36 source=periodic
```

That case previously produced `unavailable=1` and would have been marked INVALID. Cost when the flag is
off: nothing (the reporter is not registered). Cost when it is on: one increment and one branch per
graph on thread 0, and a handful of `fprintf`s per arm.

**Why the anchor is graph 1 and not merely "small".** A 262-token prefill is **one** graph, so an
identity prefill arm is warmup (~9 graphs) plus the measured request, i.e. **~10-15 graphs in total** -
an anchor of 16 would sit above those arms entirely and silently restore the shutdown dependency this
mechanism exists to remove. The anchor is only useful at its floor: graph 1, which always lands during
warmup, well outside any measured window, and guarantees a counter on disk for **any** run that executes
a single graph. The 512 cadence then keeps the in-window writes to a handful. Steady state with the
defaults, on a 36-graph run:

```
BONSAI_PK why=periodic graphs=1  hits=1   misses=2
BONSAI_PK why=exit     graphs=36 hits=926 misses=77
```

With the exit line deleted - what a SIGKILL leaves - the arm still reports
`hits=1 misses=2 rate=33.3 graphs=1 source=periodic`. That is **valid but early**, and the two facts it
supports are the two that matter: the cache demonstrably engaged (`hits > 0`, so the arm is not INVALID)
and the mechanism no longer depends on a clean shutdown. The *rate* from an early line is not the arm's
final rate, so the report renders the graph count beside it - `93.8% @g1024` - and a stale figure is
visible as such instead of passing for a measurement. There is no longer a kill window that yields no
line at all.

**One correction to my own reasoning, recorded because it matters:** a `_Thread_local` cache means each
worker packs its own copy, so a two-node test must expect **~1 pack per worker**, not ~1 per node. The
node-alias test originally asserted `<= 2` and flagged its own cells as anomalous at `-t 6`/`-t 8`; the
bound is now `<= n_threads + 1`. The counters were right and my expectation was wrong.

### A.2 The pattern behind the three stale assertions

Three of the gate failures this session were the *same* failure, and none of them was a code bug:

| assertion | what had changed underneath it | what was corrected |
|---|---|---|
| `B_misses <= 2` (node-alias) | a `_Thread_local` cache packs once **per worker**, not once per node | the bound: `<= n_threads + 1` |
| `<= 2 packs per node` (same cell, same reasoning) | same | same |
| `grep 'BONSAI_PK hits='` (host gate) | the format string became `BONSAI_PK why=...` when periodic lines were added | the pattern: `grep 'BONSAI_PK why='` |

Each time the fix was to correct the **assertion**, not the code - and that is the right instinct
precisely because the alternative (bending the code until the old assertion holds) is how a gate stops
being a gate. But it only works if the assertion's own correctness is testable, which is why:

* the analyzer carries a self-test (now covering the packed-arm verdict rule, the gate verdicts, the
  telemetry parsing and the injected deltas), because a report generator that silently parses the wrong
  field produces confident, wrong tables;
* the kernels carry Gate 1 at two levels (48 kernel pairs, 6 graph shapes), because a kernel that is
  merely "close" is a wrong model;
* the harness carries unit tests for its own control flow (the gate-retry stub, 4 cases).

**Treat a failing gate as an accusation against the assertion first and the code second** - but require
evidence for which one it is, and write the evidence down. The three corrections above are recorded
here so the next session can see that the gate has been wrong before, and how it was told apart.

---

## Pre-run fixes (before the phone is plugged in)

### The blocker: the reporter flag was on the wrong arms

`BONSAI_PK_REPORT` was exported only inside `s8_arm_ll.sh`'s **ident** block (`:280`), while the
validity rule reads the counter on **timed** arms. As staged, every timed `C2`/`D1` arm would have
emitted `S9PK unavailable=1` and been marked **INVALID by my own rule** - measured and published
neither. Fixed by carrying the flag **with the packed flag** in the arm defaults:

```
C2_ENV=${C2_ENV-BONSAI_MOE_X4_PREFILL_PACKED=1 BONSAI_PK_REPORT=1}
D1_ENV=${D1_ENV-BONSAI_Q8_0_X4=2 BONSAI_PK_REPORT=1}
```

The `-` form (not `:-`) is deliberate so `C2_ENV=""` still skips the arm. The export was **not** moved
to the top of the arm script: that would put a counter line on `A0` and the `B*` baselines, whose
verdicts are `n/a`, and pollute the very logs they are compared against.

**The coupling is now asserted, not remembered** (host gate), and the gate prints what it resolved:

```
resolved C2_ENV=[BONSAI_MOE_X4_PREFILL_PACKED=1 BONSAI_PK_REPORT=1]
resolved D1_ENV=[BONSAI_Q8_0_X4=2 BONSAI_PK_REPORT=1]
OK   every packed arm carries BONSAI_PK_REPORT
```

Negative-tested, because a gate that cannot fail is decoration: running with
`D1_ENV="BONSAI_Q8_0_X4=2"` produces

```
FAIL packed arm D1_ENV is missing BONSAI_PK_REPORT: [BONSAI_Q8_0_X4=2]
FAIL every packed arm carries BONSAI_PK_REPORT
=== host-side prerequisites: 1 failure(s) ===
```

That is the fifth assertion-vs-code drift in a day, and the second one closed by making the coupling
itself a tested property rather than a comment.

### Evidence hygiene: Stage 8's logs are archived

`LOGDIR` is fixed at `$WS/.stage8-logs`, so tonight's phases would overwrite Stage 8's raw logs under
the same tag names.

**The prescribed `cp -a` was not possible and was not attempted.** `.stage8-logs` is **79 GB** (47 GB of
it `device-tmp` from pulled device scratch) and the volume has **22 GB free** - a copy would have hit
ENOSPC part-way, leaving a truncated archive *and* a nearly full disk. It was archived by **rename**
instead, which is atomic, costs zero bytes and preserves every file:

```
mv .stage8-logs .stage8-logs-archive-stage8
mkdir -p .stage8-logs          # fresh LOGDIR for tonight
```

### The caps are checked before phase 0, and abort rather than wait

A hot start would burn up to ~120 x 10 s of `CAP_WAIT` per arm before the arm begins - the only real
schedule risk in a ~3.5 h plan. `preflight_caps()` now runs at the top of every device phase and
**aborts** (`exit 3`) unless `cpu0 = 3532800` and `cpu6 = 3840000`:

```
PREFLIGHT caps cpu0=<..> (want 3532800) cpu6=<..> (want 3840000)
PREFLIGHT ABORT: the device is still throttled. Let it cool and restart; do NOT let the
                 harness absorb this one arm at a time (that is ~120 x 10 s per arm).
```

It is skipped under `HOST_CHECK=1`, where there is no device to ask.

### Unchanged

`D1` and `C2` stay in the matrix, `C1` stays as a free live control, and the ship rule is untouched:
both mirrored passes positive on the targeted phase, neutral-or-better on the other, with no truncated
and no zero-hit arm admissible as evidence in either direction.

### The `BONSAI_Q8_0_X4=1` mis-classification, and why `hits=0` is correct there

`BONSAI_Q8_0_X4=1` (the Stage 7 `A6` variant) repacks **unconditionally** into the fixed per-thread
`g_q8_0_x4_buf` and **never touches the cache**:

```c
if (q8m_eff >= 2) pky = bonsai_pk_cached(src1_col, ...);      /* mode 2 -> g_pk_* counters */
else            { bonsai_pack_q8_0_x4(n, g_q8_0_x4_buf, ...); /* mode 1 -> no cache */
                  pky = g_q8_0_x4_buf; }
```

So a `=1` arm reports `hits = 0` **by design**. Judging it with the cache rule would reject a perfectly
healthy arm with a confident false INVALID - the exact mirror of the vacuous-pass problem, and the
sixth assertion defect of the day.

**What was actually wrong, precisely.** The analyzer already excluded `=1` (`PACKED_FLAGS` held only
`=2` and `PREFILL_PACKED=1`), so no false INVALID could have been emitted from the table; the defects
were that the **arm script** emitted an `S9PK` line for `=1` arms and the **gate** demanded
`BONSAI_PK_REPORT` on them, which would have failed a legitimate `=1` arm at the gate. Both are fixed,
and the premise is recorded as tested rather than assumed.

**One authority, and the consumer reads it.** The arm script decides the class:

| env | class | what it emits |
|---|---|---|
| `BONSAI_Q8_0_X4=2`, `BONSAI_MOE_X4_PREFILL_PACKED=1` (or both) | `cache` | `S9PK class=cache hits=.. misses=..` |
| `BONSAI_Q8_0_X4=1` | `uncond` | `S8PACK packs=N` only - **never** `S9PK` |
| anything else | `none` | nothing |

The class is written **into** the line (`class=cache`), and `packed_arm()` prefers that declaration over
re-deriving it from the env, so producer and consumer cannot drift. The env-based fallback survives only
for logs written before the field existed.

Verified directly at the shell level (the authority) and at the gate:

```
[<none>]                                           -> none (silent)
[BONSAI_Q8_0_X4=1]                                 -> uncond (S8PACK only, NO S9PK)
[BONSAI_Q8_0_X4=2]                                 -> cache (S9PK verdict applies)
[BONSAI_MOE_X4_PREFILL_PACKED=1]                   -> cache (S9PK verdict applies)
[BONSAI_Q8_0_X4=1 BONSAI_MOE_X4_PREFILL_PACKED=1]  -> cache (S9PK verdict applies)
```

The verdict table gained the explicit row, so `=1` can never be read as a cache verdict:

| arm state | verdict |
|---|---|
| cache-using, `hits > 0`, 6/6 | engaged |
| cache-using, `hits == 0` | INVALID (0 hits) |
| cache-using, no counter | INVALID (no counter) |
| **`BONSAI_Q8_0_X4=1`** | **`n/a (unconditional pack, no cache)`** |
| any arm, `n < meas` | INVALID (truncated 4/6) |
| not cache-using, 6/6 | n/a |

**The fixture is negative-tested.** An `A6` arm (`=1`, `hits=0`, 6/6) renders
`n/a (unconditional pack, no cache)`; forcing `=1` into the cache-using set flips exactly that row to
`INVALID (no counter)`, and both directions are asserted. My first version of that negative test passed
the *timing* table's row instead of the *validity* row - they share the `| decode | 1 | A6 |` prefix - so
the assertion now matches the 11-column validity shape explicitly.

### Preflight message

A low cap is now immediately distinguishable from a wrong app state:

```
FATAL caps not at profile: cpu0=2841600 (want 3532800) cpu6=2841600 (want 3840000)
  -> throttled, OR app not foregrounded in Game Space. Cool or foreground, then re-run.
  -> do NOT let the harness absorb this one arm at a time (~120 x 10 s per arm).
```

---

## Executor command: corrections, window order, and what is still open

### 1a. D1 confirmation arms added, and they run LAST

`phase1b_confirm()` in `s8_shootout.sh`, invoked at the end of phases `2` and `3`, after the timing
matrix:

| arm | lib | env | purpose |
|---|---|---|---|
| `prof-ours-A0` | `$LIB_PROF` | `BONSAI_PROF_EVERY=1` | reference |
| `prof-ours-D1` | `$LIB_PROF` | `BONSAI_Q8_0_X4=2 BONSAI_PK_REPORT=1 BONSAI_PROF_EVERY=1` | candidate |

Gated by the **same-engine** canary (`GET_ROWS`, `GATED_DELTA_NET`, 2.5%), never the cross-engine one:
same binary, same shapes, one flag apart. Implemented in the new
`MoeDroid/tools/s9_d1_confirm.py`, which prints both arms' op table, the canary verdict, and
`MUL_MAT_ID` before -> after with the delta. Exit 0 only on a passing canary, and a failure writes
`pair_D1(...)=FAIL` and states that the delta must not be published.

Both directions verified before it can gate anything:

```
# identical logs (self-check)
# worst same-engine canary |delta| = 0.00% (limit 2.5%)
S8D1_CONFIRM=PASS mul_mat_id_delta_pct=+0.00

# real mismatched pair (ours vs ik)
# worst same-engine canary |delta| = 99.50% (limit 2.5%)
S8D1_CONFIRM=INVALID (GET_ROWS 99.50%, GATED_DELTA_NET absent)
```

Ship bar unchanged and now fully scheduled: (1) both mirrored decode passes positive, (2) `@gN`
`pk_hits > 0`, (3) `MUL_MAT_ID` ms/token negative under a passing canary.

### 1b. Coordinates corrected, and the citation rule adopted

The `ggml-cpu.c:1367` citation was stale and is not a D1 hook. Re-verified against HEAD **now**:

| what | function | line |
|---|---|---|
| D1 hook, decode/single-token (dense) | `ggml_compute_forward_mul_mat_one_chunk` | **1494** |
| D1 hook, expert path | `ggml_compute_forward_mul_mat_id_one_chunk` | **2000** |
| batch/paired-token cache reads | `_mul_mat_id_one_chunk` pre-loop | 1933-1934 |
| PK cache machinery | `bonsai_pk_invalidate` / `bonsai_pk_cached_tag` | **1229** / **1253** |
| packed 4-row kernel body | `bonsai_x4_rows4_mode1_packed` | `iqk_x4.c` **482** |
| its wrapper | `bonsai_vec_dot_q2_0_q8_0_x4_packed4` | `iqk_x4.c` **618** |

**Rule from here on: every citation is FUNCTION NAME plus line, re-verified against HEAD immediately
before filing.** Line numbers drift as the file moves; four stale digits have now been caught
(`17.31`, `:1367`, `<=26`, and the `:1930`/`:1429` pair which had already moved to `2000`/`1494` by the
time this was checked), and each was correct when written.

### 2. Host track - status, stated plainly

* **2a `Gate 3(b)` on `BONSAI_MOE_X4=2` (`BONSAI_MULMATID_FLAT=2`)**: **not launched.** It is an
  overnight job (two model loads plus 32 chunks on the 176B model) and belongs at the top of the device
  window; the window has not opened. Nothing is blocked on it, but nothing is gained either until it
  starts.
* **2b census:**
  * **f32 `MUL_MAT` call count: derived as 130 (833 - 703), NOT independently counted.** The archived
    per-type table stores milliseconds only (`types` -> `{f32: 6498.95, ...}`), so the count cannot be
    recovered from existing logs. Confirming "exactly 130" needs two things I did not do: a per-type
    **call** counter in the profiler dump (`BONSAI_PROF`, a few lines) and one device run to produce it.
    Until then **Row 2 stays unestablished** by the plan's own rule.
  * **174 `GET_ROWS` by tensor name and shape:** attributed by builder in Task 3 (96 QSA pool selector
    + 48 MoE group selector + ~30 embeddings/PLE/last-layer crops/MTP), but **not** enumerated
    per-tensor-name. That enumeration is still the Phase 2 precondition.
  * **Row-5 residual (HC/CPY/CONCAT after f32 attribution):** not computed.

### 3. Window order (unchanged, with the addition)

```
preflight caps gate (refuses a hot start, exit 3)
  -> s8_shootout.sh 0    identity gates, per-arm triage (abort only on A0)
  -> s8_shootout.sh 3    mirrored timing passes
  -> phase1b_confirm     prof-ours-A0, prof-ours-D1 (LAST; canary-gated)
```

### 4. Document corrections

* **Ledger:** "MATH-INSTRUCTIONS" is not a currency. D3 removes **STACK-SPILL LOADS + issue-slot
  pressure**; it pays **REGISTERS + SCRATCH BYTES**.
* **"~0 spills" in Phase 3 is aspirational**, not a fact: recorded unpacked mode-2 is 16/17 (STAGE7
  B.5) and the packed+lean variant has never been counted. The host `clang -S` check is restored as a
  **precondition**, and the measured post-change spill count is required as **evidence** - not a veto.
* **Structural stop rule:** kept, plus the **coupling probe** - count the external symbols each ik
  kernel pulls in before any window is scheduled against the transplant. It is a port, not a copy.
* **DEFERRED (with owners):**
  (a) `MADV_DONTNEED` premise test - dense-class eviction instrumented, not just bytes freed.
  (b) expert-chunking arm - pending the D2 grep.
* **Header line for v3:** "Land 8.2-8.5 t/s decode with full attribution; the structural rule explains
  the remainder." So nobody reads Phases 1-4 as 9.24 by default.

---

## Per-TYPE call counts + bytes: the piece that makes Row 2 decidable

The archived per-type table stored **milliseconds only**, so the f32 call count could not be recovered
from disk. The `BONSAI_PROF` dump now carries, per weight type:

```
  MATMUL f32               5.76 ms total    0.120 ms/graph      130 calls (mm=130 mmid=0)      1234480 bytes
```

i.e. **calls, split by op** (`mm` for `MUL_MAT`, `mmid` for `MUL_MAT_ID`) and **weight bytes touched**,
both keyed on `src0->type` and accumulated in `ggml_compute_forward_mul_mat_one_chunk`'s timing block.
No new device arm is needed: the two profiler arms already scheduled tonight (`prof-ours-A0`,
`prof-ours-D1`) deliver the census.

**What it decides.**

| f32 `MUL_MAT` calls | consequence |
|---|---|
| **exactly 130** | Row 2 is **count-certified work-equivalent** (833 - 130 = 703 = ik's `MUL_MAT` count) and Phase 4 gains solid footing |
| anything else | Row 2 stays **unestablished** and the plan's certified deficit drops to 49% |

**The derived column, and the three-way verdict.** The analyzer prints `calls/token`, `KiB/token`,
`us/call` and `GB/s` per type, then a verdict block:

```
## three-way verdict per type (the falsification test for the decomposition)
  counts+bytes both match the other engine -> certified kernel deficit, phase target stands
  counts differ                            -> ours does different WORK, not slower work
  type          calls/tok   bytes/tok  us/call  verdict
  f32              130.00     1234480    15.38  measure ik's row to classify
```

That is the falsification test for the whole decomposition, and it runs from tonight's existing data
once ik's log is read the same way. `us/call` is the number that distinguishes "same work, our kernel is
slower" from "we are doing more calls" - the distinction the Stage 8 op-count comparison could not make.

**Verification actually performed** (not asserted):

* an **old** log still renders the 3-column table (no counts -> graceful degradation, no crash);
* a **two-dump synthetic** log in the real dump format renders `calls/tok = 130.00`, bytes, `us/call`
  and the verdict block - the numeric path, not just the column headers;
* the first synthetic attempts read 0.00 because I gave every dump the *same* counter value, so the
  window delta was zero: the counters are cumulative and the columns are deltas. Worth recording, since
  a zero column would otherwise have looked like a parser bug.

**One pre-existing crash fixed in passing:** a log whose window has no delta (identical or degenerate
dumps) hit `ZeroDivisionError` in the t/s line. It now refuses to divide and says so:

```
## total compute = 0 ms/token: the window has no measurable delta (identical or degenerate dumps)
```
