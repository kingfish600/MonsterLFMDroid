# Stage 6 daytime prep (host-only, no phone, no adb)

Date: 2026-10-16 (daytime). Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite, 24 GB, CPU only.
**The phone is away for 12 hours. Nothing in this document was measured on the device.** Every
device number quoted here is carried over from Stage 3/4/5 and is labelled as such.

This file is written incrementally, one section per completed step, in the order the session
override specified. Sections are appended as they land, not batched at the end.

| step | scope | status |
|---|---|---|
| 0 | host environment and what it enables | done |
| 1 | resolve the `+0.092%` vs `+0.290%` PPL discrepancy and re-check `varD4` | done |
| 2 | `ik_llama.cpp` gap attribution: source-level call-path map | done |
| 3 | `GOAL.md` + `iterations.jsonl` (goal-md convention, Stages 1-5 backfill) | done |
| 4 | Candidate B: `ik_llama.cpp` affinity flags, `BONSAI_PROF` port, tonight's harness | see section |
| 5 | Candidate A: highest-leverage piece ported into `upstream-llama` | done |

---

## 0. Host environment, and one correction to the brief

The brief for this session says `qemu-aarch64-static` is not installed. **That is no longer true,
and nothing else in this document depends on it being true.** It was not installed as a system
package (there is no usable sudo on this host: `/etc/sudo.conf` is owned by uid 65534 and
`no_new_privs` is set), but the Ubuntu package was fetched and unpacked into the workspace:

```
apt-get download qemu-user-static          # works as an unprivileged user
dpkg-deb -x qemu-user-static_1%3a8.2.2+ds-0ubuntu1.18_amd64.deb ext
# -> ext/usr/bin/qemu-aarch64-static   (qemu-aarch64 version 8.2.2)
```

Statically linked Android aarch64 binaries then run directly on the host, and every NEON feature
the project's kernels use is emulated correctly:

| feature | probe program under qemu-aarch64-static | result |
|---|---|---|
| `vdotq_s32` (dotprod) | `dotprod: 60 60 60 60` | works |
| `vmmlaq_s32` (i8mm) | `i8mm: 120` | works |
| `vbfdotq_f32` (bf16) | `bf16: 6.000000` | works |

This matters because it is the only way to test *real* ARM64 kernel code on the host: the kernel is
compiled by the NDK with the shipping flags (`-march=armv8.2-a+dotprod+i8mm+fp16+bf16`, static), so
the instructions under test are the instructions the device runs. It is used as the verification
method in section 5. QEMU timing is meaningless and was never used for performance claims.

Other environment facts, for reproducibility:

| item | value |
|---|---|
| host | 32 cores, 124 GB RAM, 112 GB free on `/`, network egress available |
| NDK | `/home/kingfish600/workspace/tools/sdk/ndk/28.2.13676358` |
| ninja | `/home/kingfish600/workspace/tools/ninja-bin/ninja` |
| existing build dirs | `upstream-llama/build-arm64-ship` (ninja), `build-arm64-prof` (Unix Makefiles, `-DBONSAI_PROF=1`), `build-x86`; `ik_llama.cpp/build-arm64-ik` (Unix Makefiles) |

**Operational trap found the hard way, worth recording:** in this harness every `bash` invocation
gets a *private* `/tmp`, so a file written to `/tmp` by one command or one subagent is invisible to
every other. All scratch and handoff artefacts must live inside `/home/kingfish600/workspace`.
Two delegated workstreams had to be redirected mid-flight because of this.

---

## 1. Step 1: the perplexity-recovery discrepancy (`+0.092%` vs `+0.290%`), and `varD4` re-check

### 1.1 What the two stages did

Stage 2B and Stage 3 record the dense rewrite's quality cost as `baseline -> varH = +0.092%`
(32 chunks, `t = 1.02`). Stage 5 section 2.2 recovered per-chunk perplexity with the correct
formula and reported the same comparison as `+0.290% overall / +0.300% paired arithmetic
(t = 1.18)`, calling the difference with Stage 2B "unresolved".

Both numbers come from the same kind of log: `llama-perplexity` prints `[i]` followed by
`exp(nll/count)` with **cumulative** `nll` and `count`, so the printed value after chunk `i` is the
running estimate over chunks `1..i`, not the perplexity of chunk `i`. The correct per-chunk value is
`c_i^i / c_{i-1}^{i-1}`.

### 1.2 The measurement

Tool: `MoeDroid/tools/s6_ppl_recheck.py` (new, reproducible), output saved to
`.stage6/ppl-recheck.txt`. Per-chunk perplexity is recovered as `ppl_i = c_i^i / c_{i-1}^{i-1}`
with `c_0 = 1`; for equal-sized chunks this is exact and the chunk token count cancels. Three
estimators are reported so they can never be confused again: the ratio of final estimates, the
32-chunk paired mean of per-chunk ratios (the project's gate statistic), and the per-token NLL delta
`mean(ln ppl_new - ln ppl_base)` in nats. The naive "difference the printed cumulative values"
estimator is also printed, purely to identify it.

First, the surviving logs. There are **more PPL logs than the brief assumed**, and the varH run used
for Stage 2B's table is *not* the same file as `ppl32-new.log`:

| log | chunks | final cumulative PPL | what it is |
|---|---|---|---|
| `ppl32-orig.log` | 32 | **2.8952** | original shipping shard, sha256 `69820c02...` |
| `varH-ppl.log` | 32 | **2.9036** | varH (the number in Stage 2B's table) |
| `ppl32-new.log` | 32 | **2.9253** | the *full* rewrite (varH + `output.weight`) |
| `varO-ppl.log` | 32 | 2.9174 | `output.weight Q5_K -> Q4_0` only |
| `varD4-ppl.log` | 32 | 2.9214 | `q3_K`+dense `q2_0` -> `q4_K` |
| `varD5-ppl.log` | 32 | 2.9002 | -> `q5_K` |
| `varD8-ppl.log` | 32 | 2.9067 | -> `q8_0` |
| `ppl.log`, `ppl-full-{orig,new}.log` | 8 | 2.3692 / 2.3550 / 2.3692 | 8-chunk runs on a different setup; **not** comparable to the 32-chunk table |

Results (`n = 32` for every comparison):

| comparison | overall (ratio of finals) | paired mean | sd | SE | t | chunks worse | NLL delta (% of baseline) | naive estimator (WRONG) |
|---|---|---|---|---|---|---|---|---|
| varH vs orig | **+0.290%** | **+0.300%** | 1.439 | 0.254 | **+1.18** | 17/32 | +0.273% | +0.116% |
| full rewrite vs orig | +1.040% | +1.060% | 2.033 | 0.359 | +2.95 | 22/32 | +0.973% | +0.849% |
| **varD4 vs varH** | **+0.613%** | **+0.628%** | 1.771 | 0.313 | **+2.01** | 19/32 | +0.573% | +0.957% |
| **varD4 vs orig** | **+0.905%** | **+0.921%** | 1.823 | 0.322 | **+2.86** | 23/32 | +0.847% | +1.072% |
| varD5 vs varH | -0.117% | -0.107% | 1.425 | 0.252 | -0.43 | 15/32 | -0.110% | -0.339% |
| varD5 vs orig | +0.173% | +0.186% | 1.632 | 0.288 | +0.64 | 20/32 | +0.162% | -0.224% |
| varD8 vs varH | +0.107% | +0.115% | 1.313 | 0.232 | +0.50 | 14/32 | +0.100% | **+0.579%** |
| varD8 vs orig | +0.397% | +0.407% | 1.441 | 0.255 | +1.60 | 20/32 | +0.373% | +0.693% |
| varO vs varH | +0.475% | +0.497% | 2.113 | 0.374 | +1.33 | 23/32 | +0.445% | +0.674% |

### 1.3 Verdict

**Stage 5 is right; Stage 2B's `+0.092%` is not reproducible, and the true cost of varH is about
three times larger than recorded - but it still passes the gate.**

1. My recomputation reproduces Stage 5's numbers **exactly**: varH = `+0.290%` overall / `+0.300%`
   paired (t = 1.18); varD4 = `+0.613%` / `+0.628%` (t = 2.01); varD5 = `-0.117%` / `-0.107%`
   (t = -0.43); varD8 = `+0.107%` / `+0.115%` (t = 0.50). The method in Stage 5 section 2.2 is
   correct and is now the project's method.
2. **The flawed estimator is identified beyond doubt.** The naive cumulative-difference estimator
   gives `+0.579%` for varD8 - the exact figure Stage 5 recorded as the wrong answer (the correct
   value is +0.115%). So that estimator is confirmed wrong, and it overstates by 2-5x.
3. **Stage 2B's `+0.092%` (SE 0.091%, t = 1.02) cannot be reproduced from any surviving log.** Its
   listed baseline PPL of **2.7680 exists in no log anywhere in the workspace** - every 32-chunk
   baseline log gives 2.8952. Against the surviving baseline, none of the three correct estimators
   yields +0.092%: they give +0.290% (overall), +0.300% (correct paired) and +0.273% (NLL). The only
   value anywhere near +0.092% is the naive estimator's `+0.116%`, which is the estimator proven
   wrong in point 2. Stage 2B's "26/32 chunks worse" also does not reproduce (it is 17/32).
4. **The discrepancy therefore cannot be fully closed, and that is the honest answer:** the baseline
   series Stage 2B paired against is no longer on disk, and its printed final value (2.7680)
   contradicts every surviving baseline log. What *can* be stated is that Stage 2B's number is not
   supported by any evidence that survives, while Stage 5's is exactly reproducible.
5. **It does not change the ship decision, but it does shrink the margin.** varH costs
   **+0.30%** paired against the original shard, not +0.09%. That is 1.7x under the `<= 0.5%` gate
   rather than 5x under it, so varH still passes - and it is banked on an on-device measurement
   (+8.19% decode, t = 7.09, 4/4 pairs) that no perplexity question touches.
6. **`varD4` is confirmed rejected, from both references.** `+0.628%` paired vs varH (t = 2.01,
   19/32 worse) and `+0.921%` vs the original shard (t = 2.86, 23/32 worse), with the NLL view
   agreeing (+0.573% and +0.847%). The effect is roughly twice the gate and is statistically
   resolved, so the predicted +5.8% decode does not buy it back. It was also never run on device,
   so there is no evidence the speed prediction holds either.
7. **A gate-hygiene point that fell out of this:** the Stage 5 table judges `varD{4,5,8}` *against
   varH*. Against the **original shipping shard** - the reference a user who never adopted varH
   actually has - all three are worse: varD4 `+0.921%`, varD8 `+0.407%`, varD5 `+0.186%`. So varH
   remains the best quality-per-byte point of the four, and any future requant must state which
   reference its delta is against. `varD5`'s `-0.117%` is "marginally better than varH", not
   "better than the model we started from".

---

## 2. Step 2: source-level attribution of the `ik_llama.cpp` gap

Full 868-line evidence document with a `file:line` for every claim:
`.stage6/stage6-task2-attribution.md`. This section is the decision-relevant digest; I re-verified
the four load-bearing claims myself (marked **[verified]**) rather than trusting the delegated pass.

Five premises in the session brief are wrong, and they change what the port should target:

| brief says | source says |
|---|---|
| trace `iqk_mul_mat_moe` / `iqk_mul_mat` dispatch in `ggml/src/ggml-cpu/` | ik has **no `ggml/src/ggml-cpu/`**; it is the old monolithic layout, all CPU code is in `ggml/src/ggml.c`. Entry points are `ggml_compute_forward_mul_mat_id` (`ggml.c:18590`) and `ggml_compute_forward_mul_mat_id_up_gate` (`ggml.c:18909`) |
| `mul_mat_q2_0_q8_2_x4` is the aarch64 path | it exists but is **x86-only** (`#ifdef __x86_64__`, `iqk_gemm_legacy_quants.cpp:2980`). aarch64 uses `mul_mat_q2_0_q8_0_x4` (`:4511`, registered `:4636`) |
| `GGML_IQK_MUL_MAT` picks kernels per type at runtime | there is **no runtime feature dispatch in iqk at all** (no `ggml_cpu_has_*` in the iqk tree); `IQK_IMPLEMENT` and the q8_0-x4 vs q8_2 choice are compile-time (`iqk_config.h:13`, `ggml.c:689-693`) |
| the fork's `vbfdotq_f32` is a differentiator to defend | ik has **no `vbfdotq_f32` anywhere**; its bf16 path widens by shift plus `vfmaq_f32` (`iqk_gemm_floats.cpp:776,779`). Our bf16 kernel is the better one; ik is not winning on bf16 |
| a fused `GATED_DELTA_NET` is a candidate | ik's op is named `GGML_OP_DELTA_NET` (`ggml.h:698`) and it **is** fused with a NEON kernel (`ggml.c:24579` -> `iqk_mul_mat.cpp:1791`); our fork also has a fused op but a generic kernel. So this is a kernel-quality difference, not a graph-structure one |

### 2.1 The largest single contributor needs no iqk code at all

For `varH`, ik emits **one fused `MOE_FUSED_UP_GATE`** for the up and gate expert projections,
where our fork emits **two `MUL_MAT_ID`**. Three expert matmuls per layer become two. **[verified]**

* ik: `if (can_use_fmoe && lctx.cparams.fused_moe_up_gate && up_exps->type == gate_exps->type && ...)`
  -> `ggml_moe_up_gate(...)` (`ik src/llama-build-context.cpp:1742-1756`), else the two `llm_build_lora_mm_id` calls.
  The parameter defaults to `true` (`ik src/llama.cpp:8891`, `common/common.h:456`).
* ours: the separate-gate-and-up path calls `build_lora_mm_id(up_exps, ...)` and then
  `build_lora_mm_id(gate_exps, ...)` (`upstream-llama/src/llama-graph.cpp:2338` and `:2351`).
  The merged-tensor branch above it (`ggml_view_3d` over `gate_up`) does not apply to this model.
* our fork has **no** `MOE_FUSED_UP_GATE` symbol anywhere (`grep` over `src/` and `ggml/include/`). **[verified]**
* the type gate `up_exps->type == gate_exps->type` is satisfied: all three `ffn_*_exps` tensors in
  varH are `Q2_0` (type 42), and there is no merged `gate_up` tensor in the GGUF, so this fusion is
  the only available route and it works on separate tensors.

Expert `MUL_MAT_ID` is **45.9% of decode** and **61.9% of prefill** (Stage 4), and it is the bucket
where the fault-attribution also lands, so a structural change here is aimed at the largest target.

### 2.2 `GGML_EXPERT_CHUNKING` runs at every token count; our equivalent does not

ik's chunking is enabled in the actual build (`GGML_EXPERT_CHUNKING:BOOL=ON` in
`build-arm64-ik/CMakeCache.txt`, and present in the real `compile_commands.json`). **[verified]**
It sets `chunks_per_expert = MAX(1, MIN(nth, ne01/32))` and claims work from **one atomic queue over
(expert x chunk)** (`ggml.c:18738`), and it applies at all token counts.

Our `BONSAI_MULMATID_FLAT` is **decode-only** (gated on `ids->ne[1] == 1`, `ggml-cpu.c:1864`) and
uses fixed half-expert units. This asymmetry is the best available explanation for why ik's prefill
advantage (+43%) exceeds its decode advantage (+25%): prefill is 61.9% expert work and is the case
our flat path does not cover at all.

### 2.3 Where the `n = 1` decode speedup comes from

At `n = 1` there is exactly one activation row, so `nrc_y = 1` and the `_x4` structure cannot share
anything on the `y` side *across tokens*. The candidates in the brief, answered from source:

| candidate | verdict |
|---|---|
| `GGML_EXPERT_CHUNKING` | **contributes** - one global queue over expert x chunk at all token counts (§2.2). This is the structural difference, not a kernel one |
| `q8_2` activation quantisation | **not on aarch64.** The q8_0-x4 vs q8_2 choice is compile-time x86-gated (`ggml.c:689-693`); aarch64 uses `mul_mat_q2_0_q8_0_x4` |
| runtime kernel selection | **does not exist** in ik; everything is compile-time |
| fused `DELTA_NET` | real but small: our `GATED_DELTA_NET` is 2.9% of decode (Stage 4), so even a perfect kernel there is worth <3% |
| `iqk/fa` flash attention | active for `-fa on` (`ggml.c:23467`, needs F32 q / F16 mask), but attention is 1.5% of our decode. Bounded small |
| fused up+gate (§2.1) | **yes, and it is in the largest bucket** |
| activation-load sharing inside the kernels | **yes** - this is what the `_x4` shape actually buys at `n=1`: the quantised activation resident across several weight rows instead of reloaded per row |

### 2.4 The arithmetic bound, and the number that actually reconciles

The two engines measured **6.90 vs 8.61 t/s** decode and **8.40 vs 11.98 t/s** prefill
(Stage 5 section 1.4, matched flags, same model file, q8_0 KV both sides). **Both oracle figures are
floors, not point targets** - ik ran them with its default 32 context checkpoints while our fork ran
`-ctxcp 0`, so ik was carrying overhead our engine was not; see section 12.4. The port plan's
"~+65% decode" headline came from an earlier, **unmatched** run (f16 KV on ik, no `q8_0` matched
re-run) and is superseded; the port plan's own caveat says so. Use +25% / +43%.

Because our decode is 86.8% the two matmuls, at least **6.66 of the 19.86 percentage points** of
total time reduction must come from the matmul bucket, i.e. that bucket must run **1.08x-1.30x**
faster. So the gap is in the matmuls, and the useful decomposition is: fused up+gate (graph, no iqk),
all-token chunking (scheduling, no iqk), and activation-sharing kernels (the `_x4` shape) - in that
order of cost to implement.

### 2.5 The call-path map, condensed

One `varH` (`qwen4exp`) forward pass, from the graph walk down to the aarch64 kernel, with the
citation for each edge. Everything here is from `.stage6/stage6-task2-attribution.md`; the four
claims marked **[verified]** in §2.1/§2.2 were re-checked directly.

```
llama_decode -> llama_context::decode -> ggml_backend_sched_graph_compute
  ggml_graph_compute                                   (ik ggml/src/ggml.c:29625)
    #pragma omp parallel num_threads(n_threads)        (ik ggml.c:29658)   <- ik's compute is OpenMP
    per node:
      ggml_compute_forward(&params, node, cgraph, node_n)   (ik ggml.c:29601)
      |
      +-- GGML_OP_MUL_MAT_ID  (routed experts, weight type Q2_0, 144 calls/token in our profile)
      |     ggml_compute_forward_mul_mat_id            (ik ggml.c:18590)
      |       +-- MOE_FUSED_UP_GATE (up+gate in ONE op, 3 expert ops/layer -> 2)   [verified]
      |       |     ggml_compute_forward_mul_mat_id_up_gate   (ik ggml.c:18909)
      |       |     emitted by llama-build-context.cpp:1742-1756 when
      |       |       fused_moe_up_gate (default true, ik src/llama.cpp:8891,
      |       |       common/common.h:456) AND up_exps->type == gate_exps->type (both Q2_0)
      |       |     CLI switch to disable it: --no-fused-moe   (ik common/common.cpp:2251)
      |       +-- GGML_EXPERT_CHUNKING work distribution       [verified]
      |       |     chunks_per_expert = MAX(1, MIN(nth, ne01/32)), ONE atomic queue over
      |       |     (expert x chunk) -> ggml.c:18738; applies at ALL token counts
      |       |     (our equivalent, BONSAI_MULMATID_FLAT, is gated ids->ne[1]==1)
      |       +-- dispatch predicate for the optimised path: ggml.c:18705
      |       |     ne13==1 && dst->type==F32 && ne00%32==0 && typeB==GGML_TYPE_Q8_0_X4
      |       +-- aarch64 kernel: mul_mat_q2_0_q8_0_x4
      |             iqk_gemm_legacy_quants.cpp:4511 (registered :4636)
      |             (mul_mat_q2_0_q8_2_x4 exists but is x86-only, :2980)
      |
      +-- GGML_OP_MUL_MAT  (dense backbone)
      |     ggml_compute_forward_mul_mat -> per-type iqk kernels:
      |       q3_K  -> the iqk_gemm_kquants.cpp template family
      |       q2_0  -> mul_mat_q2_0_q8_0_x4 (same kernel as the expert path)
      |       q4_K/q5_K/q6_K, q8_0, iq4_xs -> their iqk template families
      |       bf16  -> iqk_gemm_floats.cpp:776,779 (shift + vfmaq_f32; NO vbfdotq_f32 in ik)
      |       F32   -> NOT in the aarch64 float registry (iqk_gemm_floats.cpp:1103) -> stock path
      |     no runtime feature dispatch anywhere in iqk; IQK_IMPLEMENT/config is compile-time
      |     (iqk_config.h:13), and q8_0-x4 vs q8_2 is chosen by #ifdef __AVX2__ (ggml.c:689-693)
      |
      +-- GGML_OP_DELTA_NET (fused Gated DeltaNet; ik name is DELTA_NET, ggml.h:698)
      |     ggml.c:24579 -> iqk_mul_mat.cpp:1791
      +-- GGML_OP_FLASH_ATTN_EXT (iqk FA, active for -fa on: needs F32 q / F16 mask)
      |     ggml.c:23467 -> iqk/flash_attn... (iqk_flash_attn.cpp:167)
      +-- GET_ROWS / CPY / CONCAT / MUL / UNARY / RMS_NORM  -> stock ggml paths
```

Wrong-premise corrections that the map makes obvious (all in §2): there is no
`ggml/src/ggml-cpu/` in ik and no `iqk_mul_mat_moe`; the q8_2 path is x86-only; there is no runtime
kernel selection; ik has no `vbfdotq_f32`; and `nrc` cannot be the whole `n = 1` story because with
one activation row there is nothing to share on the `y` side across tokens - the sharing that
remains is *within* a call (across weight rows, and across the fused up+gate pair).

### 2.6 Deliberately not verified

* Whether the Stage 5 ik benchmark passed `--prefetch-experts` (default false, `llama.cpp:8888`).
  If it did not, ik's MoE prefetch engine contributed nothing and is not part of the measured win.
  **Tonight's harness logs the engine's own configuration lines to settle this** (§5.4).
* Whether any of this generalises to the aarch64 kernels beyond what `q2_0` shows - the x86 build
  compiles different code paths, so all aarch64 conclusions come from source plus the qemu test.

---

## 3. Step 3: `GOAL.md` and `iterations.jsonl`

Delivered. `MonsterQwenDroid/GOAL.md` (318 lines) and `MonsterQwenDroid/iterations.jsonl`
(35 entries, all parse, `iteration` 1-35 sequential: 15 kept, 20 closed, 0 reverted).

The `jmilinovich/goal-md` convention (repo `8f39b8ab`) defines one JSON object per line with core
keys `iteration`, `before`, `after`, `action`, `result` (`kept|reverted`), `note` - as a
non-normative example at `template/GOAL.md:117-126`; there is no JSON-Schema validator in the repo.
Those core keys are used verbatim and extended with the fields this project's gates need:
`date, stage, hypothesis, change, verdict, evidence (file:line), protocol, numbers{}`.

`GOAL.md` records both measurement protocols as first-class methodology:

| protocol | description | value on varH, shipping flags |
|---|---|---|
| arm protocol | fresh model load, one warmup request, one measured request (`s3_arm.sh`) | 5.2264 t/s |
| long-lived server steady state | one load, many repeated requests | 6.90 t/s |

and the rule that follows from it: **every t/s number must name its protocol**; arm-protocol
absolutes understate steady-state decode by ~25-30%, and relative results remain valid only within a
stage. The gates recorded are: kernel changes require greedy token identity (`--temp 0` md5);
requantisation requires 32-chunk paired PPL <= +0.5% using the per-chunk recovery `c_i^i / c_{i-1}^{i-1}`;
speed requires mirrored blocks on a matching cap profile; models are staged on `/sdcard`, never
`/data/local/tmp`.

### 3.1 Discrepancies the backfill surfaced

1. **The `+0.092%` vs `+0.290%` varH perplexity delta** - resolved in section 1.
2. Oracle gap magnitude: `+25%/+43%` (matched flags, Stage 5 section 1.4) vs `~+65%` decode in the
   port plan (§2.4). The matched run governs.
3. `STAGE5-report.md:17` says `varD8` is `-4 to -6%` while its own section 2.3 measures `-3.87%`
   paired / `-3.89%` group. The log uses the measured value.
4. **The session brief conflated two unrelated results** as "`varO` / MTP speculative decoding":
   `varO` is `output.weight Q5_K -> Q4_0` (-75.8 MiB, **+0.814%** PPL, t=9.52, fails);
   MTP speculative decoding is the separate 4x-slower sidecar result (3.94 -> 0.95 t/s, 77.5%
   acceptance, majflt 1,130 -> 6,100/token). Now logged as two separate iterations.
5. `docs/PERFORMANCE.md` carries a pre-varH op budget (dense 116 ms, experts 93 ms) against
   Stage 4's measured 173.2 ms/token. Stale, not contradictory; noted in `GOAL.md`.

Also correct in the deliverables: ik's `Q2_0` is the 18-byte block (not the brief's 10-byte; the
34-byte type is ik's separate `PQ2_0`, `QK=128`), and our `ggml_vec_dot_q2_0_q8_0` is already
hand-written NEON with a dual-accumulator unroll (`arch/arm/quants.c:222`), not the brief's
"14-op scalar loop".

---

## 4. Step 4: Candidate B, `ik_llama.cpp` affinity and profiling

### 4.1 Why ik cannot be fixed with the flag we used on our own fork

`ik_llama.cpp` has **no mainline threading layer at all**: no `common_cpu_params`, no
`postprocess_cpu_params`, no `ggml_threadpool` (zero grep hits). Its compute is **OpenMP**
(`#pragma omp parallel num_threads(n_threads)`, `ggml.c:29658`; the build has `GGML_USE_OPENMP` with
`-fopenmp=libomp`, and `libggml.so` NEEDs `libomp.so`).

That has two consequences, and they are the opposite of what the Stage 5 report assumed:

* **Affinity was always applied, per worker.** `set_cpu_thread_affinity(cplan, state->ith)` pins
  worker `ith` to `cpu_affinity[ith % n]` (`ggml.c:29578`, impl `:29030`). So the bug that cost our
  fork +21.5% prefill - `-tb` silently leaving the batch pool with an all-zero, unapplied mask - does
  **not** exist in ik. What ik lacked was the ability to give the two pools *different* masks.
* **Correction to Stage 5 section 1.3**: "ik does not implement `--cpu-strict` ... so it cannot take
  the batch-pool pinning that buys our fork +21.5% prefill" is half right. ik's single affinity list
  was applied strictly to both pools, so its *decode* was already pinned to `0x3f`; what it could not
  do is run decode on `0x3f` and prefill on `0xff` simultaneously. That specific lever was ours
  alone, which means the Stage 5 "best vs best" claim was not quite best-vs-best on prefill.

### 4.2 What was added

| flag | status | notes |
|---|---|---|
| `--cpu-mask` / `-cm`, `--cpu-range` / `-cr`, `--cpu-affinity` | already existed | one list for both pools (`common/common.cpp:2420-2446`) |
| `--cpu-strict` | **added** | default is **1** in ik (its historical behaviour), against mainline's 0 |
| `-cmb` / `--cpu-mask-batch`, `-crb` / `--cpu-range-batch`, `--cpu-strict-batch` | **added** | batch list carried on `llama_context_params` (`include/llama.h:536-543`), selected per graph (`llama.cpp:7230,7760,8158`) |
| `--poll`, `--poll-batch` | **added, with a caveat** | ik has no hand-rolled spin loop, so this cannot be a literal port: `poll 0` -> `KMP_BLOCKTIME=0` + `OMP_WAIT_POLICY=PASSIVE`, `>=100` -> `-1`/`ACTIVE`, otherwise `poll*2` ms, plus `kmp_set_blocktime()` via `dlsym` |
| non-strict whole-mask pinning | **added** | mirrors mainline `ggml_thread_cpumask_next` (`ggml.c:29043-29070`) |

Batch-mask propagation with `-tb` is **unconditional** in ik (it is never gated on the thread count,
`common/common.cpp:1044`), plus a token-list fallback at init (`llama.cpp:9819-9829`). This was
proven, not argued:

* a **host build plus a purpose-written test** shows `-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1
  --cpu-mask-batch 0xff --cpu-strict-batch 1 --poll 0` -> token pool `n=6 [0-5]`, batch pool
  `n=8 [0-7]`, strict 1/1; the control `-t 6 -tb 8 --cpu-mask 0x3f` -> batch `n=6 [0-5]`, never empty
  (`.stage6/host-proof-pinning.txt`);
* at kernel level, after a real ggml compute with mask `{0,1,2,3}`, `/proc/self/task/*/status` shows
  `strict=0` -> every worker `0-3`, `strict=1` -> workers split `1`, `2`, `3`.
* Caveat carried forward: only the **per-graph pool switch** rests on code-path reasoning (there is
  no host model to run); everything about mask parsing, inheritance and the kernel affinity itself is
  measured.

### 4.3 The `GOMP_CPU_AFFINITY` / `OMP_PLACES` / `taskset` route, compared honestly

The override asks for this comparison explicitly. It is the *wrong tool for this job*, and the reason
is structural rather than a matter of taste:

| route | what it can express | verdict for tonight |
|---|---|---|
| `taskset -c 0-5` (or `sched_setaffinity` on the process) | **one** CPU set for the whole process, inherited by every OpenMP worker | cannot express the measured optimum, which is decode on `0x3f` **and** prefill on `0xff` in the same run. It would also have to be undone for the batch phase, which is not switchable per graph |
| `GOMP_CPU_AFFINITY="0-5"` / `OMP_PLACES=...` / `KMP_AFFINITY` | one placement policy per process; it *overrides* the per-worker `set_cpu_thread_affinity` that ik already applies | same limitation, plus it silently fights ik's own pinning. Useful only as a fallback for an **unpatched** ik, which then cannot do the 0xff batch pool at all |
| `OMP_WAIT_POLICY` / `KMP_BLOCKTIME` | the OpenMP spin-wait behaviour, i.e. the closest thing ik has to our `--poll 0` | **this is the one worth testing**, and it is exactly what the new `--poll` flag maps to |
| ik's native flags (new) | independent masks and strictness per pool, selected per graph | **the route to use.** It is the only one that can reproduce the shipping configuration |

So the harness uses the native flags and treats the environment variables as a *control*: setting
`OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` externally must be **equivalent** to `--poll 0`, and that
equivalence is worth one cheap arm tonight because it tells us whether the OpenMP spin-wait costs
anything on this device. `GOMP_CPU_AFFINITY` / `taskset` are documented here but are not used in the
shootout, because pinning both pools identically makes the two engines' flag sets incomparable.

### 4.4 `BONSAI_PROF` ported into `ik_llama.cpp` (`build-arm64-ik-prof`)

The port plan's step 1 is "profile first, do not write a kernel until the gap is attributed", and
until today only *our* engine had a per-op timer. ik now has the same timer, so tonight's two budget
tables can be differenced op by op on the same device.

Design (in `ik_llama.cpp/ggml/src/ggml.c`, guarded by `#if BONSAI_PROF`, runtime-gated by
`BONSAI_PROF=1`, `BONSAI_PROF_EVERY=N`):

* ik's per-node dispatch is a single call site (`node_n = ggml_compute_forward(&params, node, cgraph, node_n)`,
  `ggml.c:29601`) inside `ggml_graph_compute_thread`, which runs on **every** OMP worker. Timing that
  call is therefore the op's compute time, and it folds any sub-graph an op expands into the parent
  op - which is what we want for ik's fused ops.
* Each worker accumulates into **its own row** `g_bp_ns[ith][op]` (no atomics, no lock, no race); the
  dump sums the rows. Only thread 0 dumps, **after** the loop, and the loop's last action per node is
  `ggml_barrier(state->shared)` (`ggml.c:29611`), so every worker's writes are ordered before the dump.
* Records per-op total ms, ms/graph and call count, plus a per-weight-type split for `MUL_MAT` /
  `MUL_MAT_ID` (keyed on `src[0]->type`), plus process-wide minor/major fault counters from
  `/proc/self/stat` in the dump header, so op time can be regressed against faults exactly as in
  Stage 4. Output goes to stderr as `BONSAI_PROF OP ...` / `BONSAI_PROF TYPE ...` lines.

Build and verification:

| item | value |
|---|---|
| build dir | `ik_llama.cpp/build-arm64-ik-prof`, identical options to `build-arm64-ik` plus `-DBONSAI_PROF=1` in both `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` |
| wall time | 2m43s (`make -j24`), 0 errors, 1 pre-existing unrelated warning (`ggml.c:24178`, unreachable break) |
| instrumentation present | `BONSAI_PROF` / `BONSAI_PROF_EVERY` / the dump format strings appear **5 times** in `build-arm64-ik-prof/ggml/src/libggml.so` and **0 times** in `build-arm64-ik/ggml/src/libggml.so` |
| artifacts | `libggml.so` 321,927,488 B, `libllama.so` 52,663,568 B, `llama-server` 100,795,488 B, all ARM aarch64 ELF |
| pinning flags in the prof binary | yes (2 occurrences of `cpu-mask-batch` in `llama-server`) |

The profiler build is **not** used for timing arms: instrumentation is cheap but the environment
differs, and the project's rule is that a timing number comes from the shipping build.

### 4.5 Tonight's script inventory

| file | side | purpose |
|---|---|---|
| `MoeDroid/tools/s6_shootout.sh` | host (adb) | driver: preflight, sha256 gate, staging, identity gate, two ordered passes, optional profiler pass |
| `MoeDroid/tools/s6_arm_ll.sh` | device (`/system/bin/sh`) | one long-lived-server steady-state arm; prints `S6REQ`/`S6MED`/`S6IDENT`/`S6CONF`/`S6CAP` lines |
| `MoeDroid/tools/s6_ll_table.py` | host | parses the arm logs into the decision table (medians, within-pass deltas, position control, group means, identity, engine config) |
| `MoeDroid/tools/s6_q2_0_x4_test.c` | NDK aarch64 | numeric verification of the ported kernel under qemu (section 5) |
| `MoeDroid/tools/s6_ppl_recheck.py` | host | the perplexity re-check (section 1) |

All five were syntax-checked, and the parser was validated end-to-end against synthetic arm logs
(`.stage6/fakelogs`) before being trusted - a parser that has never parsed anything is not a
deliverable. The device script avoids the two traps that cost real time in earlier stages: it never
uses `pkill -f libllamaserver` (it uses a `ps` + `kill` loop, because `pkill -f` matches the invoking
`adb shell` and kills it), and it stages nothing on `/data/local/tmp` except libraries.

---

## 5. Step 5: Candidate A, the highest-leverage piece into `upstream-llama`

Per the override, no attempt was made to port `ggml/src/iqk/` (46.5k lines / 26 files) into
`upstream-llama`. Instead the single highest-leverage piece that is *self-contained and verifiable
today* was ported: ik's four-weight-row `q2_0` kernel shape. This was completed before the override
arrived; nothing further was started after it, and no new port was begun.

### 5.1 What ik's kernel actually does, and what was ported

`mul_mat_q2_0_q8_0_x4` (`iqk_gemm_legacy_quants.cpp:4509-4602`) loops `nrc_x` weight rows with the
quantised activation resident, expands a whole 64-weight block with **one** 16-byte load plus four
`vqtbl1q_u8` from a 64-entry table, and at `nrc_x = 4` restarts `acc[iy]` per row. Two properties
matter for us:

* the four 32-weight group sums of a block pair are reduced into one `int32x4` with two
  `vpaddq_s32`, then scaled by a per-lane `float32x4` `{d_A, d_A, d_B, d_B}` folded with the
  activation scales, so **one `vfmaq_f32` retires 128 weights**;
* the activation is **loaded once and reused across weight rows** - which at `n = 1` (one activation
  row, so `nrc_y = 1`) is the *entire* available win from this shape.

Ported: both properties. **Not** ported: ik's `block_q8_0_x4` repacked activation type. That is the
rest of the story and it is a different size of change - it needs `GGML_TYPE_Q8_0_X4`,
`block_q8_0_x4`, `block_q8_2`, `quantize_row_q8_0_x4`, the `iqk_config`/`iqk_utils`/`iqk_common`
machinery (`DataInfo`, `Q8<>`, `mul_mat_qX_K_q8_K_T`) and an enlarged work-buffer contract, because
it changes the activation layout that the graph and the repack pass both know about. Our kernel
therefore reads plain `block_q8_0` and pays two 16-byte loads per 32-weight group where ik pays one
`vld1q_s8_x2` from its repacked buffer. That remainder is a follow-up, not a defect.

The port itself (`ggml/src/ggml-cpu/arch/arm/iqk_x4.c`, new, ~250 lines with the derivation comment):

| mode | `BONSAI_MOE_X4` | accumulation | gate it can pass |
|---|---|---|---|
| 1 | `=1` | four rows share the activation load; each 32-weight group accumulated with the same scalar `vmlaq_n_f32`, in the same block order, and with two accumulators alternating even/odd blocks exactly as the shipping kernel does | **bit-exact** -> eligible for the token-identity gate |
| 2 | `=2` | additionally ik's packing: 8 `vdotq_s32` + 4 `vpaddq_s32` + 1 `vcvtq_f32_s32` + 1 `vfmaq_f32` per 128 weights per row, instead of 8 + 4 + 4 | near-exact only (**not** identity-safe by construction); it exists to bound what the full ik shape buys here |
| 0 (default) | unset/0 | the shipping `ggml_vec_dot_q2_0_q8_0` path is used unchanged | - |

Integration (two call sites, both behind the same gate and only when four whole rows are in range,
with the remainder falling through to the original per-row `vec_dot`):

| site | what it covers | share |
|---|---|---|
| `ggml_compute_forward_mul_mat_id_one_chunk` (`ggml-cpu.c:1610`) | routed experts, `Q2_0`, single-token decode | **45.9% of decode** (Stage 4) |
| `ggml_compute_forward_mul_mat_one_chunk` (`ggml-cpu.c:1254`, only when `num_rows_per_vec_dot == 1`) | dense `Q2_0` | 2.5% of decode today, but the same kernel is what the expert path needs |

`type_traits[GGML_TYPE_Q2_0].nrows` was deliberately **left at 1**. Raising it would route through
mainline's generic multi-row path, and that path is unusable for decode by construction: the guard
`if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ...) num_rows_per_vec_dot = 1;` forces `nrc = 1` at
`n = 1` because there is only one activation row. The hook is therefore explicit, which also keeps
our own `BONSAI_MULMATID_FLAT` work distribution untouched.

### 5.2 Verification: real ARM64 code, run on the host under qemu

Method (`MoeDroid/tools/s6_q2_0_x4_test.c`, built by the NDK as a **static** aarch64 binary and run
under `qemu-aarch64-static`, §0): on identical random `block_q2_0` rows and a random `block_q8_0`
activation row, compare

* `ref` - the real `ggml_vec_dot_q2_0_q8_0` from `arch/arm/quants.c`, compiled into the same binary;
* `x4 mode 1` and `x4 mode 2`;
* a double-precision scalar oracle built from the fp16-rounded scales the kernels actually see.

Because the binary is statically linked, the instructions executed under qemu are the instructions
the device will execute (`vdotq_s32`, `vqtbl1q_u8`, `vshlq_u8`), with `-march=armv8.2-a+dotprod+i8mm+fp16+bf16`.

Result (`BONSAI_Q20_UNROLL=1`, the shipping configuration, n = nb x 64 with nb in {1,2,3,5,8,16,32,64}):

| nb | mode 1 vs reference | mode 2 max abs deviation | reference vs double oracle |
|---|---|---|---|
| 1, 2, 3, 5, 8, 16, 32, 64 | **BIT-EXACT on all 8 cases** (`max abs diff = 0.000e+00`) | <= 7.63e-05 (relative <= 2.9e-7 at scale 266) | <= 4.30e-05 |

`ALL CHECKS PASSED`, exit 0. With `BONSAI_Q20_UNROLL=0` the reference itself changes accumulation
order, so mode 1 is near-exact rather than bit-exact there - the test reports that distinction
instead of failing on it, and the comparison that matters is the one against the shipping default.

**The harness earned its keep: it caught two real bugs that review did not.** Row 1, 2 and 3 were
initially accumulating with row 0's block scale (a ~74% error on the first case), and the mode 2
odd-block tail was double-counting a block. Both were visible immediately as `mode1 DIFFERS` against
a reference that was itself correct once a separate oracle mis-indexing was fixed. This is the
argument for the qemu harness: without it, both bugs would have shipped to the device and been
mistaken for "the port does not help".

### 5.3 Builds

| build | result |
|---|---|
| `upstream-llama/build-arm64-ship` (NDK, ninja) | green, 7.0 s incremental; `bonsai_vec_dot_q2_0_q8_0_x4` (4144 B) and `bonsai_q2_0_x4_mode` are exported from `bin/libggml-cpu.so` |
| `upstream-llama/build-arm64-prof` (Unix Makefiles, `-DBONSAI_PROF=1`) | green, 11.3 s; rebuilt so tonight's profile includes the x4 path |
| `upstream-llama/build-x86` (host) | green, 11.5 s, and **no `bonsai_*` symbol in `bin/libggml-cpu.so`** - the `__aarch64__` guards provably compile the kernel and both hooks out on x86 |

### 5.4 What to expect, honestly

Single-digit percent at best on decode, and this port is a *measurement* whose result decides the next
move rather than a likely step change. The reasons are in the project's own numbers:

* Stage 4 measured the expert path at 79.5 ms/token with **31% of its excess over the dense rate
  attributable to page-fault stalls**, so at most ~43 ms/token is even addressable by kernel compute;
* the mode 1 win is activation-load sharing only (mode 2 adds the packing). At `n = 1` there is no
  second activation row to share, which is exactly why the port plan said `nrc` is "worth only a few
  ms" and why it said to profile before writing kernels;
* on a **dense cache-resident** type the same shape is worth much more (see 5.5), because there the
  activation traffic is the dominant term rather than noise on top of page faults.

### 5.5 Not done today, and the honest reason

| not attempted | why | what it is worth |
|---|---|---|
| full `iqk` wiring into `upstream-llama` | explicitly excluded by the override, and the dependency list in §5.1 confirms it is not a one-day job | needed for the last 10-20% of the gap |
| `q3_K` four-row kernel | a ~94-line hand-written NEON path (`arch/arm/quants.c:2225-2319`) with 6-bit scales and an hmask, so a four-row variant is a materially larger and riskier job than `q2_0`; with the session timebox it would have shipped unverified | **the largest remaining dense lever. See the arithmetic below** |
| fused up+gate op (`MOE_FUSED_UP_GATE`) | it is a new ggml op plus a CPU kernel plus a graph change, and it touches the flat/tiled/prefetch expert machinery, the profiler and the app's graph; a subtle break would poison tonight's session. Crucially, **it no longer has to be guessed at**: ik ships `--no-fused-moe` (`common/common.cpp:2251`), so tonight's B2 arm *measures* the fusion's value on ik before we write any of it | 3 expert ops per layer -> 2, inside the 45.9%/61.9% bucket |
| all-token-count expert chunking (port of `GGML_EXPERT_CHUNKING`) | same reasoning: it is a work-distribution change in our flat path, and it is best sized after tonight shows how much of the prefill gap it explains | the best explanation of prefill (+43%) exceeding decode (+25%) |

**Why `q3_K` is the top follow-up, with numbers** (all block sizes verified from
`ggml/src/ggml-common.h`): `QK_K = 256`; `block_q8_K` is 292 B (9.125 bits/weight), `block_q3_K` is
110 B (3.44 bits/weight), `block_q5_K` is 176 B (5.5 bits/weight). For a 2560-wide weight row at
`nrc = 1`, the kernel streams the **activation row** plus the weight row: `q3_K` moves 2920 + 1100 =
4020 B per row to deliver 1100 B of weights (3.65x overhead), while `q5_K` moves 2920 + 1760 = 4680 B
for 1760 B (2.66x). Stage 4 measured `q3_K` at 22.6 GiB/s and `q5_K` at 45.9 GiB/s of *weight* bytes,
i.e. roughly 82 and 122 GiB/s of total traffic - so `q3_K` also pays a ~1.5x per-byte unpack premium,
which bounds the payoff. Sharing the activation across four weight rows cuts the `q3_K` per-row
traffic to 1100 + 730 = 1830 B (0.46x), so the traffic-bound component could improve by up to 2.2x;
blended with the unpack premium, the realistic band is **22.6 -> ~34-50 GiB/s**, and since `q3_K` is
20.6 ms/token (11.9% of the 173.2 ms decode budget), that is a **4-6% decode** prize - larger than
anything the `q2_0` shape can plausibly return on the expert path, and it is a pure dense,
cache-resident win with none of the page-fault noise. **Prediction, not measurement: it needs the
same qemu harness extended to `q3_K` before it is believed.**

---

## 6. Tonight's turnkey shootout

### 6.1 Preconditions (the run is worthless without these)

1. Phone **unlocked**, screen awake, RedMagic Game Space **ON**. A locked phone unmounts
   credential-encrypted `/sdcard`, and then nothing can be loaded at all (handoff rule 7.6).
2. Fan on. Stage 5 ran with one and started at 29-35 C instead of Stage 4's 41-43 C; the cap gate
   (`cpu0 = 3532800`, `cpu6 = 3840000` for 3 consecutive 10 s polls) must pass before **every** arm.
3. The driver verifies `varH` shard 1 sha256 `e7c694526e82f633eda7f37b5e35ed7075fd39140018c99d73e305818ded12e2`
   and shard 2 `316b46f3...c161e113` on device, and **aborts if either is wrong**. Both engines read
   the *same file* (`/sdcard/Download/mf-model/varH-00001-of-00002.gguf`), so there is one page-cache
   universe and one less confound.
4. `governor`/cap state is recorded before and after every arm and sampled every 5 s across it.

### 6.2 Run it

```sh
cd /home/kingfish600/workspace
# stage + identity gate only (about 12 minutes, decides whether mode 1 may ship)
MoeDroid/tools/s6_shootout.sh 0
# the full shootout: two ordered passes, 10 timing arms (about 70-80 minutes)
MoeDroid/tools/s6_shootout.sh 2
# the same plus the per-op profiler pass on both engines (about +30 minutes)
PROF=1 MoeDroid/tools/s6_shootout.sh 2
# analysis
python3 MoeDroid/tools/s6_ll_table.py /home/kingfish600/workspace/.stage6-logs
```

### 6.3 The matrix, and what each arm decides

`A0 A1 A2 A3 A4 B0 B1 B2` forward, then `B2 B1 B0 A4 A3 A2 A1 A0` reverse, with a 120 s cooldown
between arms. Every arm is a **long-lived server**: one load, 4 warmup requests, then 6 measured
greedy requests, and the whole per-request series is printed rather than only the last value - if it
has not plateaued, the median is not trustworthy and that is visible in the output.

`B1` is in the decode pass **on purpose** (audit fix 1): `B0` lacks `--poll 0` while `B2` has
`--poll 0` *and* `--no-fused-moe`, so `B0 - B2` conflated the two. With `B1` present, `B1 - B0`
isolates ik's flag delta and `B1 - B2` isolates the MoE fusion at identical flags.

Part 2 adds a **second, prefill-mode pass** (`A0 AP B0 B1`, then `B1 B0 AP A0`), because the two new
levers and ik's batch-mask lever act only on prefill. Budget: 16 decode arms plus 8 prefill arms,
roughly 2.5 hours including the 120 s cooldowns.

| arm | engine | configuration | what it decides |
|---|---|---|---|
| `id-x4off` / `id-x4on` | ours | `ident` mode, `BONSAI_MOE_X4=0` vs `=1` | **the ship gate**: md5 of the 64 generated token ids must be identical. Mode 1 is bit-exact per call (§5.2), so a mismatch here would mean the integration, not the arithmetic, is wrong |
| `A0` | ours | `BONSAI_MOE_X4=0` (shipping) | the baseline every other arm is differenced against in its own pass |
| `A1` | ours | `BONSAI_MOE_X4=1` (bit-exact x4) | **how much the activation-sharing shape is worth on the expert path** - the single most informative number of the night, because the same mechanism underlies ik's fused MoE kernel |
| `A2` | ours | `BONSAI_MOE_X4=2` (packed) | whether ik's `dx` packing adds anything beyond mode 1. If A2 > A1 materially, the packed accumulate is worth keeping despite not being identity-safe |
| `A3` | ours | `BONSAI_Q3K_X4=1` (q3_K x4, bit-exact) | the dense `q3_K` lever from section 8: 11.9% of decode, predicted 22.6 -> 34-50 GiB/s. Bit-exact, so it is token-identity safe by the section 8.2 argument |
| `A4` | ours | `MOE_X4=1 + Q3K_X4=1` | both bit-exact kernels together: the number the ship decision actually needs, and a check that the two effects are additive rather than overlapping |
| `AP` | ours | `BONSAI_MULMATID_FLAT=2` (single prefill queue) | the prefill scheduling change from section 9. Bit-identical to stock (verified), so any difference here is purely scheduling. Prefill pass only |
| `B0` | ik | Stage 5 flags (`-t 6 -tb 8 --cpu-mask 0x3f -ub 512`) | the oracle re-run under the corrected protocol |
| `B1` | ik | + `--cpu-mask-batch 0xff --cpu-strict-batch 1` (+ `--cpu-strict 1 --poll 0`) | **best vs best for the first time**: ik was never able to pin the batch pool to `0xff` before today (§4.1). This is the arm that makes the prefill comparison fair |
| `B2` | ik | B1 + `--no-fused-moe` | **isolates the up+gate fusion** (§2.1), the largest single suspected contributor, on the same binary and the same session. The engine prints `fused_moe = 1/0` at load and the harness records it, so the configuration is proven rather than assumed |
| `prof-ours-x4off` / `prof-ours-x4on` | ours | `BONSAI_PROF_EVERY=1`, `BONSAI_MOE_X4=0/1` | our per-op budget with and without the port, to be differenced with `s4_prof.py` against Stage 4's table |
| `prof-ik` / `prof-ik-nofmoe` | ik | `BONSAI_PROF_EVERY=1`, with and without `--no-fused-moe` | **ik's per-op budget for the first time** (§4.4). This is what turns the static attribution of §2 into a measured comparison: `MUL_MAT_ID` vs `MOE_FUSED_UP_GATE` vs `DELTA_NET` vs attention, in ms/token, on the same device |

Interpretation rules, fixed in advance so the result cannot be rationalised afterwards:

* `B0` today should land **near or above** Stage 5's 8.61 median decode / 11.98 mean prefill, which
  are **floors** rather than expected values (section 12.4: ik ran those with checkpointing on and
  tonight runs it off). A *large* miss still means something in the environment changed (cooler
  phone, different page-cache state), and the whole session should be read relative to `A0` **within
  its own pass**, never against Stage 5's absolutes.
* `B1 - B0` is ik's prefill gain from the batch pinning our fork already had. If it is ~+20%, Stage 5's
  oracle gap was flattered on prefill and must be restated.
* `B0 - B2` is the fusion's worth. If it is small, the fused-op work is not worth writing and `q3_K`
  (§5.5) becomes the top lever instead.
* `A1 - A0` and `A2 - A0` bound the shared-activation mechanism. If they are ~1%, then ik's advantage
  is *not* the `_x4` shape and the answer lies in chunking and the fused op; if they are 5%+, the
  kernel port direction is confirmed and the `q8_0_x4` activation repack becomes the next step.
* Position control: the same config in pass 1 and pass 2 at mirrored positions. A spread >3% there is
  the noise floor for the night, and any effect smaller than it is not claimed.

### 6.4 Traps this harness already avoids

* **Protocol**: everything is long-lived-server steady state. Mixing this with Stage 3/4 arm-protocol
  absolutes would manufacture a fake 25-30% effect (Stage 5 §1.5).
* **`pkill -f`**: never used (it kills the invoking `adb shell`); the device script uses `ps` + `kill`.
* **Storage**: models are read from `/sdcard` only. The same bytes on `/data/local/tmp` measured
  0.09 t/s against 4.76 t/s (handoff rule 7.10).
* **Stale engines**: the arm script kills any surviving `libllamaserver` before the cap gate, because
  a survivor holds the port and makes the next arm report `ready after ~0s` while measuring the old
  model.
* **`/tmp` is private per bash invocation** in this harness, so all logs go to
  `/home/kingfish600/workspace/.stage6-logs`, which `s6_ll_table.py` reads.
* **libomp**: the ik lib sets get `libomp.so` pushed from the NDK (`clang/19/lib/linux/aarch64`),
  since `libggml.so` NEEDs it and ours does not.

---

## 7. Files changed in this session

Nothing was committed and no `adb` command was run, per the session rules. Everything below is in
the working tree on the host.

### 7.1 New files

| path | purpose |
|---|---|
| `MonsterQwenDroid/docs/STAGE6-daytime-prep.md` | this report |
| `MonsterQwenDroid/GOAL.md` | goal-md objective document, both protocols, gates, model/engine state |
| `MonsterQwenDroid/iterations.jsonl` | 35 backfilled iterations, Stages 1-5 (15 kept, 20 closed) |
| `upstream-llama/ggml/src/ggml-cpu/arch/arm/iqk_x4.c` | the ported four-row `q2_0` kernel, modes 1/2, gated by `BONSAI_MOE_X4` |
| `MoeDroid/tools/s6_q2_0_x4_test.c` | static aarch64 numeric verification (bit-exactness vs the real reference and a double oracle) |
| `MoeDroid/tools/s6_shootout.sh` | host driver: preflight, sha256 gate, staging, identity gate, two passes, profiler pass |
| `MoeDroid/tools/s6_arm_ll.sh` | device-side long-lived-server arm, prints `S6*` lines |
| `MoeDroid/tools/s6_ll_table.py` | parses the arm logs into the decision table |
| `MoeDroid/tools/s6_ppl_recheck.py` | the perplexity re-check of section 1 |
| `.stage6/stage6-task2-attribution.md` | full 868-line ik call-path attribution, every claim with `file:line` |
| `.stage6/stage6-ik-all.patch` | combined ik diff (pinning + profiler), 644 lines |
| `.stage6/stage6-ik-pinning.patch` | pinning-only subset; reverse-applies cleanly against the current tree, so it is provably a subset |
| `.stage6/ppl-recheck.txt`, `.stage6/host-proof-pinning.txt`, `.stage6/stage6-ik-pinning-notes.md` | raw evidence for sections 1 and 4 |
| `.stage6/qemu/` | extracted `qemu-user-static` 8.2.2 (the ARM64 execution enabler, section 0) |

### 7.2 Modified files

| repo | files | what |
|---|---|---|
| `upstream-llama` | `ggml/src/ggml-cpu/CMakeLists.txt` (+1) | add `arch/arm/iqk_x4.c` to the ARM source list |
| | `ggml/src/ggml-cpu/ggml-cpu-impl.h` (+7) | prototype the kernel and the gate, `__aarch64__`-guarded |
| | `ggml/src/ggml-cpu/ggml-cpu.c` | two gated call sites (expert and dense single-token `mul_mat`) plus the pre-existing uncommitted `BONSAI_PROF`/`BONSAI_MOE_PREFETCH` work |
| | `src/llama-context.{cpp,h}` | the pre-existing uncommitted `BONSAI_SHIELD` work, untouched today |
| `ik_llama.cpp` | 9 files | the pinning flags (subagent work, `.stage6/stage6-ik-pinning.patch`) |
| | `ggml/src/ggml.c` | additionally carries the `BONSAI_PROF` port (§4.4); the only file both changes touch, in non-overlapping regions |
| `MonsterQwenDroid` | `GOAL.md`, `iterations.jsonl` | new, from Step 3 |

### 7.3 Build state left behind for tonight

| build dir | define | state |
|---|---|---|
| `upstream-llama/build-arm64-ship` | shipping flags | green, kernel exported |
| `upstream-llama/build-arm64-prof` | `-DBONSAI_PROF=1` | green, rebuilt with the port |
| `upstream-llama/build-x86` | host | green, kernel and hooks compiled out (no `bonsai_*` symbol) |
| `ik_llama.cpp/build-arm64-ik` | shipping flags + pinning patch | green |
| `ik_llama.cpp/build-arm64-ik-prof` | `-DBONSAI_PROF=1` | green, 2m43s, instrumentation verified present in `libggml.so` |

### 7.4 What tonight decides, in one line each

1. **Ship gate**: `id-x4off` vs `id-x4on` token-identity md5 (`A1` is bit-exact per call, so this
   tests the integration).
2. **Is the activation-sharing shape the answer?** `A1/A2 - A0`.
3. **Is the up+gate fusion the answer?** `B0 - B2` on ik's own binary, no code needed.
4. **Was the Stage 5 oracle flattered on prefill?** `B1 - B0`.
5. **Where does ik's time actually go?** `prof-ik` vs `prof-ours-*`, the first measured op budgets
   from both engines on the same device.

If (2) is small and (3) is large, the next change is the fused MoE op, not more kernels. If (2) is
large, the next change is the `q8_0_x4` activation repack and then the `q3_K` four-row kernel of
§5.5, which is the biggest single dense lever left.

---

# PART 2 (same daytime session, host-only, no phone)

Part 2 implements the three highest-leverage items identified in Part 1 (§2.1, §2.2, §5.5) and
verifies each numerically with the qemu-aarch64-static bench stood up in §0. All three are gated
off by default and are ready for tonight's on-device arms alongside `BONSAI_MOE_X4`.

## 8. Task 1: four-row `q3_K` kernel (`bonsai_vec_dot_q3_K_q8_K_x4`, `BONSAI_Q3K_X4`)

### 8.1 Why this is the biggest remaining dense lever

From §5.5, with block sizes verified from `ggml/src/ggml-common.h`: at `nrc = 1` the kernel streams a
2920 B `block_q8_K` activation row for every 1100 B `block_q3_K` weight row, a 3.65x overhead, and
Stage 4 measured the result at 22.6 GiB/s against 43-50 GiB/s for every other dense type. `q3_K` is
20.6 ms/token, 11.9% of the 173.2 ms decode budget.

### 8.2 Why bit-exactness is structural here, not luck

Reading the reference NEON path (`arch/arm/quants.c:2225-2318`) showed the property that decides the
whole design: within a 256-weight block the kernel accumulates **every term into an exact `int32`
(`isum`)** - `isum += vaddvq_s32(vdotq_s32(...)) * scale[k]` - and only then does

```c
sum += d * isum;      // d = y[i].d * FP16(x[i].d), one float fma per block
```

All the per-group terms are integer, so `isum` is exact and independent of the order the groups are
summed in, and the float part is a single fma per block per row. Therefore a four-row kernel that
reproduces the same integer accumulation gets a **bit-identical** result regardless of instruction
scheduling, register pressure or spilling. No tolerance is needed anywhere.

### 8.3 What the kernel shares, and what it cannot

`bonsai_vec_dot_q3_K_q8_K_x4` (in `arch/arm/iqk_x4.c`, appended after the q2_0 kernel) loads the
`q8_K` activation registers **once per 128-weight group and reuses them for all four rows**, and
keeps the per-row work that genuinely differs: the 16 unpacked int8 scales, the `hmask` high-bit
vectors, the q3 unpack and the group order.

| per 128 weights, per row | reference | x4 |
|---|---|---|
| q8_K activation loads | 128 B | **32 B** (shared four ways) |
| q3_K `qs` + `hmask` loads | 48 B | 48 B (unchanged) |
| `vdotq_s32` / `vaddvq_s32` / int muls | unchanged | unchanged |

So the per-row loaded bytes drop from 352 B to 160 B (about 45% less), which is exactly the term the
traffic arithmetic in §5.5 blames. The predicted band is 22.6 -> 34-50 GiB/s, i.e. 4-6% of decode -
still a **prediction**: the kernel is verified numerically, not for speed, because qemu timing is
meaningless. Tonight's on-device arm is what settles it.

### 8.4 Verification: three independent implementations

`MoeDroid/tools/s6_q3_K_x4_test.c`, built by the NDK as a static aarch64 binary and run under
`qemu-aarch64-static` (§0), compares on identical random data:

* `ref_neon` - `ggml_vec_dot_q3_K_q8_K`, the shipped NEON kernel;
* `ref_gen` - `ggml_vec_dot_q3_K_q8_K_generic`, the scalar implementation
  (`ggml-cpu/quants.c:617`), an independent cross-check of the reference itself;
* `x4` - the port.

| nb | n | x4 vs shipped NEON reference | generic vs NEON reference |
|---|---|---|---|
| 1 | 256 | **BIT-EXACT** | 3.05e-05 |
| 2 | 512 | **BIT-EXACT** | 2.10e-05 |
| 3 | 768 | **BIT-EXACT** | 6.10e-05 |
| 5 | 1280 | **BIT-EXACT** | 6.10e-05 |
| **10** | **2560** | **BIT-EXACT** | 1.07e-04 |
| 16 | 4096 | **BIT-EXACT** | 4.58e-05 |

`ALL CHECKS PASSED`, exit 0. The two independent references agree to ~1e-7 relative (on values of
order 300-850), which cross-validates that the NEON reference is not itself doing something odd. The
test also validates `bsums` consistency against `qs` so the data is realistic rather than arbitrary.

### 8.5 Integration

One call site, the same shape as the q2_0 port: `ggml_compute_forward_mul_mat_one_chunk`, only when
`type == GGML_TYPE_Q3_K` and `num_rows_per_vec_dot == 1`, four rows at a time, with the remainder
falling through to the original per-row `vec_dot`. Gate: `BONSAI_Q3K_X4` (default 0). `q3_K` is a
dense type only (its tensors are read by `MUL_MAT`, not by the expert path; `token_embd.weight` is
`q3_K` but read by `GET_ROWS`, so it is untouched).

## 9. Task 2: all-token-count expert chunking (`BONSAI_MULMATID_FLAT=2`)

### 9.1 The obstacle, and the fact that removes it

ik's `GGML_EXPERT_CHUNKING` drains one atomic queue over (expert x chunk) at every token count; our
`BONSAI_MULMATID_FLAT` did that only for single-token decode (`ids->ne[1] == 1`).

Restructuring the prefill path is only safe if the tiled `MUL_MAT_ID` path can never claim an expert,
because **tiling walks experts in lockstep across threads and does collective work**, so reordering
the expert visits to balance a global queue would corrupt it. Checking the gate rather than assuming
it: `ggml_compute_forward_mul_mat_id_tiled` requires `ggml_tiled_supported(...)` (`tiled.cpp:605`),
and that switch lists `Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_XS, IQ2_*, IQ3_*, IQ1_*` - **`GGML_TYPE_Q2_0`
is not among them, and every routed expert in this model is `Q2_0`.** For varH, therefore, the stock
path is *always* `ggml_compute_forward_mul_mat_id_one_chunk` at every token count, and replacing the
counter structure cannot change a single arithmetic operation.

To keep that reasoning honest for other models instead of hardcoding a type check, the tiled
eligibility was exported as `ggml_compute_forward_mul_mat_id_tiled_eligible(src0, src1, cne1)`
(`tiled/tiled.h`, `tiled.cpp`), documented as a **conservative upper bound** on the real per-expert
dispatch (it applies the type and batch gates but not the final dispatch, so it may report true where
the dispatch would decline). Mode 2 requires it to be false for every expert with rows; on a model
whose expert type *is* tiled, the mode silently stays on the stock path.

### 9.2 Implementation

`BONSAI_MULMATID_FLAT=2` extends the existing single-counter path to `ids->ne[1] > 1`:

* chunks per expert mirror ik's `MAX(1, MIN(nth, ne01/32))`, which makes the unit -> (expert, chunk)
  mapping **constant-time with no prefix table** (`expert = u / cpe`, `chunk = u % cpe`) - that is the
  property that made ik's design cheap and it is why a global queue needs no extra allocation or
  `wsize` contract change;
* one `atomic_fetch_add` on the existing counter word; experts with no rows and empty chunks are
  skipped;
* every unit calls the *same* `ggml_compute_forward_mul_mat_id_one_chunk` with the same parameters the
  stock path would use, so each output element is produced by exactly one call with identical inputs
  (this is also why the decode flat path was already bit-identical in Stage 2);
* mode 2 keeps mode 1's decode behaviour unchanged (`ids->ne[1] == 1` still takes the decode branch).

### 9.3 Verification: byte-for-byte on a synthetic `MUL_MAT_ID` graph

`MoeDroid/tools/s6_mmid_prefill_test.c` (host x86, against `build-x86`) builds a real `MUL_MAT_ID`
graph with Q2_0 weights produced by `ggml_quantize_chunk` - the same quantiser and type as the model -
at 512 experts, `-t` batch widths and several token counts. Driver:
`MoeDroid/tools/s6_mmid_prefill_verify.sh`. Gate: byte equality of the float outputs, plus stability
controls on both modes so a vacuous comparison cannot pass.

| case | shape | threads | result |
|---|---|---|---|
| 0 | n_embd 256, n_ff 512, 512 experts, used 10, **16 tokens** | **8** | mode2 == stock, both stable |
| 1 | 256/512, 256 experts, used 4, 16 tokens | 8 | mode2 == stock, both stable |
| 2 | 128/256, 512 experts, used 8, 32 tokens | 8 | mode2 == stock, both stable |
| 3 | 256/512, 512 experts, used 10, 16 tokens | **1** | mode2 == stock, both stable |
| 4 | 256/512, 512 experts, used 2, 4 tokens | 8 | mode2 == stock, both stable |
| 5 | 256/512, 512 experts, used 10, 1 token | 6 | mode2 == stock, both stable |

`PASS: BONSAI_MULMATID_FLAT=2 is bit-identical to stock on all cases, both stable`. Case 3 (single
thread) matters because it removes thread-interleaving as an explanatory variable; case 5 confirms
the decode path is untouched.

Both builds are green with the change: `build-arm64-ship` and `build-x86`.

### 9.4 What is and is not claimed

* **Claimed**: bit-identical output, verified on six shapes including the 16-token/8-thread prefill
  case the task asked for, with the mechanism (uniform chunks, constant-time mapping, no tiling) cited
  to source.
* **Not claimed**: the throughput gain. This is a scheduling change whose effect depends on how much
  the per-expert counters actually stall 8 batch threads on this workload; that is unmeasured and is
  exactly what tonight's arms decide. Stage 4 showed the expert path is the prefill bottleneck (61.9%
  of prefill) and that Stage 3's *within-expert* pinning was worth +21.5%, so a queue-level effect of
  a few percent is plausible but not established.

## 10. Task 3: fused `up+gate` - feasibility answered, the kernel shape is constrained, deliberately not wired

### 10.1 Can it be wired cleanly? Yes, and this fork has already done it once

The precedent is our own `GGML_OP_MOE_SUM`, and it shows the touch points are small and known:

| file | MOE_SUM | for a fused up+gate op |
|---|---|---|
| `ggml/include/ggml.h` | `GGML_OP_MOE_SUM,` at `:604` (before `GGML_OP_COUNT`) | +1 enum entry |
| `ggml/src/ggml.c` | name string at `:1114`, builder at `:2522` | +1 name, +1 builder |
| `ggml/src/ggml-cpu/ggml-cpu.c` | dispatch case at `:2178`, compute at `:2654` | +1 case, +1 compute |
| `src/llama-graph.cpp` | - | graph change at the up/gate branch |

Two hazards found by reading the pattern: `GGML_OP_NAME` (`ggml.c:1002`) and `GGML_OP_SYMBOL`
(`ggml.c:1119`) are both positional `[GGML_OP_COUNT]` arrays, and **`MOE_SUM` has an entry in the
first but not the second**, so `ggml_op_symbol()` returns NULL for it today - a latent blemish worth
fixing whenever the next op is added. Neither array is validated at build time, so a mismatch is a
silent misalignment, not a compile error.

### 10.2 The kernel shape is constrained, and a naive fusion would be a regression

The brief's framing ("turning 3 expert calls per layer into 2 ... walking the routed token/activation
structures once for both up and gate") suggests fusing the pair. Sizing it first shows that the
minimal pair kernel is **worse than the `x4` kernel already implemented in Part 1**:

| kernel | activation bytes per row | weight bytes per row | total per row | vs reference |
|---|---|---|---|---|
| reference, 1 row (all `nrc = 1`) | 136 B | 36 B | 172 B | 1.00x |
| `x4` (4 weight rows share one activation load) | 34 B | 36 B | 70 B | **2.46x** |
| fused 1 gate + 1 up row | 68 B | 36 B | 104 B | 1.65x |
| **fused 4+4** (4 gate and 4 up rows share one load) | **17 B** | 36 B | **53 B** | **3.25x** |

(per 128 weights: two `block_q2_0` at 18 B each = 36 B of weights, four `block_q8_0` at 34 B each =
136 B of activation, a 3.78x activation-to-weight ratio. Block sizes from `ggml-common.h`.)

So the fusion only pays as a **4+4** kernel, and even then its **incremental** value over `x4` is
**1.32x on the load-bound component only** - `x4` already captures the larger 2.46x share of the same
mechanism. This also resolves an internal inconsistency in the Stage 5 port plan, which says in
section 1.1 that `_x4` "dequantizes and scales the activation once for four weight rows instead of
four times" and then concludes in section 3 that at `n = 1` "there is nothing to share the `y` load
with". The sharing is across *weight rows*, which exists at `n = 1`; it is not across tokens.

### 10.3 Why byte-count arguments cannot settle it, and why that argues for measuring first

The 1.32x applies only if the expert kernel is load-bound. Stage 4 says it is not purely so: 31% of
its excess over the dense rate is page-fault stalls and 42.9 ms/token is left as an unexplained
residual. The activation re-reads themselves live in L1/L2 (about 2.4 GiB/token of L1 traffic at a
few ms of L1 time), not in DRAM, so they cannot simply be added to the weight traffic. Meanwhile the
expert path is 45.9% of decode (79.5 of 173.2 ms), so if the mechanism does land it lands on the
largest bucket in the budget - the honest position is that the mechanism is large, its end-to-end
size is unknown, and the two arms that measure it are already in tonight's plan.

### 10.4 Explicitly not done today, and the reason

The `BONSAI_MOE_FUSED_UP_GATE` op and graph path were **not** implemented in this session. Three
reasons, in order of weight:

1. **The decisive measurement is already scheduled and costs nothing.** ik ships
   `--no-fused-moe` (`common/common.cpp:2251`), so tonight's `B2` arm measures exactly this fusion on
   ik's own binary, same model, same session (`B0 - B2` in section 6.3). `A1 - A0` measures how much
   of the same mechanism our existing kernel already captures. Wiring an op that the very next
   measurement might show is worth ~1% would be the wrong order of work.
2. **A bit-exact 4+4 kernel is register-bound.** Matching the shipping dual-accumulator order needs
   two `float32x4` accumulators per row for eight rows, plus four to eight live activation registers
   and the unpack temporaries. That will spill, and the host can only run it under qemu where timing
   is meaningless, so it cannot be speed-validated before the device session. Writing it blind risks
   shipping a kernel that is *slower* than the `x4` one it would complement.
3. **A new op can break the build we are preparing.** All three name/dispatch tables are positional
   and unvalidated; a mistake there takes down tonight's session rather than improving it.

### 10.5 The recipe, ready for when the measurement justifies it

1. `ggml.h`: add `GGML_OP_MOE_FUSED_UP_GATE` before `GGML_OP_COUNT`.
2. `ggml.c`: append to **both** `GGML_OP_NAME` and `GGML_OP_SYMBOL` (and fix `MOE_SUM`'s missing
   symbol while there); add `ggml_moe_fused_up_gate(ctx, gate_exps, up_exps, cur, ids)` with
   `src[0] = gate_exps`, `src[1] = up_exps`, `src[2] = cur`, `src[3] = ids`, output
   `[n_ff, n_used, n_tokens]` holding the already-gated `silu(gate) * up`, with the unary op in
   op_params.
3. `ggml-cpu.c`: dispatch case plus compute that reuses the `mul_mat_id` work distribution (including
   the new prefill queue of section 9), calls the 4+4 fused q2_0 kernel where the activation is shared,
   falls back to two `vec_dot` calls per element for other types, then applies `silu(gate) * up`.
4. `llama-graph.cpp`: at the up/gate branch (`:2338`/`:2351`), when the env gate is on and
   `up_exps->type == gate_exps->type`, build the fused op instead of the two `build_lora_mm_id` calls.
   **Hazard**: the fused op output is already gated, so the `silu` that the unfused path applies to the
   gate must be skipped - applying it twice is the obvious way to get a subtly wrong model.
5. Verification order, all three of which are now cheap because the harness exists: a host x86
   synthetic-graph fused-vs-unfused byte-equality test (the section 9.3 pattern), a qemu test of the
   4+4 kernel against eight reference `vec_dot` calls (the section 8.4 pattern), then token identity
   on device.

### 10.6 What is verified today, instead

| item | state |
|---|---|
| `q3_K` four-row kernel (Task 1) | implemented, **bit-exact** vs the shipped NEON reference on 6 sizes incl. n=2560, cross-checked against the scalar implementation |
| all-token expert queue (Task 2) | implemented, **byte-identical** to stock on 6 synthetic `MUL_MAT_ID` shapes incl. 16 tokens / 8 threads, with stability controls |
| `q2_0` four-row kernel (Part 1) | implemented, **bit-exact**, with a packed near-exact mode |
| fused up+gate op (Task 3) | **not implemented**; feasibility confirmed (3-file precedent in this fork), kernel shape constrained to 4+4 by the table in 10.2, recipe above, and the value is measured tonight before any code is written |

## 11. Part 2 artifacts, and the state left behind

| path | what |
|---|---|
| `MoeDroid/tools/s6_q3_K_x4_test.c` | q3_K four-row kernel verification: NEON reference vs scalar reference vs the port, 6 sizes incl. n=2560 |
| `MoeDroid/tools/s6_mmid_prefill_test.c` | synthetic `MUL_MAT_ID` graph (real Q2_0 weights) for the prefill-queue gate |
| `MoeDroid/tools/s6_mmid_prefill_verify.sh` | one-command Task 2 verification, with stability controls on both modes |
| `.stage6/mmid/` | the raw per-case float dumps and logs the byte comparison is made from |

Modified for Part 2 (`upstream-llama`): `arch/arm/iqk_x4.c` (q3_K kernel + gate),
`ggml-cpu-impl.h` (prototypes), `ggml-cpu.c` (q3_K hook, prefill queue, gate reads),
`tiled/tiled.h` + `tiled/tiled.cpp` (exported tiling-eligibility predicate).

Gates, all default **off** and all env-toggled at runtime so one binary carries them:

| env | default | effect |
|---|---|---|
| `BONSAI_MOE_X4` | 0 | 1 = q2_0 four-row kernel (bit-exact), 2 = plus ik's packed accumulate (near-exact) |
| `BONSAI_Q3K_X4` | 0 | q3_K four-row kernel (bit-exact) |
| `BONSAI_MULMATID_FLAT` | 1 | 2 = single (expert x chunk) queue for prefill as well as decode |

Build state, all current with the tree at the end of the session: `build-arm64-ship` (green),
`build-arm64-prof` (green, 0 files to rebuild), `build-x86` (green), `ik_llama.cpp/build-arm64-ik` and
`build-arm64-ik-prof` (green, unchanged in Part 2).

Verification summary for the whole session:

| gate | evidence | result |
|---|---|---|
| `BONSAI_MOE_X4=1` | `s6_q2_0_x4_test.c` under qemu, 8 sizes | **bit-exact** vs the shipped reference |
| `BONSAI_MOE_X4=2` | same test | near-exact, <= 2.9e-7 relative (34x inside Gate 3(a)); Gate 3(b) statistical test not run, so off by default (12.6) |
| `BONSAI_Q3K_X4=1` | `s6_q3_K_x4_test.c` under qemu, 6 sizes incl. n=2560 | **bit-exact** vs the shipped reference, cross-checked against the scalar implementation |
| `BONSAI_MULMATID_FLAT=2` | `s6_mmid_prefill_verify.sh`, 6 shapes incl. 16 tokens / 8 threads | **byte-identical** to stock, both modes stable |
| fused up+gate op | not implemented (section 10.4) | recipe and decision rule recorded instead |

Nothing was committed; no `adb` command was run in either part of this session.

---

# PART 3: pre-shootout audit fixes (applied on the host, no phone)

Four of the audited items are applied and verified below. **Item 5 was cut off in the request**
(the message ends mid-sentence at "Document the finding in `STAGE6-daytime-"), so it is not applied;
section 12.5 lists what I would propose for it plus two extra confounds I found while auditing.

## 12. Applied fixes

| # | audit finding | status |
|---|---|---|
| 1 | `B1` missing from the decode pass, so `B0 - B2` conflated `--poll 0` with `--no-fused-moe` | **fixed** (12.1) |
| 2 | silent `batch_pool = false` defaults in `ik_llama.cpp` | **fixed** (12.2) |
| 3 | identity gate lacked `BONSAI_MOE_X4=2` and a long-prompt prefill-queue check | **fixed** (12.3) |
| 4 | `-ctxcp 0` asymmetry between the engines | **fixed, and the finding is bigger than expected** (12.4) |
| 5 | not received | pending (12.5) |

### 12.1 Fix 1: `B1` restored to the decode pass, and the confound closed

The audit is right and the confound was real: `B0` runs without `--poll 0` while `B2` carries
`--poll 0 ... --no-fused-moe`, so the previous `B0 - B2` pair on decode measured *both* changes at
once.

* `decode_pass` now runs `A0 A1 A2 A3 A4 B0 B1 B2` forward and `B2 B1 B0 A4 A3 A2 A1 A0` reverse.
* `s6_ll_table.py` gained a **named contrasts** table, because the existing A0-relative table cannot
  express these pairs: on decode it reports `B1 vs B0` (the pinning/`--poll 0` flag delta) and
  `B2 vs B1` (the `--no-fused-moe` delta **at identical flags**); on prefill it reports `B1 vs B0`
  (the `0xff` batch mask). The table prints the sign convention inline so a negative "fusion OFF"
  delta is not misread.
* Added at the same time: `A4` = `MOE_X4=1 + Q3K_X4=1` on decode, so the night also answers "what do
  our two bit-exact kernels buy *together*", which is the number the ship decision actually needs.

### 12.2 Fix 2: no silent defaults left in `ik_llama.cpp`

Both `llama_graph_compute` (`src/llama.cpp::7244`) and `llama_graph_compute_sched` (`:7268`) had
`bool batch_pool = false` removed from their definitions. All five call sites are now explicit:

| call site | what it computes | argument |
|---|---|---|
| `:7760` | main ubatch graph | `n_tokens != 1` (pre-existing, unchanged) |
| `:8158` | main graph, other path | `n_tokens != 1` (pre-existing, unchanged) |
| `:8453` | `llama_build_graph_defrag` | `false` (explicit) |
| `:8493` | `llama_build_graph_k_shift` | `false` (explicit) |
| `:8520` | `llama_build_graph_s_copy` | `false` (explicit) |

The three previously-defaulting sites are **maintenance graphs** (KV defrag, KV shift, recurrent
state copy), not batch matmuls, so `false` reproduces the previous behaviour exactly; the fix removes
the silent default without changing any measurement. Each site carries a one-line comment saying why.

Two findings worth recording:

* `llama_graph_compute_sched` in `llama.cpp` has **no callers in that translation unit** (it is
  `static`, so it cannot be called from elsewhere). The same-named static in
  `src/llama-dflash.cpp:222` is a *different* function, called at `:614`, and is unrelated to the
  Stage 6 change. Removing the default there is therefore documentation only.
* Both `build-arm64-ik` and `build-arm64-ik-prof` rebuilt clean (9.1 s each, 0 errors, no new
  warnings). Both patches were regenerated and validated: `stage6-ik-all.patch` (674 lines, includes
  the `BONSAI_PROF` port) and `stage6-ik-pinning.patch` (517 lines, **0** `BONSAI_PROF` references,
  still reverse-applies against the tree, so it remains a provably valid subset).

### 12.3 Fix 3: identity gate now covers mode 2 and a real 256-token prefill

* `id-x4p` (`BONSAI_MOE_X4=2`) is in the gate. Mode 2 is **near-exact by construction** (<= 2.9e-7
  relative per call under qemu), so it is *reported, not gated*: the driver prints the first token
  index where it diverges, which is the useful fact - "identical on this prompt" is a bonus, not a
  guarantee. Mode 2 is now formally judged by the new **Gate 3** in `GOAL.md`: (a) max relative error
  <= 1e-5 against the double-precision oracle under qemu (satisfied: <= 2.9e-7, a 34x margin), and
  (b) a 32-chunk paired PPL delta statistically indistinguishable from zero (`|t| < 2.0` and paired
  mean <= +0.25%, about 1 SE) - not yet run. The first-token divergence index is logged for the
  record. Gate 3(b) is deliberately statistical rather than a fixed <= +0.05%: this test's own SE is
  0.23-0.37% (section 1.2), so a 0.05% threshold is 0.2 SE and would reject a true zero effect about
  42% of the time. And "coherent continuation" was dropped as a criterion - nothing in this project
  is judged by reading generated prose. The divergence index itself is computed on the host by
  `id_div()` in `s6_shootout.sh`, from the token ids each arm prints (`S6IDTOK`).
* Long-prompt check added for the prefill queue: the arm script gained `IDENT_REPS` / `IDENT_N`, and
  the driver runs `id-mm1` (`MULMATID_FLAT=1`), `id-mm2` (`=2`) and `id-mm2b` (repeat control) at a
  **~256-token prefill + 16 greedy tokens**. This matters because the queue only engages when
  `ids->ne[1] > 1`, so a 6-token identity prompt could not have exercised it at all.
* The arm script now also prints the token ids (`S6IDTOK`), and the analyzer reports the **first
  divergent index** instead of only "differs", grouped by prompt family so a long-prompt arm is never
  compared against a short-prompt baseline (that was a real bug in my first version of this table: it
  showed the long-prompt arms as "DIVERGES at 0" purely because the prompt differed).

### 12.4 Fix 4: the `-ctxcp` asymmetry, and it was worse than a spelling difference

**Both engines have context checkpointing; only the spelling differs, and the defaults were not
aligned:**

| | ours | ik |
|---|---|---|
| short option | `-ctxcp` | `-ctx-ckpt` |
| long options | `--ctx-checkpoints`, `--swa-checkpoints` (`upstream-llama/common/arg.cpp:1701`) | `--ctx-checkpoints` (`ik common/common.cpp:3213`) |
| default | not passed by us, but our arms pass `-ctxcp 0` explicitly | **32 checkpoints** (`ik common/common.h:587`), interval 512, tolerance 5 |
| family | n/a | also `--ctx-checkpoints-interval/-tolerance/-spill-dir/-live-n` |

So the asymmetry the audit suspected is real: our arms ran checkpointing **off** while ik ran with its
default **32**. That applied to the Stage 5 oracle numbers as well, since that comparison also passed
`--cpu-mask` and friends but nothing about checkpoints. Fixed by adding `-ctx-ckpt 0` to ik's flag set
in `s6_arm_ll.sh`, with the reasoning in a comment at the flag.

**Audit note, recorded as requested, with one factual correction to the framing.** The operational
rule to carry forward is: **never pass `-ctxcp` to `ik_llama.cpp`** - that short spelling is our
fork's (`upstream-llama/common/arg.cpp:1701`), while ik's equivalent is `-ctx-ckpt`
(`ik common/common.cpp:3213`, whose long form is also `--ctx-checkpoints`). The correction is that ik
does **not** lack the option: it carries the same context-checkpoint cache with a **default of 32**
checkpoints (`ik common/common.h:587`), interval 512, tolerance 5. So "our fork runs `-ctxcp 0` to
match `ik_llama.cpp`" is true only once ik is *also* given `-ctx-ckpt 0`; without that, our explicit 0
was being compared against ik's default 32, and that is precisely the asymmetry this fix closes.
Both engines' versions of this mechanism are context/SWA checkpoint caches, not the `qwen4exp`
recurrent-state path; only the spelling and the default ever differed. Whether the option landed
upstream before or after ik's fork point was not something I could establish from the working trees,
so it is recorded as the requester's context rather than as a verified fact.

**Consequence, stated prominently: the Stage 5 oracle is a FLOOR, not a target.** The headline
`8.61 t/s` decode / `11.98 t/s` prefill were measured with ik's checkpointing at its default of 32
while our fork ran `-ctxcp 0`, so ik was paying overhead ours was not. With both engines now running
checkpointing off, ik can only move **up**, never down - so every gap percentage computed against
8.61/11.98 is a *lower bound* on the work still to do. Two honest caveats: the magnitude of the
effect is unmeasured (with `interval = 512` and `tolerance = 5`, a 262-token prefill may trigger few
or zero checkpoints, in which case the correction is small), and tonight's `B0`/`B1` arms re-measure
ik under the matched flags, which supersedes the Stage 5 number for gap purposes. Use Stage 5's
figure only as the floor it now is.

Two clarifications, because "checkpointing" is overloaded here:

* This is **not** the `qwen4exp` recurrent-state handling the brief worried about. It is the
  prompt/context checkpoint cache (snapshot and reuse of context state), an unrelated mechanism that
  exists in both engines.
* `-ctxcp` is **not** an ik option. Related forensic note: the Stage 5 loader at
  `.stage5-ikload.sh` passes `-ctxcp 0` to ik, which ik does not accept (ik accepts `--cache-type-k`
  and `--cache-type-v`, so those two flags were fine). That script is inconsistent with the command
  actually documented as successful in `STAGE5-report.md` section 1.3, and the report's command is the
  one consistent with the recorded runs.

### 12.5 Fix 5, and two extra confounds I found while auditing

**Fix 5 was truncated in the request** (the message ends at "Document the finding in
`STAGE6-daytime-"), so nothing was applied for it. Send it and I will apply it immediately.

In the meantime, the audit pass over the same four files turned up two more asymmetries that are
candidates for it, both cheap to close:

1. **`-ub` is passed to ik but not to us.** `s6_arm_ll.sh` gives ik `-ub 512` explicitly while our arm
   relies on the default. Stage 4 measured `-ub` as non-monotonic with a ~19% spread and called 256
   and 128 traps, so if the two engines' defaults ever differ the comparison is invalid. They agree at
   512 today, so this is hygiene rather than a live bug - but it should be pinned explicitly on both
   sides.
2. **`--no-fused-moe` disables only the *MoE* up+gate fusion.** ik has a separate
   `--no-fused-up-gate` / `-no-fug` for the dense up-gate fusion (`common/common.cpp:2259`), which
   `B2` leaves enabled. That is the right behaviour for isolating the MoE fusion, but the report must
   say so, otherwise `B1 - B2` could be misread as "ik without fusion" rather than "ik without the MoE
   fusion".

Not confounds, checked and left alone: `-fa on`, `--jinja`, `-c 8192`, `-np 1`, `-t 6`, `-tb 8`,
`--cpu-mask 0x3f`, q8_0 KV on both sides, and the model file (one file, sha256-verified by the
driver). ik's `--prefetch-experts` and our `BONSAI_MOE_PREFETCH` both default to off, so MoE prefetch
is symmetric; the driver records each engine's own configuration lines (`S6CONF`) so this is verified
from the run rather than assumed.

### 12.6 Gate 3 added to `GOAL.md` (documentation only)

`GOAL.md` defined two correctness gates, and `BONSAI_MOE_X4=2` (arm `A2`, packed accumulate) fitted
neither: it changes FP summation order by construction, so it can never pass a bit-exact token-`md5`
gate, and it is not a weight requantisation either. A third gate class was added for that case:

> **Gate 3 - Near-Exact / Re-Associated Kernel Changes (e.g. packed or fused FP32 accumulation):**
> **both** parts must pass. **(a)** max relative error `<= 1e-5` across all tested row sizes against
> the double-precision oracle under `qemu-aarch64-static`. **(b)** the 32-chunk paired perplexity
> delta must be **statistically indistinguishable from zero**: `|t| < 2.0` **and** paired mean delta
> `<= +0.25%` (about 1 SE), with the first-token divergence index logged for the record.

The `<= +0.25%` / `|t| < 2.0` pair replaced an earlier fixed `<= +0.05%` limit, which was 0.2 SE of
this test's own noise (SE 0.232-0.374%, section 1.2) and would have rejected a **true zero effect**
about 42% of the time - a gate that fails a correct change two runs in five is not a gate. The
original alternative, "or a coherent greedy continuation", was also dropped: judging a numeric change
by reading generated prose is exactly what this project's rules forbid.

Status against it, from this session's evidence:

| gate | member | evidence | state |
|---|---|---|---|
| Gate 1 (token identity) | `BONSAI_MOE_X4=1`, `BONSAI_Q3K_X4=1`, and arm **`A4`** (both together) | `0.000e+00` max abs deviation vs the shipped reference under qemu, on every tested size | **(a) satisfied**; device token identity is tonight's `id-*` arms |
| Gate 3 (near-exact) | `BONSAI_MOE_X4=2` only | part (a): `<= 2.9e-7` relative across `nb` 1-64, i.e. 34x inside the 1e-5 limit; part (b): not run | **open -> stays off by default** |

The important consequence for tonight: **arm `A4` (`BONSAI_MOE_X4=1` + `BONSAI_Q3K_X4=1`) qualifies
directly under Gate 1**, because both kernels it enables are bit-exact rather than near-exact. Only
`A2` needs Gate 3, and only `A2` needs the divergence-index reporting that `s6_shootout.sh` now
prints. `GOAL.md`'s gate list, its summary table and its report template were updated to match, and
the numbering of the non-correctness rules after it shifted by one (old 3-8 are now 4-9).

### 12.7 Fix 5 (received in full this time): loud `libc++_shared.so` staging, and flag symmetry

**What was wrong.** Four staging lines (driver lines 84 and 91, plus two in the profiler pass) did

```sh
adb shell "cp $D/lib-flat/libc++_shared.so $DEST/libc++_shared.so 2>/dev/null" >/dev/null 2>&1
```

`/data/local/tmp/lib-flat/` is a Stage 3 artefact, and every failure mode of that command was
suppressed: missing source, failed copy, permission error. A missing library would then have surfaced
much later and much less clearly, as an engine that starts and dies.

**What it does now.** `LIBCXX` points at the NDK sysroot
(`.../sysroot/usr/lib/aarch64-linux-android/libc++_shared.so`, verified present, 9,236,352 B), and a
`push_libcxx <dest>` helper is used by all four sites. It fails the run with a named FATAL if the
host file is missing, if the `adb push` fails, or if a device-side `[ -f ... ]` check does not
confirm the file landed - i.e. staging aborts before an arm is attempted instead of degrading into
"S6FAIL engine died".

**Finding while applying it.** Neither engine currently *needs* the file. All twelve artefacts the
driver pushes have **zero** `libc++_shared.so` entries in their dynamic sections (`llvm-readelf -d`),
because both NDK builds link libc++ statically:

| artefacts checked | NEEDs `libc++_shared.so` |
|---|---|
| ours: `libggml-base`, `libggml-cpu`, `libggml`, `libllama`, `libllama-common`, `libllama-server-impl`, `libmtmd`, `llama-server` | 0 of 8 |
| ik: `libggml`, `libllama`, `libmtmd`, `llama-server` | 0 of 4 |

So the silent `cp` could not have caused an engine death for *these* builds, and Fix 5's value is
forward-looking: it makes a future dynamic dependency fail loudly at staging time. That is worth
having, and it is now the behaviour.

**Also in this pass (request item 2).** `-ub` is pinned explicitly on **both** engines via
`UB=${UB:-512}` -> `-ub $UB`, and ik uses the long spelling `--ctx-checkpoints 0` (identical to
`-ctx-ckpt 0`, `ik common/common.cpp:3213`). The two flag sets are now:

| | ours | ik |
|---|---|---|
| model / host / port / ctx | `-m ... --host 127.0.0.1 --port ... -c 8192` | same |
| decode threads | `-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1 --poll 0` | `-t 6 -tb 8 --cpu-mask 0x3f` (+ `--cpu-strict 1 --poll 0` with the Stage 6 patch) |
| batch pool | `--cpu-mask-batch 0xff --cpu-strict-batch 1` | `--cpu-mask-batch 0xff --cpu-strict-batch 1` (with the patch) |
| microbatch | `-ub 512` (now explicit) | `-ub 512` |
| KV cache | `--cache-type-k q8_0 --cache-type-v q8_0` | `-ctk q8_0 -ctv q8_0` |
| context checkpoints | `-ctxcp 0` | `--ctx-checkpoints 0` |
| other | `-fa on -ngl 0 --jinja -np 1` | same (`-fa on -ngl 0 --jinja -np 1`) |

The only remaining asymmetries are the ones under test (batch-pool pinning on ik, and the kernels),
which is the point.
