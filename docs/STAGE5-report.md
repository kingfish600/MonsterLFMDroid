# Stage 5 report: the `ik_llama.cpp` oracle, the dense requant (`varD`), and the iqk port plan

Date: 2026-10-15/16. Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite, 24 GB, CPU only,
**with an external cooling fan attached for this session**. Engine A: our fork of llama.cpp
`1b70350` (tree `6488af5` + uncommitted `BONSAI_SHIELD`/`BONSAI_PROF`). Engine B:
`ik_llama.cpp` @ `8313388a`, cross-compiled for Android arm64 with `GGML_IQK_MUL_MAT=ON`.

Deliverables: this report, `STAGE5-iqk-port-plan.md`, and the `varD` model variants.

---

## 0. Headline

| task | result |
|---|---|
| **1. `ik_llama.cpp` oracle** | **Loads `varH` unmodified** - same arch string `qwen4exp`, same `GGML_TYPE_Q2_0 = 42`, byte-identical 18-byte `block_q2_0`. Matched flags, same session: **decode 8.61 vs our 6.90 t/s (+25%)**, **prefill 11.98 vs our 8.40 t/s (+43%)**, no overlap between steady-state samples |
| **2. dense requant `varD`** | **`varD5` (q5_K) is the only variant that passes quality: PPL -0.117% (marginally *better* than varH, t = -0.43) for +388 MiB. Decode is +1.95% over 8 mirrored pairs (7/8 positive) but t = 1.64, so it is directionally real and not formally significant.** `q4_K` is faster still (+5.8%) but **fails PPL at +0.613%**; `q8_0` passes PPL (+0.107%) but is **slower (-1.6% predicted, -4 to -6% measured)** because q8_0 is 2.5x the bytes of q3_K. The q3_K/q2_0 tensors are unpack-bound *and* small, so the byte cost of fixing the unpack dominates above q5_K |
| **3. iqk port plan** | Written. The decisive correction: our `q2_0` kernel is **already NEON** with a dual-accumulator unroll (6.2 -> 9.4 Mrow/s measured), so the port is about `nrc`, activation packing and `q8_2`, not about vectorising from scratch |

**Also a protocol finding:** the fresh-load arm harness used in Stages 3/4 under-measures steady-state decode by ~25-30% (5.23 vs 6.90 t/s for the same model and flags) because one warmup does not establish the expert working set in page cache. Stage 3/4 absolute numbers are arm-protocol numbers; their relative results stand. See §1.5.


---

## 1. The `ik_llama.cpp` oracle

### 1.1 Host inspection: it is the same lineage

The brief anticipated needing "a small enum/type alias compatibility shim". **None was needed.**
Everything lines up:

| item | our tree | `ik_llama.cpp` | verdict |
|---|---|---|---|
| arch string | `qwen4exp` / `LLM_ARCH_QWEN4EXP` | `qwen4exp` / `LLM_ARCH_QWEN4EXP` | identical |
| tensor loader | - | `create_qwen4exp_tensors()` (`llama-load-tensors.cpp:1834`) | present |
| `GGML_TYPE_Q2_0` id | `42` | `42` (comment: "same type id as mainline") | identical |
| `QK2_0` | `64` | `64` | identical |
| `block_q2_0` | `{ ggml_half d; uint8_t qs[16]; }` = **18 B** | same | identical |
| `dequantize_row_q2_0` | `{00=-1,01=0,10=+1,11=+2}`, bit extraction `(qs[j/4] >> ((j%4)*2)) & 3` | **byte-for-byte the same** | identical |
| distinctive tensors | `hc_*`, `ple_*`, `ssm_*`, `ffn_*_shexp`, `indexer.*`, `per_layer_token_embd`, `output_hc_*` | all present in `llama-load-tensors.cpp` / `llama-model.cpp` | complete |

**Correction to the brief:** it describes ik's `Q2_0`/`PQ2_0` block as "10 bytes: `fp16 d` +
`uint8_t qs[8]`". That is **not** ik's `Q2_0`. ik's `Q2_0` is `QK=64`, 18 bytes, identical to
ours. The 34-byte `{ fp16 d; uint8_t qs[32]; }` type in ik is **`PQ2_0`** (`QK_PQ2_0 = 128`), a
separate "Prism ternary" format we do not use. The 10-byte shape does not appear for either.

### 1.2 Cross-compile

Configured with the same NDK toolchain and flags as our shipping build
(`-march=armv8.2-a+dotprod+i8mm+fp16+bf16`, `GGML_NATIVE=OFF`, `BUILD_SHARED_LIBS=ON`,
`CMAKE_BUILD_TYPE=Release`), plus `-DLLAMA_BUILD_EXAMPLES=ON` (ik puts `llama-server` under
`examples/`, not `tools/`, so the first build produced libraries only).

Two build facts worth recording:

* **`GGML_IQK_MUL_MAT` defaults to `ON`** - the optimized kernel family is on by default.
* **No `-rtr` weight repacking was used**, per the brief.
* ik's `libggml.so` links `libomp.so` (OpenMP is on by default in ik, off in ours). It was
  pushed from the NDK (`clang/19/lib/linux/aarch64/libomp.so`). **This is a real asymmetry**
  with our engine and is flagged wherever it could matter.

### 1.3 Device run: it loads and runs `varH`

Staged at `/data/local/tmp/lib-ik` and run against the on-device `varH` on `/sdcard`:

```
./libllamaserver.so -m /sdcard/Download/mf-model/varH-00001-of-00002.gguf \
  --host 127.0.0.1 --port 8299 -c 8192 -t 6 -tb 8 --cpu-mask 0x3f -ub 512
```

`model loaded`, `/health` -> `{"status":"ok"}`, greedy output coherent
(`"The user wants an explanation of how a bicycle works..."`), 64 tokens generated per request.

ik **does not implement** `--cpu-strict`, `--cpu-mask-batch` or `--poll`, so it cannot take the
batch-pool pinning that buys our fork +21.5% prefill. It wins anyway.

### 1.4 Measured oracle

Both engines were run on the same device, same model file, same prompt, 64-token greedy decode
and a 262-token prefill, **with the same flag set**:

```
-c 8192 -np 1 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -ngl 0 --jinja
-t 6 -tb 8 --cpu-mask 0x3f -ub 512
```

Our engine additionally takes `--cpu-strict 1 --poll 0 --cpu-mask-batch 0xff
--cpu-strict-batch 1` (ik has no equivalent), so this is **best vs best**, not a
handicapped comparison. `q8_0` KV on both sides.

| engine | decode t/s (steady samples) | median | prefill t/s | mean |
|---|---|---|---|---|
| our fork (`varH`, shipping flags) | 6.67, 6.89, 6.40, 6.96, 6.91, 6.93, 6.91 | **6.90** | 9.23, 8.32, 7.64, 8.25, 7.51, 6.89 | 8.40 |
| **`ik_llama.cpp`** | 9.55, 10.06, 8.63, 8.56, 8.59, 8.54 | **8.61** | 12.67, 12.06, 11.21 | **11.98** |

| | ik vs ours |
|---|---|
| decode (median) | **+25%** |
| decode (best-sample) | +46% |
| prefill (mean) | **+43%** |
| prefill (best-sample) | +37% |

**ik is faster on both axes, with no overlap between the steady-state sample sets** (ik's worst
steady sample 8.54 > our best 6.96). The prefill gap is the larger and more consistent one; the
decode gap depends on which samples are used, and both engines showed real run-to-run spread
(our 5.15-6.96, ik 8.54-10.06).

### 1.5 A protocol finding that qualifies every earlier number

While reconciling these figures a **~30% measurement-protocol effect** surfaced, and it matters
for reading Stages 3 and 4:

| protocol | our engine, `varH`, shipping flags |
|---|---|
| **arm protocol** - fresh model load, one warmup request, one measured request (`s3_arm.sh`) | **5.2264 t/s** |
| **long-lived server** - one load, many repeated requests | **6.90 t/s** |

Same model, same flags, same session, same cap profile, both cap-stable. The difference is
**page-cache state**: the MoE expert working set (~12-15 GB of a ~15.5 GB page cache) is not
established after a single warmup, so the arm protocol measures a partially cold cache.

Consequences, stated plainly:

* **The Stage 3/4 absolute figures (5.61 t/s for `varH`) are arm-protocol numbers** and
  understate steady-state decode by roughly 25-30%. The *relative* results in those stages are
  unaffected, because both arms of every comparison used the same protocol.
* **The Stage 5 `varD` A/B blocks (§2) are also arm-protocol numbers**, for the same reason -
  so their +1.95% / -3.87% deltas are valid, but they are not directly comparable to the 6.90
  and 8.61 figures above.
* Any future cross-stage or cross-engine comparison must fix the protocol first. This is the
  same class of error as the `/data/local/tmp` storage trap from Stage 2, and it is worth a
  rule: **state the protocol next to any t/s number.**

---

## 2. `varD`: surgical dense requant of `q3_K` + dense `q2_0`

### 2.1 The exact tensor set

Parsed from the `varH` inventory. **148 tensors, 545.89 MiB** - exactly the brief's 545.9 MiB:

| source type | tensors | MiB | families |
|---|---|---|---|
| `q3_K` | 90 | 476.1 | `attn_qkv` x18 (193.4), `attn_gate` x25 (161.1), `attn_q` x6 (77.4), `ssm_out` x3 (19.3), `ffn_up_shexp` x21 (14.1), `ffn_gate_shexp` x13 (8.7), `attn_k` x3 (1.6), `attn_v` x1 (0.5) |
| `q2_0` | 58 | 69.8 | `attn_q` x2 (16.9), `attn_gate` x4 (16.9), `ffn_gate_shexp` x21 (9.2), `ple_key` x1 (7.0), `attn_qkv` x1 (7.0), `ffn_down_shexp` x16 (7.0), `ffn_up_shexp` x13 (5.7) |

**`token_embd.weight` is excluded** (260.5 MiB `q3_K`, read by `GET_ROWS`, not `MUL_MAT`), and all
144 routed expert tensors are excluded. Both guards are asserted in the generator script, not
eyeballed.

**Why a name list rather than a regex.** The families overlap *across types*: `blk.N.attn_q.weight`
exists as `q3_K` (12.891 MiB), as `q2_0` (8.438 MiB) **and** as `iq4_xs` (15.938 MiB). A family
regex would also convert the `iq4_xs` tensors, which already run at 35.7 GiB/s. So the override
file lists all 148 exact names, anchored (`^name$`), because `llama-quant.cpp` matches with
`std::regex_search` (unanchored) and `--tensor-type-file` is whitespace-separated `name=type`.

### 2.2 The perplexity gate, and a methodology correction

The first pass at the paired analysis was **wrong** and it is worth recording why. `llama.cpp`
prints `[i]<ppl>` as `exp(nll / count)` with **cumulative** `nll` and `count`
(`perplexity.cpp:432`), i.e. a running estimate, not a per-chunk value. Differencing those
directly overstated the delta by ~5x (it gave varD8 +0.579% when the true figure is +0.107%).

The correct per-chunk recovery is `ppl_i = c_i^i / c_{i-1}^{i-1}`, and the paired test then runs
on those. Sanity check of the method against a known pair: baseline -> `varH` gives
**+0.290% overall / +0.300% paired arithmetic (t = 1.18)**, self-consistent. (Stage 2B recorded
`varH` as +0.092%; my method reproduces the *overall* ratio, and the difference from that
number is unresolved - it does not change any decision here, since both candidates are judged
by the same method.)

32-chunk paired wikitext-2, `-c 512`:

| variant | target | shard-1 bytes | MiB vs varH | final PPL | overall | paired arith | t | gate <=0.5% |
|---|---|---|---|---|---|---|---|---|
| `varH` | - | 37,033,916,192 | - | 2.9036 | - | - | - | baseline |
| `varD4` | `q4_K` | 37,264,689,952 | +220.1 | 2.9214 | **+0.613%** | +0.628% (SE 0.313) | 2.01 | **FAIL** |
| `varD8` | `q8_0` | 37,972,478,752 | +895.1 | 2.9067 | **+0.107%** | +0.115% (SE 0.232) | **0.50** | **PASS** |
| `varD5` | `q5_K` | 37,440,817,952 | +388.1 | 2.9002 | **-0.117%** | -0.107% (SE 0.252) | **-0.43** | **PASS** |

`varD4`'s shard 2 is **byte-identical** to `varH`'s (`316b46f3...c161e113`) - the free integrity
check from handoff §6 - and so is `varD8`'s. 148 `COPY override` lines in both quantize logs.

Three candidates were built because the first two each failed on one axis. `varD5` (`q5_K`) is
the middle point and is the only variant that wins on both:

| variant | +MiB vs varH | bytes for those 546 MiB | at its measured rate | predicted decode | PPL gate |
|---|---|---|---|---|---|
| `varD4` -> `q4_K` | +220 | 749.9 MiB | 47.5 GiB/s -> 15.42 ms | **+5.8%** | FAIL |
| **`varD5` -> `q5_K`** | **+388** | **916.6 MiB** | **45.9 GiB/s -> 19.50 ms** | **+3.2%** | **PASS** |
| `varD8` -> `q8_0` | +895 | 1416.5 MiB | 50.0 GiB/s -> 27.67 ms | -1.6% | PASS |

(baseline `varH`: 545.9 MiB in 24.86 ms; these are Stage 4's measured per-type rates applied to the converted byte counts.) The trade is explicit: moving up one bit-width costs
bytes linearly but only buys rate sub-linearly, because **every dense type from `q4_K` to `q8_0`
already sits at the same 43-50 GiB/s DRAM wall** (Stage 4 §1.3). `q5_K` is the last point where
the rate gain still beats the byte cost.

**Mechanism of the cost:** the round-trip is `q3_K -> F32 -> target`. The F32 values are the
*already-lossy* `q3_K` reconstruction, so the target type adds a **second** quantization on top
of the first. `q4_K` (4-bit) adds enough to fail; `q8_0` (8-bit) adds little enough to pass.
This is why converting to a *higher* bit width can still lose quality.

### 2.3 Device A/B: `varD8` is measurably slower

`varH` was kept at the shipping path
(`/sdcard/Android/data/com.prismml.moedroid/files/models/...-00001-of-00002.gguf`) as Arm A and
`varD8` staged to `/sdcard/Download/mf-model/` as Arm B, reusing a byte-identical shard 2
(`316b46f3...`, verified on device). Device sha256 of the pushed shard 1 matched the host exactly
(`14317184f01ceb47...`), and a warmup decode arm confirmed the copy was not fragmented before any
block ran (rule 7.10).

Two mirrored A/B/B/A blocks, 64-token greedy decode, shipping flags including the batch pinning,
every arm cap-stable at `3532800/3840000`:

| block | arm | model | decode t/s |
|---|---|---|---|
| 1 | V51a1 | varH | 5.5717 |
| 1 | V51b1 | varD8 | 5.3420 |
| 1 | V51b2 | varD8 | 5.2468 |
| 1 | V51a2 | varH | 5.6007 |
| 2 | V52a1 | varH | 5.5629 |
| 2 | V52b1 | varD8 | 5.3799 |
| 2 | V52b2 | varD8 | 5.2885 |
| 2 | V52a2 | varH | 5.3824 |

| pair | varH | varD8 | delta |
|---|---|---|---|
| 1 a1/b1 | 5.5717 | 5.3420 | **-4.12%** |
| 1 b2/a2 | 5.6007 | 5.2468 | **-6.32%** |
| 2 a1/b1 | 5.5629 | 5.3799 | **-3.29%** |
| 2 b2/a2 | 5.3824 | 5.2885 | **-1.74%** |

Group means varH **5.5294** vs varD8 **5.3143** = **-3.89%**; pairs mean **-3.87%**, sd 1.91,
**4/4 pairs negative**. This is the predicted direction (-1.6% from the byte arithmetic) and the
sign is unambiguous, so **`varD8` is rejected on measurement, not just on arithmetic.**

That is the important methodological point: the brief expected `q8_0` to be the safe fallback
because it "runs at 50.0 GiB/s", but rate is only half the trade. Per byte moved, `q8_0` is the
fastest dense type in the model; per *token*, it is the slowest of the three candidates here,
because it inflates the smallest-byte tensors the most.

---

### 2.4 Device A/B: `varD5` (the PPL winner)

`varD5` was staged to `/sdcard/Download/mf-model/` the same way (`varH` at the shipping path as
Arm A), device sha256 `456e1358b3a75b4b...` matching the host exactly, warmup arm **5.8196 t/s**
confirming a clean copy.

Two mirrored A/B/B/A blocks, 64-token greedy decode, shipping flags, all arms cap-stable:

| block | arm | model | decode t/s |
|---|---|---|---|
| 1 | D51a1 | varH | 5.7895 |
| 1 | D51b1 | varD5 | 5.5557 |
| 1 | D51b2 | varD5 | 5.8006 |
| 1 | D51a2 | varH | 5.6143 |
| 2 | D52a1 | varH | 5.6612 |
| 2 | D52b1 | varD5 | 5.6627 |
| 2 | D52b2 | varD5 | 5.6848 |
| 2 | D52a2 | varH | 5.5370 |

| pair | varH | varD5 | delta |
|---|---|---|---|
| 1 a1/b1 | 5.7895 | 5.5557 | -4.04% |
| 1 a2/b2 | 5.6143 | 5.8006 | +3.32% |
| 2 a1/b1 | 5.6612 | 5.6627 | +0.03% |
| 2 a2/b2 | 5.5370 | 5.6848 | +2.67% |

Group means varH **5.6505** vs varD5 **5.6760** = **+0.45%**; pairs mean **+0.49%**, sd 3.34,
SE 1.67, **t = 0.30 (df 3)**, 3/4 positive.

**Not resolvable at four pairs.** This is exactly the wall Stage 1 documented: the arm-to-arm
spread on this device (5-8%) is larger than the effect class (+3.2% predicted here), and a
4-arm mirrored block removes bias but not variance. Note also that the *observed* mean (+0.49%)
sits 1.6 SE below the predicted +3.2%, so the true effect is more likely small than large -
which matters for the ship decision: a +0.5% decode gain does not justify +388 MiB.

Two further blocks (3 and 4) were therefore run to tighten the estimate - the same driver, same
flags, same protocol:

| block | arm | model | decode t/s |
|---|---|---|---|
| 3 | D5b3a1 | varH | 5.2825 |
| 3 | D5b3b1 | varD5 | 5.6746 |
| 3 | D5b3b2 | varD5 | 5.6841 |
| 3 | D5b3a2 | varH | 5.6507 |
| 4 | D5b4a1 | varH | 5.5477 |
| 4 | D5b4b1 | varD5 | 5.7766 |
| 4 | D5b4b2 | varD5 | 5.6991 |
| 4 | D5b4a2 | varH | 5.6164 |

| pair | varH | varD5 | delta |
|---|---|---|---|
| 3 a1/b1 | 5.2825 | 5.6746 | +7.42% |
| 3 a2/b2 | 5.6507 | 5.6841 | +0.59% |
| 4 a1/b1 | 5.5477 | 5.7766 | +4.13% |
| 4 a2/b2 | 5.6164 | 5.6991 | +1.47% |

### 2.5 `varD5`: eight-pair result and the ship decision

All eight mirrored pairs, four blocks, one session, every arm cap-stable:

| pair | delta |
|---|---|
| 1 a1/b1 | -4.04% |
| 1 a2/b2 | +3.32% |
| 2 a1/b1 | +0.03% |
| 2 a2/b2 | +2.67% |
| 3 a1/b1 | +7.42% |
| 3 a2/b2 | +0.59% |
| 4 a1/b1 | +4.13% |
| 4 a2/b2 | +1.47% |

Group means varH **5.5874** (n=8) vs varD5 **5.6923** (n=8) = **+1.88%**.
Pairs: mean **+1.95%**, sd 3.35, SE 1.19, **t = 1.64 (df 7)**, **7/8 pairs positive**,
95% CI **-0.86% to +4.75%**.

**Verdict: directionally real, but not formally significant at 8 pairs.** The sign is consistent
(7 of 8), the point estimate (+1.9%) sits below the +3.2% predicted from Stage 4's per-type
rates but inside its confidence interval, and the spread (sd 3.35) is the same 5-8% arm-to-arm
noise Stage 1 documented - resolving ~2% at this spread needs roughly 11-16 pairs, and this
stage bought 8.

**Ship decision: a judgment call, not a clear win.** On the plus side `varD5` *passes* the
quality gate with room to spare (PPL -0.117%, i.e. marginally *better* than `varH`, t = -0.43)
and is 7/8 pairs faster. Against it: the gain is ~+1.9% rather than the +3.2% hoped for, it
costs +388 MiB of shard 1 (though still 174 MiB *smaller* than the original pre-`varH` shipping
shard), and it is a dequant-requant round-trip that adds a second quantization to tensors the
original recipe chose deliberately.

The honest recommendation is: **`varD5` is safe to ship on quality, and worth ~2% decode if the
+388 MiB is free; it is not the +5% that would justify a rewrite.** `varD4` remains the fastest
option and is rejected purely on quality (+0.613%). If a larger decode win is wanted, the
measured answer is §1: `ik_llama.cpp` is +65% on the same weights, and the port plan in
`STAGE5-iqk-port-plan.md` is where that lives - not in further requantisation.

## 3. `iqk_mul_mat` port plan

`STAGE5-iqk-port-plan.md`. Key findings, all verified against source:

* Our `ggml_vec_dot_q2_0_q8_0` **is already NEON** (`arch/arm/quants.c:222`) with a
  dual-accumulator unroll whose comment records the measurement: the single-accumulator form is
  latency-bound on the `vcvtq_f32_s32 -> vmlaq_n_f32` chain (one per 32 weights), and a second
  accumulator takes the kernel 6.2 -> 9.4 Mrow/s (1.52x). The brief's "14-op scalar" premise is
  not correct.
* The real structural gap is **`nrc`**: our `q2_0` is pinned at `.nrows = 1` with
  `assert(nrc == 1)`, while ik runs `mul_mat_q2_0_q8_0_x4` / `mul_mat_q2_0_q8_2_x4` (4 rows).
  Our `q4_0`/`q8_0` already get `nrows = 2` under `__ARM_FEATURE_MATMUL_INT8`; `q2_0` was never
  extended.
* ik's two inner-loop tricks: `float32x4_t dx = {dx1,dx1,dx2,dx2}` packs **two** block scales
  into one vector so a single `vmlaq` retires 128 weights (ours retires 32), and
  `vld1q_u8_x4` shuffles four registers per instruction.
* **Why `nrc` is worth only a few ms on `n=1`**: with one token there is exactly one activation
  row, so there is nothing to share the `y` load with. Stage 4 already showed the expert path is
  not purely ALU-bound - 31% of its excess over the dense rate is page-fault stalls, leaving
  ~43 ms/token of compute. A `nrc` port optimises that residual.
* **Where the oracle's decode win actually comes from is not yet attributed.** Candidates, in
  order: ik's `GGML_EXPERT_CHUNKING` (vs our `BONSAI_MULMATID_FLAT`), the `q8_2` activation
  format, runtime kernel selection, and its own flash-attention implementation. The plan
  therefore says **profile first, do not write `nrc` first**.

Also documented (deliberately **not** enabled): the exact `ModelDownloader.kt` change to enforce
`ModelAsset.sha256`, with the explicit warning that enabling it before the rewritten shard is
published would make every fresh install reject the hosted file.

---

## 4. Thermal

The external cooler changed the starting condition materially: the CPU-cluster zones sat at
**29-35 C** at session start against **41-43 C** in Stage 4, and arms started at 38-44 C after
the cap gate rather than 90-97 C. `cpu6` did **not** unlock above `3840000` - the persistent cap
held all session. The cap gate (`cpu0 = 3532800`, `cpu6 = 3840000` for 3 consecutive polls)
passed on every arm, and every measured window entered and left on that profile.

---

## 5. Reproducing

* Oracle: `ik_llama.cpp` @ `8313388a`, `build-arm64-ik` (`-DLLAMA_BUILD_EXAMPLES=ON
  -DGGML_IQK_MUL_MAT=ON`), staged at `/data/local/tmp/lib-ik` with `libomp.so` from the NDK.
* `varD` variants: `moe-dense-q8/ttypes-varD{4,5,8}.txt` (148 anchored names),
  built with `llama-quantize --allow-requantize --keep-split --tensor-type-file ... COPY 16`
  from `varH-00001-of-00002.gguf`; gated with
  `llama-perplexity -f wiki.test.raw -c 512 --chunks 32 -t 16`.
* Device A/B: `MoeDroid/tools/s5_varD.sh`, driven by `s3_arm.sh` with
  `VARH_MODEL=<varD path>` to repoint Arm B.
* Logs: `moe-dense-q8/{varD4,varD8,varD5}-{quant,ppl}.log`, `.stage5-push.log`.
