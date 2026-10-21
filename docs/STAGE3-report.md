# Stage 3 report: banking `varH`, batch-thread affinity, and prefill eviction shielding

Date: 2026-10-14/15. Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite, 24 GB, CPU only.
Engine: fork of llama.cpp `1b70350`; tree at `6488af5` plus the Stage 3 `BONSAI_SHIELD`
addition in `src/llama-context.cpp` / `src/llama-context.h`. arm64 build
`armv8.2-a+dotprod+i8mm+fp16+bf16`, staged on the device at `/data/local/tmp/lib-shield`.
Shipping run flags held constant unless an arm says otherwise:
`-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1 --poll 0 -c 8192 -ctxcp 0 -np 1 -fa on
--cache-type-k q8_0 --cache-type-v q8_0 -ngl 0`.

Both models live on `/sdcard` (rule 7.10). Device sha256 of the staged copies:

| file | sha256 |
|---|---|
| `varH-00001-of-00002.gguf` | `e7c694526e82f633eda7f37b5e35ed7075fd39140018c99d73e305818ded12e2` |
| `varH-00002-of-00002.gguf` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Both match the host, and shard 2 is byte-identical to the source shard 2, so the copy path
is byte-exact (handoff §6 integrity check). `/sdcard` supports neither symlinks nor hard
links (both return `EPERM`), so shard 2 could not be linked and was copied on-device
(28,800,138,432 B in 37 s).

## 0. Headline

| finding | result | confidence |
|---|---|---|
| **`varH` banked** | decode **+8.19%** group mean, 4/4 pairs positive | t = 7.09 (df 3), clears the noise floor |
| **Batch (prefill) pool affinity** | the prefill pool does **not** inherit `--cpu-mask 0x3f`; it runs unpinned across `0xff`. Adding `--cpu-mask-batch 0xff --cpu-strict-batch 1` is worth **+21.5% prefill**, and in a realistic prefill+decode request **+17.9% prefill with decode +3.5% (no regression)** | 4/4 pairs positive across two independent prefill blocks, t = 9.08 (df 3) |
| **Prefill eviction shielding** | the post-prefill decode dip is real (first 16 tokens at ~77% of steady state), but it is **I/O-volume-bound, not fault-bound**: `WILLNEED` on the dense tensors cuts decode-phase major faults 70-79% for +0.33% decode (n.s.); `MADV_RANDOM` on the experts costs **-33% prefill** | identity gate passes for all modes; the negative result is far outside noise |

The prefill affinity win is a **flag-only change** - no engine patch - and it is the single
largest result in this stage.

---

## 1. `varH` validated and banked

### 1.1 What `varH` actually converted

`moe-dense-q8/varH-quant.log` contains exactly **192** `COPY override` lines, all
`bf16 -> q8_0`. The full model inventory parses to 1,224 tensors (1,223 in shard 1 plus the
`per_layer_token_embd` table in shard 2):

| category | tensors | size |
|---|---|---|
| experts (`*_exps.weight`) | 144 | 32,400.0 MiB (31.64 GiB) |
| `blk.N.hc_{attn,ffn}_{up,down}.weight` - **converted to Q8_0** | 192 | 637.4 MiB (was 1,200.0) |
| other dense (attn/ssm/norms/routers/tables) | 884 | 2,257.6 MiB |
| `output_hc_{up,down,norm}` | 3 | 12.5 MiB |
| `per_layer_token_embd.weight` (shard 2) | 1 | 27,466.0 MiB |

The 192 converted tensors are the **complete set of `blk.*.hc_*_(up|down)` matrices**
(48 layers x 4). There are **two additional unmatched matrices**: `output_hc_up.weight`
and `output_hc_down.weight`, both `bf16` 2-D, 6.25 MiB each (12.5 MiB total). They exist in
shard 1 but the regex `hc_.*_(up|down)\.weight$` cannot match them: after the literal `hc_`
in `output_hc_down.weight` the next characters are `down.weight`, and the pattern still
requires a `_down`. They were **not** converted. `hc_*_inject.weight` (`ne00 = 4`) and
`hc_*_norm.weight` are not `_up`/`_down` matrices and are not candidates.

### 1.2 Warmup validation of the staged copy

One throwaway decode arm before any block, per rule 7.10:

| arm | model | decode t/s | minflt | majflt | cap samples |
|---|---|---|---|---|---|
| warm1 | varH | **5.6199** | 107,354 | 14,850 | 12/12 at `3532800/3840000` |

5.62 t/s is clear of the ~5 t/s bar and above the shipping model's Stage 2 numbers
(4.97-5.08), so the pushed copy is not fragmented.

### 1.3 Decode, mirrored A/B/B/A, thermal-gated

Two blocks, `A = shipping`, `B = varH`, 64-token greedy decode. Every arm re-ran the cap gate
before loading, and the cap pair was sampled every 5 s for the whole arm (load + warmup +
measured window). D1/D2 ran before the in-window `MEASWIN` marker was added, so for those
arms the 5 s sampler is the evidence - it covers more than the measured window, not less.

| block | arm | model | decode t/s | minflt | majflt | cap samples (cpu0/cpu6) | stable |
|---|---|---|---|---|---|---|---|
| D1 | D1a1 | shipping | 5.2162 | 113,839 | 17,099 | 3532800/3840000 x12, nothing else | yes |
| D1 | D1b1 | varH | 5.6004 | 107,904 | 15,955 | 3532800/3840000 x11, nothing else | yes |
| D1 | D1b2 | varH | 5.6271 | 105,414 | 13,957 | 3532800/3840000 x11, nothing else | yes |
| D1 | D1a2 | shipping | 5.0757 | 112,762 | 15,547 | 3532800/3840000 x12, nothing else | yes |
| D2 | D2a1 | shipping | 5.3384 | 115,484 | 15,669 | 3532800/3840000 x12, nothing else | yes |
| D2 | D2b1 | varH | 5.6314 | 110,149 | 16,340 | 3532800/3840000 x11, nothing else | yes |
| D2 | D2b2 | varH | 5.5950 | 104,664 | 13,857 | 3532800/3840000 x11, nothing else | yes |
| D2 | D2a2 | shipping | 5.1229 | 110,836 | 16,610 | 3532800/3840000 x12, nothing else | yes |

**All eight arms are cap-stable**: every arm shows exactly one distinct cap pair for its whole
duration, at the good profile (`cpu0-5 = 3532800`, `cpu6/7 = 3840000`), and no sample in any
arm fell to `2841600`. The cap gate reported `OK after 0 polls` for every arm, i.e. the device
was already back at the good profile when each arm started.

Same-profile mirrored pairs:

| block | pair | shipping | varH | delta |
|---|---|---|---|---|
| D1 | a1/b1 | 5.2162 | 5.6004 | **+7.37%** |
| D1 | b2/a2 | 5.0757 | 5.6271 | **+10.86%** |
| D2 | a1/b1 | 5.3384 | 5.6314 | **+5.49%** |
| D2 | b2/a2 | 5.1229 | 5.5950 | **+9.22%** |

Group means: shipping **5.1883**, varH **5.6135** = **+8.19%**.
Pairs: mean **+8.23%**, sd 2.32, SE 1.16, **t = 7.09 (df 3)**. All four pairs positive.

This is the first Stage 2/3 dense result that clears the noise floor on its own terms: the
Stage 1 report measured a 5-8% arm-to-arm spread and concluded that 4 mirrored pairs cannot
resolve a few percent. Here the effect (+8.2%) is larger than the spread and the pairs are
all the same sign, so it resolves. `majflt` is also consistently lower on varH
(13,857-16,340 vs 15,547-17,099 per 64 tokens), which is the expected direction for 562.5 MiB
less dense traffic.

**Banking recommendation: ship `varH`.** Host perplexity already passes at +0.092% (t = 1.02,
32 chunks), and the device decode win is +8.19% group mean with t = 7.09. Keep
`output.weight Q5_K -> Q4_0` dropped: the full rewrite (which included it) measured +2.82%
group mean over 8 arms in Stage 2, i.e. *worse* than varH alone at +8.19%, so that
conversion was negative on speed as well as on quality. Residual: 12.5 MiB of convertible
BF16 remains in `output_hc_up/down`.

---

## 2. `-tb 8` vs `-tb 6` under `--cpu-mask 0x3f --cpu-strict 1`

### 2.1 Answer: `threadpool_batch` does **not** inherit `0x3f`

The question was whether the 8 batch threads are oversubscribed onto the 6 physical cores
`cpu0..5`, or run unpinned across `0xff`. **They run unpinned across `0xff`** - the second
branch, and the opposite of oversubscription. The chain, all in this revision:

1. `common/arg.cpp:1533` - `-tb 8` sets `params.cpuparams_batch.n_threads = 8`.
2. `common/arg.cpp:886-887` - `postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams)`.
3. `common/common.cpp:294-301` - the whole `cpuparams` is copied into `cpuparams_batch`
   **only when `cpuparams_batch.n_threads < 0`**. With `-tb 8` that is false, so the copy is
   skipped and `mask_valid` stays `false`, `strict_cpu` stays `false`, `poll` stays 50.
4. `common/common.cpp:1760-1762` - `ggml_threadpool_params_from_cpu_params` copies the mask
   only `if (params.mask_valid)`, so `tpp_batch.cpumask` stays all-zero.
5. `ggml/src/ggml-cpu/ggml-cpu.c:2990-2993` - with `strict == false`,
   `ggml_thread_cpumask_next` does `memcpy(local_mask, global_mask)`, i.e. also all-zero.
6. `ggml-cpu.c:3589-3591` - affinity is applied only
   `if (ggml_thread_cpumask_is_valid(state->cpumask))`; an all-zero mask is invalid, so **no
   affinity is ever applied** and the batch workers keep the process's inherited mask.

The two pools are always distinct here because `ggml_threadpool_params_match`
(`ggml.c:8200-8206`) compares `n_threads` first (6 vs 8).

Two consequences worth recording:

* The CLI help text is wrong for this case. `--cpu-mask-batch` documents
  "(default: same as --cpu-mask)", but that default only applies when `-tb` is **not** given.
  The same holds for `--cpu-strict-batch` and `--poll-batch`: with the shipping flags the
  prefill pool silently runs non-strict and at the default `poll = 50` instead of `--poll 0`.
* The prefill pool therefore spans both clusters. Whether that costs anything is a
  measurement question, answered in 2.3.

### 2.2 Device confirmation

`s3_placement.sh` prints each thread's `Cpus_allowed_list` (the mask actually in force) and
samples `/proc/<tid>/stat` field 39 (awk field 37) every 200 ms during a 1,542-token prefill.

| config | threads with mask `0-7` | singly-pinned threads | ggml workers seen on cpu6/7 during prefill |
|---|---|---|---|
| `-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1` | **19 of 24** | 5 (cpu0-4) | **7** |
| same + `--cpu-mask-batch 0x3f --cpu-strict-batch 1` | 10 of 24 | 14 (cpu0-5) | **0** |

In the shipping config, seven threads share an identical CPU histogram, weighted to the
prime cluster (`tid 26606`: `0:18 1:16 2:22 3:15 4:25 5:15 6:92 7:66`, and six near-identical
siblings). Those are the batch pool's secondary workers, and they are the threads carrying
the prefill. Adding explicit batch affinity pins 14 threads across cpu0-5 and the ggml
workers stop touching cpu6/7 entirely; only four unrelated server threads still do.

### 2.3 256-token prefill, mirrored A/B/B/A: three configurations

256-token prompt, `n_predict 4`, varH, all arms cap-stable at
`3532800/3840000 -> 3532800/3840000` with no 5 s sample falling to `2841600`.

| config | description | prefill t/s (group mean) | vs shipping |
|---|---|---|---|
| **A** | `-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1` (shipping: batch unpinned) | 7.686 (TB1) / 7.465 (TB2+TB3) | baseline |
| **B** | `-t 6 -tb 6 --cpu-mask-batch 0x3f --cpu-strict-batch 1` (batch pinned to the 6-core cluster) | 7.792 (2 arms) | +1.38% (n.s., vs TB1's A arms) |
| **C** | `-t 6 -tb 8 --cpu-mask-batch 0xff --cpu-strict-batch 1` (batch pinned 1:1 across all 8 cores) | **9.072** (4 arms) | **+21.5%** (vs TB2+TB3's A arms) |

Configs B and C are each compared against their own block's A arms, which is the only valid
same-profile comparison; the two A baselines differ by 3.0% from each other, which is itself
a useful read on this device's block-to-block spread.

Block TB1 (A vs B):

| arm | spec | prefill t/s | minflt | majflt | stable |
|---|---|---|---|---|---|
| TB1a1 | tb=8 (unpinned) | 7.9995 | 501,695 | 187,303 | yes |
| TB1b1 | tb=6 pinned 0x3f | 7.8942 | 521,932 | 307,482 | yes |
| TB1b2 | tb=6 pinned 0x3f | 7.6896 | 536,463 | 364,468 | yes |
| TB1a2 | tb=8 (unpinned) | 7.3725 | 505,952 | 339,285 | yes |

| pair | tb=8 | tb=6 pinned 0x3f | delta |
|---|---|---|---|
| a1/b1 | 7.9995 | 7.8942 | -1.32% |
| b2/a2 | 7.3725 | 7.6896 | +4.30% |

Group means: tb=8 **7.6860**, tb=6-pinned **7.7919** = **+1.38%**.
Pairs: mean +1.49%, sd 3.97, SE 2.81, **t = 0.53 (df 1)**. The pairs disagree in sign.

**Confining prefill to the 6-core cluster is neutral.** Unlike decode, where `0xff`
measured 14-18% slower, moving prefill onto 6 cores changes nothing resolvable - which fits
the mechanism: prefill is compute-bound, so the two prime cores earn their keep there, while
decode is memory-bound and pays only the cross-cluster barrier cost.

Blocks TB2 and TB3 (A vs C, run as an independent repeat):

| arm | spec | prefill t/s | minflt | majflt | stable |
|---|---|---|---|---|---|
| TB2a1 | tb=8 (unpinned) | 7.4910 | 509,820 | 182,460 | yes |
| TB2b1 | tb=8 pinned 0xff | **9.2654** | 642,821 | 313,069 | yes |
| TB2b2 | tb=8 pinned 0xff | **9.0646** | 666,311 | 382,902 | yes |
| TB2a2 | tb=8 (unpinned) | 7.4435 | 519,752 | 304,913 | yes |
| TB3a1 | tb=8 (unpinned) | 7.8474 | 508,580 | 209,508 | yes |
| TB3b1 | tb=8 pinned 0xff | **9.0270** | 654,798 | 341,195 | yes |
| TB3b2 | tb=8 pinned 0xff | **8.9295** | 663,055 | 364,688 | yes |
| TB3a2 | tb=8 (unpinned) | 7.0782 | 540,255 | 292,935 | yes |

| block | pair | unpinned | pinned 0xff | delta |
|---|---|---|---|---|
| TB2 | a1/b1 | 7.4910 | 9.2654 | **+23.69%** |
| TB2 | b2/a2 | 7.4435 | 9.0646 | **+21.78%** |
| TB3 | a1/b1 | 7.8474 | 9.0270 | **+15.03%** |
| TB3 | b2/a2 | 7.0782 | 8.9295 | **+26.16%** |

Group means: unpinned **7.4650** (4 arms), pinned 0xff **9.0716** (4 arms) = **+21.5%**.
Pairs: mean **+21.66%**, sd 4.77, SE 2.39, **t = 9.08 (df 3)**. All four pairs positive, and
the four pinned arms span only 8.93-9.27 t/s against 7.08-7.85 for the unpinned arms - the
distributions do not overlap.

The position check rules out an ordering artefact: in TB1, where the flag is neutral, the
2nd/3rd slots ran **+1.38%** above the 1st/4th; in TB2/TB3 the same slots run **+21.5%**
above. The effect belongs to the flag.

**So the answer to "should the batch pool be pinned?" is yes - but to `0xff`, not `0x3f`.**
Eight threads pinned 1:1 to eight cores beat eight threads left to migrate, and they beat
six threads on the six-core cluster. The shipping flag set gets this wrong by accident: it
passes `-tb 8`, which silently drops the batch pool to "no affinity" rather than "same as
`--cpu-mask`".

`majflt` also rises with pinning (263,294 -> 335,975 for `0x3f`, and ~200,000 -> ~350,000
for `0xff`), so the faster configuration faults more, not less - the opposite of the
`mul_mat_id` story. The win is from removing migration, not from fault behaviour.

### 2.4 Does the flag change hurt decode?

`--cpu-mask-batch` only configures the batch threadpool, which is used when
`ubatch.n_tokens > 1`; single-token decode uses the main pool, whose parameters are
unchanged. The realistic risk is indirect: prefill now runs on cpu6/7 as well, so a session
that prefills and then decodes could arrive at the decode phase hotter. That is exactly what
the `pd` mode measures - one request, 256-token prefill then 64 decode tokens.

Block TB4, `A = tb=8` unpinned, `B = tb=8 --cpu-mask-batch 0xff --cpu-strict-batch 1`:

| arm | spec | prefill t/s | decode t/s | decode 1..16 | decode 17..64 | majflt | stable |
|---|---|---|---|---|---|---|---|
| TB4a1 | unpinned | 7.4209 | 4.2990 | 3.4589 | 4.7861 | 372,291 | yes |
| TB4b1 | pinned 0xff | 8.9808 | 4.4147 | 3.7932 | 4.7749 | 381,098 | yes |
| TB4b2 | pinned 0xff | 8.7481 | 4.4622 | 3.7178 | 4.8905 | 467,600 | yes |
| TB4a2 | unpinned | 7.6118 | 4.2812 | 3.7025 | 4.6181 | 235,416 | yes |

| metric | unpinned | pinned 0xff | delta |
|---|---|---|---|
| prefill t/s (group mean) | 7.5164 | 8.8645 | **+17.94%** |
| decode t/s (group mean) | 4.2901 | 4.4385 | **+3.46%** |
| decode 1..16 t/s | 3.5807 | 3.7555 | +4.88% |
| decode 17..64 t/s | 4.7021 | 4.8327 | +2.78% |

Pairs: prefill **+21.02% / +14.93%**, decode **+2.69% / +4.23%**. All four pairs positive.

**Decode does not regress; if anything it improves slightly (+3.5%).** That is inside the
5-8% arm-to-arm spread, so treat it as neutral-to-mildly-positive rather than a second
effect, but it is enough to say the change is safe end-to-end. The early-decode ratio also
tightens (0.723/0.802 -> 0.794/0.760), consistent with a shorter, better-pinned expert storm
leaving the page cache in a slightly better state.

**Recommendation: add `--cpu-mask-batch 0xff --cpu-strict-batch 1` to the shipping flags.**
It is flag-only, needs no engine patch, does not touch the decode pool, and buys ~18-22%
prefill. This is the largest single result in Stage 3.

---

## 3. Prefill eviction shielding (open item §4.2)

### 3.1 The post-prefill decode dip is real

`s3_arm.sh pd` issues a 256-token prefill and 64 decode tokens in **one** request with
`stream: true` and `timings_per_token: true`, so the decode rate is recoverable per token.

Baseline arms (shield off), decode split into the first 16 tokens and tokens 17-64:

| arm | decode 1..16 t/s | decode 17..64 t/s | ratio |
|---|---|---|---|
| SH1a1 | 3.6309 | 4.7836 | 0.759 |
| SH1a2 | 3.8084 | 4.8578 | 0.784 |
| SH2a1 | 3.9487 | 4.8691 | 0.811 |
| SH2a2 | 3.6458 | 4.9034 | 0.744 |

The first 16 decode tokens after a 256-token prefill run at **~77% of the steady-state
rate** (mean ratio 0.775, range 0.744-0.811). The signature is also visible in the fault
rate: using the request's own `prompt_ms` to place the prefill/decode boundary, the decode
phase carries **11,000-14,700 major faults/s** against **3,300-3,800/s** during prefill -
a 3-4x step at the transition. That is the eviction signature the handoff predicted.

### 3.2 Implementation

`BONSAI_SHIELD` in `src/llama-context.cpp`, read in `llama_context::decode` at the ubatch
level (default 0 = off):

| value | behaviour |
|---|---|
| 0 / unset | off |
| 1 | Option A: on the first decode ubatch after a prefill, `POSIX_MADV_WILLNEED` the dense tensors once |
| 2 | Option B: `POSIX_MADV_RANDOM` the expert tensors during prefill, back to `POSIX_MADV_NORMAL` at the first decode ubatch |
| 3 | both |

"Expert" is any tensor whose name contains `_exps.`; "dense" is everything else **except**
`per_layer_token_embd.weight`. That exclusion is essential: it is a single 26.8 GiB tensor
in the same `tensors_by_name` list, and a `WILLNEED` on it would readahead 26.8 GiB per
transition. Addresses are page-aligned down before `posix_madvise`.

Verified execution (server run with `-lv 4`, because libllama's INFO maps to
`LOG_LEVEL_TRACE` (4) while the default threshold is `LOG_LEVEL_INFO` (3), so the report line
is invisible at default verbosity):

```
I bonsai_shield_apply: BONSAI_SHIELD applied to 144 expert tensors (32400.0 MiB)
I bonsai_shield_apply: BONSAI_SHIELD applied to 1079 dense tensors (2907.8 MiB)
```

144 expert tensors at 32,400.0 MiB and 1,079 dense tensors at 2,907.8 MiB match the
inventory in §1.1 exactly (884 other dense + 192 converted `hc_*` + 3 `output_hc_*` = 1,079;
2,257.6 + 637.4 + 12.5 = 2,907.5 MiB). The 26.8 GiB table was correctly excluded. The expert
line is emitted at prefill and the dense line at the first decode token, i.e. the transition
fires in the intended order.

### 3.3 Correctness gate: greedy token identity - PASS

`--temp 0`, `seed 1`, a 256-token prefill and 32 greedy decode tokens in one request, so the
prefill -> decode transition is exercised:

| shield | bytes | md5 |
|---|---|---|
| 0 | 170 | `f1ac9c8a77541e447f1f4810a8cadc73` |
| 1 | 170 | `f1ac9c8a77541e447f1f4810a8cadc73` |
| 2 | 170 | `f1ac9c8a77541e447f1f4810a8cadc73` |

All three are byte-identical. Shielding only issues `posix_madvise`, so the arithmetic is
untouched and identity is the right gate here.

### 3.4 Option A (`WILLNEED` dense): mechanism works, speed does not follow

Mirrored A/B/B/A, `A = shield 0`, `B = shield 1`, 256-token prefill + 64 decode in one call:

| arm | shield | prefill t/s | decode t/s | minflt | majflt | measured window | stable |
|---|---|---|---|---|---|---|---|
| SH1a1 | 0 | 7.5262 | 4.3626 | 636,265 | 338,305 | 3532800/3840000 -> 3532800/3840000 | yes |
| SH1b1 | 1 | 7.5906 | 4.3763 | 650,052 | 231,619 | 3532800/3840000 -> 3532800/3840000 | yes |
| SH1b2 | 1 | 7.7463 | 4.4895 | 638,534 | 177,228 | 3532800/3840000 -> 3532800/3840000 | yes |
| SH1a2 | 0 | 7.3036 | 4.4737 | 618,527 | 310,131 | 3532800/3840000 -> 3532800/3840000 | yes |

| pair | shield 0 | shield 1 | delta |
|---|---|---|---|
| a1/b1 prefill | 7.5262 | 7.5906 | +0.86% |
| a1/b1 decode | 4.3626 | 4.3763 | +0.31% |
| b2/a2 prefill | 7.3036 | 7.7463 | +6.06% |
| b2/a2 decode | 4.4737 | 4.4895 | +0.35% |

Group means:

| metric | shield 0 | shield 1 | delta |
|---|---|---|---|
| prefill t/s | 7.4149 | 7.6684 | +3.42% |
| decode t/s | 4.4181 | 4.4329 | **+0.33%** |
| total majflt | 324,218 | 204,424 | **-36.9%** |
| decode 1..16 t/s | 3.7196 | 3.7868 | +1.8% |
| decode 17..64 t/s | 4.8207 | 4.8069 | -0.3% |

Splitting the faults at the `prompt_ms` boundary makes the mechanism unmistakable:

| arm | shield | majflt during prefill | majflt during decode | decode majflt/s |
|---|---|---|---|---|
| SH1a1 | 0 | 118,109 | 220,020 | 14,668 |
| SH1a2 | 0 | 141,936 | 168,160 | 11,211 |
| SH1b1 | 1 | 185,173 | 46,446 | 3,096 |
| SH1b2 | 1 | 105,802 | 71,413 | 4,761 |

**Decode-phase major faults fall by 70-79%.** The hint does exactly what it was designed to
do: the dense pages evicted by the expert storm are streamed back in one burst instead of
being demand-faulted one at a time.

It just does not make the model faster. Decode is +0.33%, the early-decode window is +1.8%,
and the steady-state window is -0.3% - all inside the noise floor, with the pairs agreeing
in sign but at a magnitude far below the 5-8% arm-to-arm spread.

**Interpretation: the dip is not fault-count-bound, it is I/O-volume-bound.** The shield
changes how the same bytes are *accounted* (one bulk readahead versus many demand faults);
it does not reduce the bytes that must come off storage. The evicted working set still has
to be re-read either way. That also explains why the Stage 2 `mul_mat_id` flat change, which
did convert a fault reduction into a speed win (719 -> 318 majflt/token, +4.8%), was a
different case: there the fault reduction came with less work per token, not just a
different accounting of the same work.

### 3.5 Option B (`MADV_RANDOM` experts): a dead end, measured

Mirrored A/B/B/A, `A = shield 0`, `B = shield 2`:

| arm | shield | prefill t/s | decode t/s | minflt | majflt | measured window | stable |
|---|---|---|---|---|---|---|---|
| SH2a1 | 0 | 7.7171 | 4.5291 | 618,228 | 220,769 | 3532800/3840000 -> 3532800/3840000 | yes |
| SH2b1 | 2 | 4.9649 | 4.1198 | 632,953 | **2,174,127** | 3532800/3840000 -> 3532800/3840000 | yes |
| SH2b2 | 2 | 5.0164 | 4.0461 | 632,298 | **2,129,453** | 3532800/3840000 -> 3532800/3840000 | yes |
| SH2a2 | 0 | 7.1095 | 4.4436 | 638,364 | 353,944 | 3532800/3840000 -> 3532800/3840000 | yes |

| pair | shield 0 | shield 2 | delta |
|---|---|---|---|
| a1/b1 prefill | 7.7171 | 4.9649 | **-35.66%** |
| a1/b1 decode | 4.5291 | 4.1198 | **-9.04%** |
| b2/a2 prefill | 7.1095 | 5.0164 | **-29.44%** |
| b2/a2 decode | 4.4436 | 4.0461 | **-8.95%** |

Group means:

| metric | shield 0 | shield 2 | delta |
|---|---|---|---|
| prefill t/s | 7.4133 | 4.9907 | **-32.68%** |
| decode t/s | 4.4863 | 4.0830 | **-8.99%** |
| total majflt | 287,357 | 2,151,790 | **+648.8%** |
| decode 1..16 t/s | 3.7973 | 3.2305 | -14.9% |
| decode 17..64 t/s | 4.8863 | 4.5819 | -6.2% |

The early-decode ratio also gets worse (0.811/0.744 -> 0.692/0.719).

**The Option B hypothesis is falsified.** Turning off fault-around and readahead on the
expert tensors does not stop the prefill from pulling in "unrouted neighbour pages" - it
converts the kernel's amortised bulk reads into ~7.5x as many individual major faults and
costs a third of prefill throughput. The readahead on this access pattern is doing useful
work, not wasted work. Since mode 3 is mode 1 plus mode 2, it is dominated by this harm and
was not worth a device block.

### 3.6 Task 3 conclusion

| option | mechanism | prefill | decode | majflt | verdict |
|---|---|---|---|---|---|
| 1: `WILLNEED` dense at prefill->decode | decode-phase faults -70-79% | +3.4% (n.s.) | +0.33% (n.s.) | -36.9% | mechanism confirmed, **no speedup**; the dip is I/O-volume-bound |
| 2: `MADV_RANDOM` experts during prefill | readahead disabled | **-32.7%** | **-9.0%** | **+649%** | **dead end** |

Neither option ships. `BONSAI_SHIELD` stays in the tree as an env-gated switch, default off,
with token identity proven for all three modes.

**What the dip actually needs.** The signature says the cost is re-reading an evicted
working set, so the lever has to reduce *bytes read after prefill*, not fault events. The
candidates that follow from this measurement:

* Reduce prefill's own expert footprint so it evicts less - e.g. route fewer experts during
  prompt processing (`BONSAI_MOE_TOPK` is already wired and trades quality for prefill
  speed; a prefill-only top-k is a smaller, different experiment).
* Or make the decode working set smaller so less has to come back - which is exactly what
  `varH` does, and is probably why it wins on decode.
* `MADV_DONTNEED` on prefill-only experts (the handoff's other suggestion) is the remaining
  untested variant; it would need the decode-hit table to avoid dropping hot experts.

One indirect lever is now measured: the §2 `--cpu-mask-batch 0xff` change cuts prefill wall
time by ~18% for the same routing work, so the expert storm is compressed into a shorter
window. Whether that helps the following decode is what the `pd` control in §2.4 checks.

---

## 4. Reproducing

* Harness (device, `/data/local/tmp/`): `s3_arm.sh` (one arm: cap gate, warmup, measured
  request, in-window cap reads, fault deltas), `s3_run.sh` (mirrored A/B/B/A driver with
  cooldown), `s3_placement.sh` (thread affinity + prefill placement sampling),
  `s3_identity.sh` (greedy token identity), `s3_shieldlog.sh` (shield report lines at
  `-lv 4`).
* Analysis (host): `MoeDroid/tools/s3_table.py` (arm tables, in-window cap stability,
  same-profile pairs, per-token decode windows from the SSE stream).
* Device library set: `/data/local/tmp/lib-shield`, built from
  `upstream-llama/build-arm64-ship`.
* Models: shipping at
  `/sdcard/Android/data/com.prismml.moedroid/files/models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`,
  varH at `/sdcard/Download/mf-model/varH-00001-of-00002.gguf` (shard 2 alongside it).
* Logs pulled to `.stage3-logs/` (`S3-D1.txt`, `S3-D2.txt`, `S3-TB1.txt`, `S3-TB2.txt`,
  `S3-TB3.txt`, `S3-TB4.txt`, `S3-SH1.txt`, `S3-SH2.txt`, plus per-arm `.sse`, `.faults`,
  `.caps`, `.log`).

### Blocks run in this stage

| block | mode | A | B | purpose |
|---|---|---|---|---|
| D1, D2 | decode 64 tok | shipping | varH | bank varH |
| TB1 | prefill 256 tok | `-tb 8` unpinned | `-tb 6` + `--cpu-mask-batch 0x3f` | does pinning to the 6-core cluster help prefill? |
| TB2, TB3 | prefill 256 tok | `-tb 8` unpinned | `-tb 8` + `--cpu-mask-batch 0xff` | does explicit 8-core pinning help prefill? |
| TB4 | prefill 256 + decode 64 | `-tb 8` unpinned | `-tb 8` + `--cpu-mask-batch 0xff` | does the win survive, and does decode regress? |
| SH1 | prefill 256 + decode 64 | `BONSAI_SHIELD=0` | `BONSAI_SHIELD=1` | Option A: `WILLNEED` dense |
| SH2 | prefill 256 + decode 64 | `BONSAI_SHIELD=0` | `BONSAI_SHIELD=2` | Option B: `MADV_RANDOM` experts |
| id0/id1/id2 | prefill 256 + decode 32 | - | - | greedy token identity per shield mode |
| warm1 | decode 64 tok | varH | - | validate the staged copy |
| pl1, pl2 | 1,542-tok prefill | no batch mask | `--cpu-mask-batch 0x3f --cpu-strict-batch 1` | thread affinity masks + prefill placement |

### Thermal methodology note

The device sits near 41-43 C in the CPU cluster zones while charging at 100% on AC, so a
temperature-threshold gate is not reachable on this bench. The gate therefore requires the
known-good cap profile (`cpu0 = 3532800`, `cpu6 = 3840000`) to hold for three consecutive
10 s polls before the model is loaded, and stability is then judged from the caps read
immediately either side of the measured window plus a 5 s sampler for the whole arm. An
earlier TB1 attempt that used only a 30 s cooldown produced two arms that dropped to
`2400000/2841600` mid-arm; those results were discarded and the block re-run with a 120 s
cooldown, after which all four arms held the profile.
