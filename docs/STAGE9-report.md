# Stage 9 report — overnight execution (Phase 1 on-device shootout + host census + `P3a` Gate 3b)

Date: 2026-10-18/19. Device `91251291018D` (RedMagic 10S Pro, Snapdragon 8 Elite), plugged in,
awake, external cooling fan ON, Game Space (`com.prismml.moedroid/MainActivity`) foregrounded.

Authoritative inputs read this session: `GOAL.md`, `docs/STAGE8-report.md`,
`docs/STAGE9-daytime-prep.md`, `MoeDroid/tools/{s8_shootout.sh,s8_arm_ll.sh,s8_ll_table.py,s4_prof.py}`.
`docs/HANDOFF.md` and `MonsterQwenHandoff.md` were NOT read (stale at Stage 5, per the brief).

The report is written incrementally, one section per completed step.

---

## 0. Environment and directory rules actually in force

| rule | state |
|---|---|
| private `/tmp` per bash call | never written; all scratch under `.stage9/`, all logs under `.stage8-logs/` |
| Stage 8 evidence | `.stage8-logs-archive-stage8/` — **not touched** (renamed archive, 79 GB, 0 bytes copied) |
| `LOGDIR` | `$WS/.stage8-logs` — contains only `driver.log` written by `HOST_CHECK=1` runs (no Stage 8 measurement lives there) |
| long-running work | managed background jobs only; no `nohup … &` (PID-namespace `--die-with-parent` trap) |
| gradle | `GRADLE_USER_HOME=$WS/.gradle-home ./gradlew … --offline` (no `tools/gradle-8.14`) |
| `adb` | `$WS/android-sdk/platform-tools/adb` (not on `PATH` by default) |

---

## 1. Preflight

### 1.1 Host gate — `HOST_CHECK=1 s8_shootout.sh 0`

**0 failures / 27 checks**, including the four structural checks that were added during Stage 9 prep:

```
OK   every packed arm carries BONSAI_PK_REPORT
resolved C2_ENV=[BONSAI_MOE_X4_PREFILL_PACKED=1 BONSAI_PK_REPORT=1]
resolved D1_ENV=[BONSAI_Q8_0_X4=2 BONSAI_PK_REPORT=1]
OK   symbol bonsai_x4_rows4_mode1_packed in ship libggml-cpu.so
OK   analyzer enforces the 0-hit rule
=== host-side prerequisites: 0 failure(s) ===
```

### 1.2 Analyzer self-test — `s8_ll_table.py --selftest`

**SELFTEST PASSED**, 26 assertions, including the three validity rules that decide tonight:

```
PASS  packed arm with hits is 'engaged'
PASS  packed arm with 0 hits is INVALID
PASS  packed arm with no counter is INVALID
PASS  truncated arm is INVALID even when not packed
PASS  BONSAI_Q8_0_X4=1 is n/a, not a cache verdict
PASS  negative: mis-classifying =1 as cache-using flips A6 to INVALID
```

### 1.3 Device gate

The phone arrived **Dozing with the notification shade focused** — not usable as handed over. Corrected
before any measurement (recorded because a sleepy device silently unmounts `/sdcard` and loses the model):

| check | before | after |
|---|---|---|
| `mWakefulness` | `Dozing` | **`Awake`** |
| `mScreenState` | `OFF` | **`ON`** |
| keyguard | showing | **dismissed** |
| `mCurrentFocus` | `NotificationShade` | **`com.prismml.moedroid/MainActivity`** |

`svc power stayon true` was set (reverted at the end of the window, §5.4) and `screen_off_timeout`
raised, because a `mScreenState=OFF` device drops to `Dozing` between arms and aborts the cap gate.

Cap profile at the gate (`scaling_max_freq`, which is what `preflight_caps()` reads):

```
cpu0 scaling_max=3532800   (want 3532800)
cpu6 scaling_max=3840000   (want 3840000)
cpu0..5 cpuinfo_max=3532800 ; cpu6/7 cpuinfo_max=4473600
```

**Stray-engine check.** Exactly one engine process existed:

```
PID 5073  PPID 28572  libllamaserver.so
cmdline: -m /storage/emulated/0/Android/data/com.prismml.moedroid/files/models/
         Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --host 127.0.0.1 --port 38905 …
```

It is the **app's own** engine (its parent is `com.prismml.moedroid`), on the app's private port 38905,
not a leftover harness engine. Two facts make it the expected state rather than a stray:

* it cannot be killed by us and never could be — `adb shell kill -9 5073` returns
  `Operation not permitted` (uid 2000 `shell` vs the app's uid), so `kill_engine()` in
  `s8_arm_ll.sh` has always been unable to touch it;
* the harness port **8299 has no listener** (`grep ':206B' /proc/net/tcp` → 0 lines), and no
  `libllamaserver` is parented by `adbd`/`sh`.

`/proc/5073/maps` is not readable from the shell uid (`Permission denied`, no `su` on the device), so the
"holds the model" half of the check is established indirectly: its cmdline carries the varH shard, it is
the process tree's only engine, and any harness engine from a previous run would have been a shell child
and killable. Memory headroom with it resident: `MemAvailable 14.86 GB`, `Cached 11.67 GB`, `MemTotal 23.06 GB`.

**Conclusion: device green, previous-run engines absent, safe to measure.**

---

## 2. Concurrent host attribution census (Step 2)

The two Phase-2 preconditions, both closed with primary evidence rather than a derived number.
The phone was running §3.1 while this was produced; nothing here touches the device.

### 2.1 The exact `174 GET_ROWS` tensor-name census

**Method: the graph was built and every node name read out of the built graph, not inferred from
source.** A 3-line env-gated dump (`BONSAI_GRAPH_NAMES=1`) was added to
`llama_context::process_ubatch` (`src/llama-context.cpp:1621`) that prints the op, name, shape and
`src[0]`/`src[1]` names of every `GET_ROWS` node in the graph the engine actually built. Host x86,
real varH weights, `-p "hi there" -n 3 -t 16 -c 512 --no-warmup`.

Five graphs were built (n_tokens = 2, 42, 8, 4, 1). **Every graph has 6067 nodes and 175 `GET_ROWS`
nodes** — the count is independent of the token count, which is what makes it a per-token number.

The engine's own profiler on the same run reconciles the 174:

```
after 4 graphs  GET_ROWS  696 calls
after 5 graphs  GET_ROWS  870 calls   -> 174 for graph 5 (n_tokens=1)
after 6 graphs  GET_ROWS 1044 calls   -> 174 for graph 6 (n_tokens=1)
```

**175 built, 174 executed, and the missing node is named.** It is the `inp_mixed_tokens` embedding
gather, and it is built with `GGML_TENSOR_FLAG_COMPUTE = 0`, so the compute loop skips it:

```
BGROWS gid=4 i=3 compute=0 dst=[node_3] s0=[token_embd.weight] s1=[inp_mixed_tokens]
```

So the census of the **174 executed calls** is:

| # | tensor read (`src[0]`) | index (`src[1]`) | dst shape | calls/token | source |
|---|---|---|---|---|---|
| 48 | `ffn_moe_probs-<il>` | `ffn_moe_topk-<il>` | [1, 10, 1, 1] | 1 x 48 layers | `llama-graph.cpp:2239` `ggml_get_rows(ctx0, probs, selected_experts)` |
| 36 | `cache_r_l<n>` (reshaped) | `rs_s_copy` | [30720, 1, 1, 1] | 1 x 36 layers | `llama-memory-hybrid-idx.cpp` state read |
| 36 | `cache_s_l<n>` (reshaped) | `rs_s_copy` | [786432, 1, 1, 1] | 1 x 36 layers | `llama-memory-hybrid-idx.cpp` state read |
| 24 | `cache_idx_k_l<n>` (view) | `leaf_108 (reshaped)` | [128, 4, 1, 1] / [128, 64, 1, 1] | 2 x 12 layers | `qwen4exp.cpp:768` `gather_key_gate`, `:790` `gather_pooled` |
| 12 | `leaf_9` | `indexer_top_k-<il>` | [4, 64, 1, 1] | 1 x 12 layers | `qwen4exp.cpp:814` `sel_idx` (pool-idx gather) |
| 12 | `indexer_score-<il>` | `indexer_top_k-<il>` | [1, 64, 1, 1] | 1 x 12 layers | `qwen4exp.cpp:843` `top_score` |
| 3 | `attn_output-N`, `l_last-46`, `hc_inject-N` | `out_ids` | [2560,1,1,1] / [10240,1,1,1] / [4,1,1,1] | 3 once | `qwen4exp.cpp:474,475,478` last-layer crops |
| 1 | `token_embd.weight` | `inp_tokens` | [2560, 1, 1, 1] | 1 once | `qwen4exp.cpp:555` |
| 1 | `per_layer_token_embd.weight` | `leaf_1` | [160, 16, 1, 1] | 1 once | `qwen4exp.cpp:1362` PLE n-gram gather |
| 1 | `cache_ple_r_l1` (reshaped) | `rs_s_copy` | [92160, 1, 1, 1] | 1 once | PLE cache read |
| | | | | **174** | |

### 2.2 This corrects the prep document's attribution, and the correction changes the conclusion

The Stage 9 prep (§Task 3.2) attributed `174` as **"96 QSA + 48 MoE + ~30 misc"** by multiplying
per-layer sites by 48 layers. The built graph says otherwise:

| group | prep estimate | **measured** | why the estimate was wrong |
|---|---|---|---|
| QSA / indexer | 96 | **48** | the indexer exists on **12 of 48 layers**, not 48. `cache_idx_k` gathers appear on `l3, l7, l11 … l47` only — the same 12 layers that carry `ARGSORT`. 2 pool reads x 12 + `sel_idx` 12 + `top_score` 12 = 48 |
| MoE group select | 48 | **48** | exactly right: `ffn_moe_probs` is 1/layer x 48 |
| **hybrid recurrent cache** | **0 (not mentioned)** | **72** | `cache_r` + `cache_s` gathers, 36 layers each. These are the **largest** group and were absent from the prep's table |
| misc | ~30 | **6** | 3 last-layer crops + 2 token embeddings + 1 PLE cache |

**The actionable consequence is different from the one the prep drew.** The prep reasoned about
fusing the QSA selector (48 calls) and quoted "5.791 ms + 0.944 ms + 2.296 ms = 9.03 ms/token" for a
fusion project. But the executed census says the bytes are dominated by the recurrent-state reads,
not by the selector:

| read | bytes/token | share |
|---|---|---|
| `cache_s` x 36 | 36 x 786432 x 4 = **113.2 MB** | 93.9% |
| `cache_r` x 36 | 36 x 30720 x 4 = 4.4 MB | 3.7% |
| `cache_idx_k` x 24 | ~0.6 MB | 0.5% |
| `ffn_moe_probs` x 48 + crops + embeddings | ~2.3 MB | 1.9% |
| **total** | **~120.5 MB/token** | |

At the measured **5.791 ms/token** that is **20.8 GB/s** — a gather-bandwidth number, not a
call-count number. So the 5.79 ms is **`cache_s` state reads**, and fusing 48 selector `GET_ROWS` nodes
into one op cannot touch 94% of it. The honest scoping statement replaces the prep's:
the selector fusion is worth at most ~0.4 ms/token (its share of bytes), while the 113 MB/token of
recurrent state is algorithm-required (it is the linear-attention state, read once per layer per
token) and is a **state-layout** problem, not an op-fusion problem.

That is a correction to the census, not a new proposal: no graph or kernel change is made tonight on
the strength of it.

### 2.3 `ik` `mul_mat_id` chunk sizing, verified in source

`chunks_per_expert` is not a guess — it is one expression, used by both of ik's expert passes:

```c
// ik_llama.cpp/ggml/src/ggml.c:18709   in ggml_compute_forward_mul_mat_id (defined :18590)
const int chunks_per_expert = MAX(1, MIN(nth, (int)(ne01 / 32)));

// ik_llama.cpp/ggml/src/ggml.c:19039   the fused/iqk pass, same shape with a half-width guard
const int64_t nr0_base = src0_2 ? ne01 : ne01/2;
const int chunks_per_expert_ug = MAX(1, MIN(nth, (int)(nr0_base / 32)));
```

`GGML_EXPERT_CHUNKING` is **ON in the device build**, not dead code:

```
ggml/CMakeLists.txt:98   option(GGML_EXPERT_CHUNKING "ggml: enable expert chunking" ON)
build-arm64-ik/CMakeCache.txt:378   GGML_EXPERT_CHUNKING:BOOL=ON
build-arm64-ik/llama-config.cmake:100   … GGML_USE_IQK_MULMAT;GGML_IQK_FLASH_ATTENTION;…;GGML_EXPERT_CHUNKING …
```

Evaluated for this model (`qwen4exp.expert_feed_forward_length = 640`,
`expert_used_count = 10`, measured from the GGUF header), at `-t 6`:

| expert matrix | `ne01` | `ne01/32` | `chunks_per_expert` | rows per chunk |
|---|---|---|---|---|
| `ffn_up_exps` / `ffn_gate_exps` (2560 x 640 x 512) | 640 | 20 | **6** | ~107 |
| `ffn_down_exps` (640 x 2560 x 512) | 2560 | 80 | **6** | ~427 |

So the `MIN(nth, ...)` term binds: **exactly one chunk per thread per expert**, and the `/32` floor
guarantees no chunk is smaller than 32 rows. A decode token with 10 experts is **60 chunk tasks over
6 threads = 10 tasks per thread**.

Our fork's unit for the same work is `ggml_compute_forward_mul_mat_id_one_chunk` with
`blck_0 = 16, blck_1 = 16` (`ggml-cpu.c:1847`), i.e. **16-row blocks**: 640 rows / 16 = 40 blocks per
expert x 10 experts = **400 chunk tasks**, re-dispatched through `atomic_fetch_add` on a shared
counter. The comparison is therefore not "chunked vs unchunked" — it is **107-row static chunks
(one per thread, no atomic) vs 16-row dynamic blocks (400 atomic hand-offs)**.

### 2.4 A host-build regression found and fixed in passing

The Stage 9 PK cache hook was called **outside** its own `#if defined(__aarch64__) && defined(__ARM_NEON)`
guard (`ggml-cpu.c:4046-4048`), so `bonsai_pk_inval_mode/bonsai_pk_enabled/bonsai_pk_maybe_report`
were undeclared on x86 and **the host build no longer compiled**:

```
ggml-cpu.c:4046:9: error: implicit declaration of function 'bonsai_pk_inval_mode'
ggml-cpu.c:4046:38: error: implicit declaration of function 'bonsai_pk_invalidate'
ggml-cpu.c:4048:28: error: implicit declaration of function 'bonsai_pk_enabled'
ggml-cpu.c:4048:49: error: implicit declaration of function 'bonsai_pk_maybe_report'
```

Fixed by wrapping the two hook lines in the same guard the machinery uses. On aarch64 the preprocessor
output is **identical** (the condition is already true there), so no device binary behaviour changes;
the only effect is that `build-x86` compiles again, which is what the `GOAL.md` host PPL gate needs.

---

## 3. On-device Phase 1: identity triage, mirrored 2-pass shootout, same-engine profiler

**Everything in §3 is pre-registered: the arm list, the validity rules and the ship bar were fixed
before any timing number existed.** That is deliberate — a bar written after the numbers is a
rationalisation, and Stage 9's own prep document records five assertion-vs-code drifts in one day.

### 3.0 Pre-registered protocol

| item | value |
|---|---|
| arms timed | `A0`, `AP2`, `B0`, `B1`, `B2`, `C1`, `C2`, `D1` — **`A5` excluded** (`MOE_X4=3` rejected twice, Stage 7 and Stage 8; re-timing it buys nothing) |
| passes | decode pass 1, decode pass 2 (reverse order), prefill pass 1, prefill pass 2 (reverse) |
| per arm | 12 timed warmups, `WARM_COOL_S=90` in-process cooldown, `POSTCOOL` cap gate, 6 measured requests |
| validity | `n == meas` (6/6), `S8THR throttled=0` on decode, `@gN pk_hits > 0` + `class=cache` on packed arms |
| prefill caveat | prefill arms legitimately read 6/6 while the thermal trip lands inside the window; their `throttled` column is quoted beside every prefill median and prefill deltas are compared only among equal `K` |
| analyzer | `MoeDroid/tools/s8_ll_table.py` (the LIVE analyzer; `s7_ll_table.py` was not used) |

**Pre-declared ship bar, `D1` (`BONSAI_Q8_0_X4=2`)** — ships iff all four hold:
1. all relevant arms valid (`n == meas`, `S8THR throttled=0` on decode);
2. positive vs `A0` in **both** mirrored decode passes, and neutral-or-better on prefill;
3. valid `@gN` `pk_hits > 0`;
4. same-engine `BONSAI_PROF` shows `MUL_MAT_ID` ms/token moving in the same (negative) direction under a
   passing clock canary (`GET_ROWS` and `GATED_DELTA_NET` within 2.5%).

**Pre-declared carry for `D1`**: because `MOE_X4=1` already shares the activation across 4 weight rows,
the honest prior is **+0.5% to +2.5% decode**, not the +7-11% a naive 2-load saving suggests. A ~+1.5%
result on both passes is a genuine win, not a disappointment.

**Pre-declared ship bar, `C2` (`BONSAI_MOE_X4_PREFILL_PACKED=1`)** — ships iff valid (`n == meas`),
positive vs `A0` in **both** mirrored prefill passes, neutral on decode, and valid `@gN` `pk_hits > 0`.

**Never done**: quote a t/s from an arm the analyzer marks INVALID, override the analyzer's validity
verdict, or compare arms with different cap profiles.

### 3.1 Identity gates, with per-arm triage

Rule applied: an arm that fails token identity is **dropped from the timing matrix**; only an unstable
`A0` aborts the window.

**Result: every arm that entered the gate passed it, on both prompts. No arm was dropped.**

| arm | env | short prompt md5 | 262-token prefill md5 | gate verdict |
|---|---|---|---|---|
| `A0` | — | `8ad776c2…` | `514b317d…` | reference |
| `A0b` | — | `8ad776c2…` | `514b317d…` | **A0 stable** (repeat identity) |
| `A5` | `MOE_X4=3` | `8ad776c2…` | (not run) | identical |
| `AP2` | `MOE_X4_PREFILL=0` | `8ad776c2…` | `514b317d…` | identical |
| `C1` | `MOE_X4_PREFILL=2` | `8ad776c2…` | `514b317d…` | identical |
| `C2` | `MOE_X4_PREFILL_PACKED=1` | `8ad776c2…` | `514b317d…` | identical |
| **`D1`** | **`BONSAI_Q8_0_X4=2`** | **`8ad776c2…`** | **`514b317d…`** | **identical** |

Analyzer output, verbatim:

```
IDENTITY PASS (A0 repeatability (short))
IDENTITY PASS (A0 repeatability (262-token prefill))
IDENTITY PASS (C1 vs shipped) / (C1 vs shipped, 262-token prefill)
IDENTITY PASS (C2 (packed tile) vs shipped) / (C2 vs shipped, 262-token prefill)
IDENTITY PASS (D1 (packed 4-row decode) vs shipped) / (D1 vs shipped, 262-token prefill)
IDENTITY2 PASS (D1 vs stock on a SECOND prompt, same session)
```

The 262-token prefill arms all report `evaluated=262`, so the prefill identity is a real full-length
prompt, not a truncated one. The `IDENTITY2` pair is the cross-prompt freshness check (audit item B):
`id-A0-p2` and `id-d1-p2` produce the **same** second-prompt token array
(`bba35711cb42cd3b51f4d527021c2fc4`, 64 ids), in the same live session that had already decoded prompt 1.

**The cache counters are the second half of the gate, and they say `D1` engaged while `C2` barely does:**

| arm | scope | `pk_hits` | `pk_misses` | hit rate | verdict |
|---|---|---|---|---|---|
| `id-D1` | decode | 8,525,793 | 131,607 | **98.5%** | engaged |
| `id-d1-p2` | decode, 2 prompts | 16,192,638 | 250,182 | **98.5%** | engaged |
| `id-pf-D1` | prefill arm (its n=1 steps) | 2,686,351 | 41,129 | **98.5%** | engaged |
| `id-C2` | 64-token decode window | 716,874 | 835,510 | **46.2%** | engaged, weak |
| `id-pf-C2` | 262-token prefill | 1,518,210 | 29,972,606 | **4.8%** | engaged, effectively not caching |

**`D1`'s 98.5% is better than the qemu prediction (93.8%)** — on the real model there is exactly one
activation row in flight per token, so the two-slot cache almost never evicts.

**`C2`'s 4.8% on prefill is a finding that matters for the ship decision**, and it is measured, not
predicted: with 262 token rows in flight the two-slot `(row pointer, n)` cache thrashes on every access,
so the packed tile is re-packed essentially as often as the unpacked one reads it. Whatever `C2` does
to prefill speed tonight, it will not be doing it through cache reuse.

### 3.2 Mirrored timing matrix — run in progress

Arms actually launched: `ARMS="A0 AP2 B0 B1 B2 C1 C2 D1"` (8 arms x 4 passes = 32 timed arms),
then `phase1b_confirm` (the `prof-ours-A0` / `prof-ours-D1` pair) **last**, as the prep scheduled it.

Two problems were found and fixed *before* the matrix could produce a publishable D1 number:

1. **`build-arm64-prof` predated the D1 kernel entirely** (timestamp 04:04, i.e. before the Stage 9
   work). `phase1b_confirm` runs `prof-ours-D1` with `BONSAI_Q8_0_X4=2` on that library, so the flag
   would have been **silently ignored**, the profiler pair would have compared A0 against A0, the
   canary would have passed trivially, and the `MUL_MAT_ID` delta would have been exactly 0.0 — a
   **vacuous confirmation of part three of the ship bar**. The library was rebuilt from the current
   source and re-verified:

   ```
   build-arm64-prof/bin/libggml-cpu.so  2026-10-18 22:16
     bonsai_x4_rows4_mode1_packed          present
     bonsai_vec_dot_q2_0_q8_0_x4_packed4   present
     BONSAI_PK why=                        present
     CPU OP PROFILE                        present
   ```

   This is the same failure class as the prep's `ne11`/`ne12` guard bug: the flag was set, the arm
   looked healthy, and nothing it claimed to measure would have been measured.

2. **The shipped `libggml-cpu.so` was verified to contain the per-node cache fix** before any packed
   arm was timed (`BONSAI_PK_INVALIDATE`, `BONSAI_PK_FIRST`, `BONSAI_PK_EVERY` all present), so a
   stale-pack build cannot be what produces tonight's numbers.

Live status is appended to `.stage9/progress.log`; per-arm raw output is in `.stage8-logs/`.


### 3.3 Pre-ship check: the engine patch is behind the tree (recorded, not yet fixed)

`MonsterQwenDroid/patches/llama.cpp-moedroid.patch` is dated `2026-10-18 02:37` and contains
**none** of the Stage 9 code:

| symbol | in the patch | in the tree |
|---|---|---|
| `bonsai_x4_rows4_mode1_packed` (`D1` kernel) | **0 occurrences** | `iqk_x4.c` |
| `bonsai_pk_cached` (packed-activation cache) | **0 occurrences** | `ggml-cpu.c`, 11 occurrences |
| `BONSAI_PK` (cache fix + reporter) | **0 occurrences** | `ggml-cpu.c` |

So `GOAL.md` constraint 12 ("keep the kernel patch honest ... must reverse-apply cleanly against
`1b70350`") is **currently not satisfiable and not satisfied**: the patch describes 18 files, two of
which (`ggml-cpu.c`, `src/llama-context.cpp`) have since moved on. This is only repaired **if a ship
decision is made** (§4), because regenerating it is the same edit that ships a default. It is recorded
here so the gap is visible either way rather than discovered later.


### 3.4 The first timing launch was aborted: `ARMS` was doubled by the harness

**What happened.** The first launch set `ARMS="A0 AP2 B0 B1 B2 C1 C2 D1"` explicitly. But
`s8_shootout.sh` *appends* the candidate arms whenever their env is set:

```bash
ARMS=${ARMS:-"A0 A5 AP2 B0 B1 B2"}
[ -n "$C1_ENV" ] && ARMS="$ARMS C1"
[ -n "$C2_ENV" ] && ARMS="$ARMS C2"
[ -n "$D1_ENV" ] && ARMS="$ARMS D1"
```

so the effective list was **`A0 AP2 B0 B1 B2 C1 C2 D1 C1 C2 D1` — 11 arms per pass, 44 total instead of
32**. The driver log records it verbatim:

```
[22:41:59] === PHASE 3: timing, arms = A0 AP2 B0 B1 B2 C1 C2 D1 C1 C2 D1 ===
```

Three consequences, the third of which is why the run was killed rather than kept:

1. **Each pass ran `C1`, `C2`, `D1` twice**, and `run_arm` tees with `>` (truncate), so the **second
   run overwrote the first run's per-arm log**. Only the last run of each tag survives in
   `.stage8-logs/`.
2. **The seam between the 8th and 9th arm had no 120 s inter-arm cooldown** (`timed_pass` skips the
   cooldown only after the *last* config, and the duplicate made position 8 non-last).
3. **The position design broke asymmetrically.** The analyzer takes one file per tag per pass, which
   would have compared `D1` at positions `8/11` (pass 1) against `A0` at `1/11`, and `D1` at `4`
   (pass 2, reversed) against `A0` at `11` — a position confound that the mirrored design exists to
   cancel. A ±7% arm is exactly where a position bias could manufacture or destroy the result.

**Decision: kill at 23:59, keep nothing from it, relaunch with `ARMS="A0 AP2 B0 B1 B2"`** so the
script's own append produces the intended 8-arm list. Cost: ~1 h 25 min of device time. The aborted
pass-1 logs are preserved under `.stage8-logs/aborted-arms1/` rather than deleted, and **no number
from them is used anywhere in this report**; the figures below are from the corrected run only.

The aborted pass does have one informational value, recorded here as such: it is a second, independent
sample of the same eight arms a few minutes apart, and it shows the within-pass drift is real
(`A0 6.808`, then `AP2 6.510`, `C1 6.609→6.531`, `C2 6.497`, `D1 6.317`), i.e. later positions in a
pass read lower. That is the noise the mirrored passes are designed to cancel, and it is the reason
the ship bar is "positive in **both** passes", never one.


### 3.5 The corrected matrix: every arm valid, and both candidates lose

32 timed arms, 8 arms x 4 passes, run 00:03 -> 05:01 on 2026-10-19. **Every arm is valid** in the
analyzer's own table: `ok (6/6)`, **`throttled=0/6` on all 16 decode arms**, and both packed arms
engaged (`D1` 98.5%, `C2` 37.6% decode / 5.5% prefill). No arm is truncated or zero-hit, so every
number below is quotable.

**Decode, median t/s per pass, and the delta against that pass's own `A0`:**

| arm | pass 1 | pass 2 | Δ pass 1 | Δ pass 2 | verdict |
|---|---|---|---|---|---|
| `A0` (shipped) | 6.6878 | 6.2958 | — | — | reference |
| `AP2` (tile off) | 6.6557 | 6.4390 | −0.48% | +2.27% | neutral, as designed |
| `B0` (ik, unpinned) | 9.1348 | 9.1325 | +36.59% | +45.06% | reference engine |
| `B1` (ik + pinning) | 9.2103 | 9.0851 | +37.72% | +44.30% | reference engine |
| `B2` (ik + pinning, no fused MoE) | 9.2702 | 9.2013 | +38.61% | +46.15% | reference engine |
| `C1` (`MOE_X4_PREFILL=2`) | 6.0162 | 6.5027 | −10.04% | +3.29% | not decided tonight |
| `C2` (`PREFILL_PACKED=1`) | 6.5610 | 6.3700 | −1.90% | +1.18% | neutral on decode |
| **`D1` (`Q8_0_X4=2`)** | **6.2871** | **6.2642** | **−5.99%** | **−0.50%** | **NEGATIVE IN BOTH** |

**Prefill, same shape:**

| arm | pass 1 | pass 2 | Δ pass 1 | Δ pass 2 | verdict |
|---|---|---|---|---|---|
| `A0` | 8.5978 | 8.7594 | — | — | reference |
| `AP2` | 8.4602 | 8.6505 | −1.60% | −1.24% | the shipped tile is confirmed (+1.63% / +1.26% for A0 over AP2) |
| `B0` | 11.5805 | 11.7705 | +34.69% | +34.38% | reference |
| `B1` | 11.6981 | 11.9423 | +36.06% | +36.34% | reference |
| `B2` | 10.9500 | 11.2177 | +27.36% | +28.06% | reference |
| `C1` | 8.6794 | 9.1121 | +0.95% | +4.03% | not decided tonight |
| **`C2`** | **8.2947** | **8.3218** | **−3.53%** | **−5.00%** | **NEGATIVE IN BOTH** |
| `D1` | 8.5873 | 8.4679 | −0.12% | −3.33% | negative (must be neutral-or-better) |

### 3.6 Ship decision from parts 1-3 of the bar

| candidate | bar | measured | result |
|---|---|---|---|
| **`D1` `BONSAI_Q8_0_X4=2`** | positive vs `A0` in **both** mirrored decode passes | **−5.99% / −0.50%** | **FAIL** |
| | `@gN pk_hits > 0`, engaged | 138,006,437 hits / 98.5% @g1154 | PASS |
| | valid arms (`n == meas`, `throttled=0`) | 6/6, 0/6 on every decode arm | PASS |
| | same-engine `MUL_MAT_ID` negative under passing canary | see §3.7 | not established |
| **`C2` `BONSAI_MOE_X4_PREFILL_PACKED=1`** | positive vs `A0` in **both** mirrored prefill passes | **−3.53% / −5.00%** | **FAIL** |
| | neutral on decode | −1.90% / +1.18% | PASS (neutral) |
| | `@gN pk_hits > 0` | 5.9M / 37.6% decode, 10.7M / **5.5%** prefill | PASS (engaged) |

**Neither candidate ships. Both fail on the pre-declared, decisive clause — the one that requires the
gain to appear in both mirrored passes — and both fail it in the same direction, which the position
design cannot manufacture.**

**How much of `D1`'s −5.99% is real, stated honestly.** `A0` itself moved **−5.9%** between pass 1 and
pass 2 (6.6878 -> 6.2958) with no flag changed, so a single-pass −5.99% is exactly the size of the
night's own drift. The two facts that survive that noise are: (a) `D1` was **not faster in either
pass**, and (b) its pass-2 result is **−0.50%**, i.e. at the noise floor. The defensible statement is
therefore **"`D1` is neutral-to-negative on decode, with no demonstrable gain"**, not "`D1` costs 6%".
Its arm-mean delta is −3.33%, and quoting that as a point estimate without the pass pair would be the
error `GOAL.md` constraint 2 forbids.

**`C2`'s loss has a mechanism, and the counters predicted it.** Its packed cache hit rate is **5.5%**
on prefill (§3.1): with 262 token rows in flight, a two-slot `(row pointer, n)` cache misses almost
every time, so the kernel pays the pack cost and collects no reuse. The prefill regression
(−3.53%, −5.00%, arm mean −4.27%) is consistent with exactly that. This is the first time the
prefill-packed idea has been measured rather than argued about, and it loses.

**The `ik` gap re-measured with the same protocol, per pass** (`B1` vs `A0`):
**+37.72% / +44.30% decode** and **+36.06% / +36.34% prefill**. The prefill gap is the tightest of the
three stages (Stage 7: +20.75%, Stage 8: +34.0%, tonight: +36.2%); the decode gap swings with `A0`'s
own pass-to-pass drift (the +44% pass is the one where `A0` read 6.2958).

### 3.7 The D1 confirmation pair: a harness bug found, and a clock that would not settle

The first `phase1b_confirm` (05:02) **did not measure `D1` at all**, and the reason is a real harness
bug rather than thermal state:

```
S7ENV tag=prof-ours-D1 … vars=…,BONSAI_Q8_0_X4=unset,BONSAI_MULMATID_FLAT=unset,…
S7FAIL engine died
error: invalid argument: BONSAI_Q8_0_X4=2
```

`phase1b_confirm` passed `$D1_ENV` in `gated_pair`'s **extra-flags** slot, so the five `BONSAI_*`
assignments became `llama-server` **argv**. The engine exited immediately (0 profiler dumps), and the
run produced no `S8THR` line. **The canary gate refused the pair** (`NO S8THR line: cannot prove the
measured window was un-throttled`) rather than accepting a pair with no data — which is the only reason
this surfaced as "unmeasurable" instead of as "`MUL_MAT_ID` delta = 0.00%, no change".

Two fixes were made and verified:

1. `gated_pair` gained an optional `envB` (the retry loop and its unit test are unchanged); the `D1`
   arm now runs with `env=[BONSAI_Q8_0_X4=2 BONSAI_PK_REPORT=1 BONSAI_PROF=1 BONSAI_PROF_EVERY=1]`,
   confirmed in the driver log, and produces **1154 profiler dumps**.
2. `s9_d1_confirm.py` **crashed with `IndexError`** on a log with no dumps instead of failing closed.
   It now raises `NotMeasured` and prints
   `S8D1_CONFIRM=INVALID (no budget: an arm produced no profiler dumps)` with exit code 2.

A new `1b` phase runs that pair alone (`s8_shootout.sh 1b`), so the confirmation can be re-run without
the whole matrix. `HOST_CHECK=1` still reports **0 failures** afterwards.

**The re-run (05:34) measured both arms, and then the clock canary failed:**

```
GATED_DELTA_NET   5.3046 -> 5.3963  +1.73%
# worst canary |delta| = 4.70% (limit 2.5%)
S8PROF_GATE=FAIL   -> CLOCK MISMATCH - INVALID: discard this comparison and re-run the pair
```

That is the gate working as designed: after 5 h of continuous measurement the two arms did not run at
the same effective clock, so their per-op budgets are not comparable. The script's own retry
(attempt 2/2, `WARM_COOL_S=120`, 240 s extended cooldown) was running when this section was written;
its result is recorded in §3.8.


### 3.8 Three fixes in one section, because they are the same failure class

Each of these looked like a clean result and was not:

| what looked fine | what was actually happening | the fix |
|---|---|---|
| `phase1b_confirm` "ran" `D1` | `$D1_ENV` sat in the **flags** slot; `llama-server` exited with `invalid argument: BONSAI_Q8_0_X4=2` before its first graph | `gated_pair` gained an `envB` parameter; the arm now runs with the env in the env |
| `s9_d1_confirm.py` "produced a delta" | it **crashed with `IndexError`** on a log with 0 profiler dumps | raises `NotMeasured`, prints `S8D1_CONFIRM=INVALID (no budget)`, exit 2 |
| the `ik` gap | hidden in the first place by `ARMS` double-appending the candidates (§3.4) | relaunch with `ARMS="A0 AP2 B0 B1 B2"` |

**Part 4 of the `D1` ship bar is NOT established.** The corrected pair ran twice and the same-engine
clock canary failed both times:

| attempt | `WARM_COOL_S` | `GET_ROWS` | `GATED_DELTA_NET` | worst | verdict |
|---|---|---|---|---|---|
| 1 | 90 | +4.70% | +1.73% | **4.70%** | FAIL |
| 2 | 120 (+240 s extended cooldown) | −1.85% | **−3.44%** | **3.44%** | FAIL |

Both arms produced 1154 profiler dumps and identical call counts (`MUL_MAT_ID` 144/144,
`MUL_MAT` 833/833, `GET_ROWS` 174/174), so the pair is *measurable*; it is the **clock** that did not
match, after 6.5 h of continuous load. Per the pre-declared rule the pair is INVALID and its
`MUL_MAT_ID` delta is **not quoted anywhere in this report** — the helper said so itself:

```
S8D1_CONFIRM=INVALID (GATED_DELTA_NET 3.44%)
# same-engine clock mismatch: the pair is not comparable, do NOT publish the MUL_MAT_ID delta
```

**This does not change the ship decision**, because `D1` already fails the decisive clause (negative
in both mirrored decode passes) and no amount of attribution can make a slower arm ship. It does mean
tonight cannot say *why* `D1` lost, only that it lost — and the honest state is "cause unestablished,
candidate rejected on measured speed".

**One thing the pair did establish for free:** the device profiler reports `GET_ROWS 174 calls` per
token, independently reproducing the §2 host census on the real device. The two methods agree exactly.


---

## 4. Ship decision: **nothing ships tonight**

| candidate | required evidence | measured | verdict |
|---|---|---|---|
| `D1` `BONSAI_Q8_0_X4=2` | positive vs `A0` in both mirrored decode passes | **−5.99% / −0.50%** | **NO SHIP** |
| `C2` `BONSAI_MOE_X4_PREFILL_PACKED=1` | positive vs `A0` in both mirrored prefill passes | **−3.53% / −5.00%** | **NO SHIP** |

Both candidates were valid, engaged, and measured on a clean protocol — and both lost. The decision is
therefore the "otherwise" branch of Step 4, executed literally:

* **no default was flipped** (`bonsai_q8_0_x4_mode()` stays `0` unless `BONSAI_Q8_0_X4` is set;
  `bonsai_q2_0_x4_prefill_packed_mode()` stays `0` unless `BONSAI_MOE_X4_PREFILL_PACKED` is set);
* **`build-arm64-ship` was not rebuilt** for shipping (it was rebuilt for the host-side diagnostic
  verification only; the profile/behaviour of the shipped library is unchanged from Stage 8);
* **no `jniLibs/arm64-v8a/` sync** in `MoeDroid` or `MonsterQwenDroid`;
* **the Stage 7 APK was not touched**: `adb install` was never run, `apk-out/` was not refreshed;
* **`patches/llama.cpp-moedroid.patch` was not regenerated** — see §3.3, where its staleness is
  recorded as a finding. Regenerating it belongs to the same edit as shipping a default, and no
  default shipped.

### 4.1 The Kotlin UI gate: not exercised, and that is the honest outcome

`MoeDroid/app/src/main/java/com/prismml/bonsaidroid/ui/ChatScreen.kt` carries an **uncompiled,
unreviewed UI edit** in the worktree (flattened chat header plus a stream-follow/scroll change and a
dropdown). The Step 4 Kotlin gate (`:app:compileDebugKotlin`, then `:app:assembleDebug`, install,
and an explicit note that the shipped APK contains the unreviewed edit) applies **only when something
ships**.

**Since nothing shipped, the unreviewed `ChatScreen.kt` edit is not in any APK installed or produced
tonight**, and the gate was deliberately not run: building and installing an APK whose only change is
an unreviewed UI edit would have been a delivery with no engine change behind it. The edit remains
uncompiled and unreviewed in the worktree, exactly as it was handed over.

### 4.2 What would change the decision

For `D1`, in priority order: (a) the reason it is slower — which part 4 could not establish under a
failed clock canary, and which therefore needs a cooled device in a fresh window rather than more
heat; (b) a version that removes the packed-accessor overhead on the *decode* path, since the cache
already hits 98.5% and the pack cost is therefore not the problem; (c) nothing at all on the strength
of the 93.8% qemu hit rate — reuse was never the missing piece.

For `C2`, the mechanism is measured and fatal: 5.5% hit rate on a 262-token prefill means the two-slot
cache cannot pay for itself at that shape (§3.1). Any future attempt needs a cache indexed by token
*row* rather than by `(pointer, n)`, or no cache at all.


---

## 5. On-device `P3a` Gate 3(b) for `BONSAI_MOE_X4=2` (device-vs-device)

This section supersedes an earlier, incomplete draft of itself: the first attempt's mode-2 run was
streamed to the host and a **host power loss at 07:07 truncated it at 12/32 chunks**. The fragment is
preserved at `.stage9/dev-ppl-mode2-fragment-12of32.log` and no verdict was drawn from it. The driver
was then changed so that **every run writes its log on the device**
(`/data/local/tmp/ppl/s9-ppl-mode*.log`) and is pulled afterwards, so a host failure can never again
destroy the measurement — only the pull. Mode 1 survived that loss in full and was reused; the device
did not reboot between the two runs (`/proc/uptime` 437,753 s at 07:16), so the pair is still one boot,
one model file, one flag set.

### 5.1 The pinned protocol, as executed

| item | value |
|---|---|
| binary | `upstream-llama/build-arm64-ship/bin/llama-perplexity` (ship build, `MOE_X4=1` is its default) |
| model | varH shard 1 on `/sdcard` (sha256 verified `e7c69452…` at the start of the window) |
| flags | `-c 512 --chunks 32 -t 6` — **identical on both runs**, thread count matched |
| run 1 | `BONSAI_MOE_X4=1` -> `.stage9/dev-ppl-mode1.log` — **32/32 chunks, final PPL 2.9049 ± 0.06374** |
| run 2 | `BONSAI_MOE_X4=2` -> `.stage9/dev-ppl-mode2.log` — **32/32 chunks, final PPL 2.9063 ± 0.06391** |
| host reference | varH host PPL **2.9036**; the device reads **+0.045%** from it, so the device pipeline is sound |
| estimator | `MoeDroid/tools/s6_ppl_recheck.py`, the established per-chunk recovery `g_i = c_i^i / c_(i-1)^(i-1)`, unchanged, now with the paired t, the 95% CI and an `--pair` verdict block plus a 7-assertion `--selftest` |

### 5.2 The result: the paired statistic says zero, the per-chunk sanity gate says FAIL

```
chunks parsed: 32/32 both arms
max per-chunk |g2-g1|/g1 = 2.466e-02   (limit 5e-03; 0 == vacuous)
paired mean = +0.052884%   sd = 0.982267   SE = 0.173642
paired t = +0.305   (|t| < 2.0 required)
95% CI = [-0.301259%, +0.407027%]   (t_crit 2.0395, df 31)
FAIL (NUMERICAL BUG): drift 2.466e-02 exceeds the float-associativity bound 5e-03
S9PPL_PAIR=FAIL
```

**Verdict, by the pre-declared rule: FAIL.** Step 5.3 makes a non-vacuous PASS conditional on
`max_i |g2_i - g1_i| / g1_i <= 5e-3`; the measurement is **2.47e-2, 4.9x over**, so the pair does not
reach the verdict criterion and the fork **keeps `MOE_X4=1`**. Nothing is flipped, and the `/FAIL`
branch's action is the same as §4's: no default changes.

The pair is **non-vacuous** — the flag demonstrably took effect (`2.9049` vs `2.9063` final, and 22 of
32 chunks differ by more than the bound), so this is not the "identical series" failure mode.

### 5.3 What the two halves of the gate actually measured, and why they disagree

The drift is **pervasive, not outlying**: median 7.9e-3, p75 1.15e-2, p90 1.31e-2, max 2.47e-2 at
chunk 2, with **22/32 chunks above 5e-3** and 11/32 above 1e-2. Yet the *signed* per-chunk differences
average to **+0.053%** with a per-chunk sd of **0.98%**, so `t = +0.305` and the CI straddles zero.

That combination is coherent and it is the important finding: **`MOE_X4=2` changes individual chunk
likelihoods by ~1% at random, with no systematic direction.** A paired t-test is blind to exactly that
— it tests the mean, not the churn — which is why the max-drift clause exists, and why **the clause,
not the t-test, is the operative one here**. The likelihood churn is far too large for the kernel's
measured 2.9e-7 dot-product error: ~1% chunk-ppl churn corresponds to roughly 1-2 near-tie token flips
per 512-token chunk, i.e. logit perturbations in the 1e-4..1e-3 range, **~1000x the error the kernel
gate measured**.

**A plausible and testable cause: the kernel gate's shape coverage misses the real shapes.** Gate 3(a)
tested `nb ∈ {1, 2, 3, 5, 8, 16, 32, 64}`, but the deployed expert matmuls have `n_embd = 2560` and
`n_ff_exp = 640`, i.e. **`nb = 40` for `ffn_up_exps`/`ffn_gate_exps` and `nb = 10` for `ffn_down_exps`**
(`nb = n/64`). Neither 40 nor 10 is in the tested set, and mode 2's packed accumulate alternates
accumulators across blocks — a pattern that is sensitive to the block count's parity. That hypothesis
is testable on the host under `qemu-aarch64-static` without the phone, and is the first thing a next
window should run; it is **not** claimed here as established.

### 5.4 Two control experiments, and why the literal drift clause cannot be the verdict

The FAIL above is a *rule* verdict; the two controls below decide what it means. Both were run after
it, because a gate that fails is first an accusation against the assertion.

**(a) The identical-config null is exactly zero.** Re-running `MOE_X4=1` with the same flags and
comparing the first 8 chunks against the stored mode-1 run:

| chunk | mode 1 | mode 1b (same config) | \|rel\| null | mode 2 | \|rel\| flag |
|---|---|---|---|---|---|
| 1 | 2.906400 | 2.906400 | **0.000e+00** | 2.915000 | 2.96e-03 |
| 2 | 6.235515 | 6.235515 | **0.000e+00** | 6.081753 | 2.47e-02 |
| 3 | 2.021059 | 2.021059 | **0.000e+00** | 2.041505 | 1.01e-02 |
| 4 | 1.769852 | 1.769852 | **0.000e+00** | 1.747345 | 1.27e-02 |
| 5 | 1.846622 | 1.846622 | **0.000e+00** | 1.816374 | 1.64e-02 |
| 6 | 1.894666 | 1.894666 | **0.000e+00** | 1.909661 | 7.91e-03 |
| 7 | 1.939352 | 1.939352 | **0.000e+00** | 1.923109 | 8.38e-03 |
| 8 | 2.302983 | 2.302983 | **0.000e+00** | 2.277183 | 1.12e-02 |

So the pipeline is **bit-reproducible**: the measurement has no run-to-run noise, and the ~1.2e-2
churn in §5.2 is genuinely caused by the flag. That removes the "instrument noise" explanation.

**(b) The kernel is as accurate as stock at the deployed block counts, so the churn is not a kernel
bug.** The Gate 3(a) harness tested `nb ∈ {1,2,3,5,8,16,32,64}`; the real expert matmuls run at
**`nb = 40`** (`n_embd = 2560`) and **`nb = 10`** (`n_ff_exp = 640`). Both were added and the test
rebuilt and re-run under `qemu-aarch64-static`:

```
nb=10  n=640   mode1 BIT-EXACT  mode2 max|d|=3.815e-06  D1 BIT-EXACT   ref-vs-oracle max|d|=3.952e-06
nb=40  n=2560  mode1 BIT-EXACT  mode2 max|d|=1.144e-05  D1 BIT-EXACT   ref-vs-oracle max|d|=9.797e-06
nb=64  n=4096  mode1 BIT-EXACT  mode2 max|d|=6.104e-05  D1 BIT-EXACT   ref-vs-oracle max|d|=4.108e-05
ALL CHECKS PASSED
```

**Mode 2's error at the real shapes is the same magnitude as the *stock* kernel's own error against a
double-precision oracle** (3.8e-6 vs 4.0e-6 at `nb=10`; 1.1e-5 vs 9.8e-6 at `nb=40`). It is not a
broken kernel and it is not less accurate than the kernel it replaces. The hypothesised size-coverage
gap is **refuted** — and it is still the right change to the harness, because the gate should test the
shapes it certifies.

**(c) Therefore the `5e-3` point-wise bound is mis-specified, and the controls prove it.** A kernel that
is *equally accurate* but associates differently flips a few near-tie tokens per chunk, which moves that
chunk's `g_i` by ~1e-2; the paired mean stays at zero because the flips have no direction. That is
exactly what was measured: churn ~1.2e-2, mean +0.053%, `t = +0.305`. The bound treats "any
re-association" as a numerical bug, so **no genuine kernel change can ever pass it** — while the
identical-config null (0.0) passes trivially. A criterion that is zero for the null and ~1e-2 for every
true alternative cannot discriminate.

### 5.5 The verdict, stated in both registers

| register | verdict | evidence |
|---|---|---|
| **the literal pre-declared rule (§5.3)** | **FAIL** — max drift 2.47e-2 > 5e-3 | §5.2 |
| **the stated PASS criterion (§5.4: non-vacuous, \|t\| < 2.0, 0 in the 95% CI)** | **PASS** — non-vacuous, `t = +0.305`, CI [−0.301%, +0.407%] | §5.2 |
| **the mechanism check (§5.4 controls)** | no kernel defect; error at `nb=40` is *equal to stock's own error vs the oracle*; the measurement is bit-reproducible, so the churn is the flag's re-association, not noise | §5.4 (a), (b) |

**What is recorded, without relaxing anything: `MOE_X4=2` shows no measurable quality regression —
paired +0.0529%, `t = 0.305`, zero inside the CI — and its likelihood churn is the expected signature of
an equally-accurate-but-differently-associated kernel, not a numerical defect.** The literal drift
clause still says FAIL, and it is *not* being overridden or waived: it is being reported as
**miscalibrated**, with the control experiments that show why, so the next window can replace it with a
criterion that discriminates (calibrate the bound from a "different code, equal accuracy" control rather
than from an identical-code null; the observed value is ~1.2e-2 in 10/10 of the largest chunks).

**Operational consequence: none, and that is deliberate.** A PASS would have changed no defaults and
shipped nothing (§5.4 of the brief). Because the two registers disagree, Gate 3(b) for `MOE_X4=2`
remains **formally OPEN** — it is *not* claimed as cleared — and the fork keeps `MOE_X4=1`. Nothing in
§4 changes.

### 5.6 The explicit no-ship rule, executed

A Step-5 PASS would have changed no defaults and shipped nothing. A FAIL records FAIL and stops. So:

* **`BONSAI_MOE_X4` default is unchanged (mode 1, bit-exact)**; mode 2 stays an env-gated experiment;
* Phase 3's precondition — `MOE_X4=2` quality clearance — is **NOT met**, and is now blocked by a
  measured likelihood-churn result rather than by an untested assumption;
* no rebuild, no APK, no `jniLibs` sync, no patch regeneration (consistent with §4).


---

## 6. Instrument findings: the "f32 MUL_MAT = 130" row was GET_ROWS counted twice

The prep left Row 2 open: *"f32 `MUL_MAT` call count: derived as 130 (833 − 703), NOT independently
counted"*. Tonight it was counted, and the answer is **not 130 — it is 0**, and the 5.758 ms attributed
to it was never matmul time.

| source | f32 "MUL_MAT" calls/token | what it actually is |
|---|---|---|
| prep, derived by subtraction | 130 | arithmetic on two totals |
| device `BONSAI_PROF` per-type table (before fix) | **184** (mm=48, mmid=0) | internally inconsistent: 184 ≠ 48 + 0 |
| **direct graph census tonight** (`BGRAPH … mm=785 mm_f32=0 mmid=144`) | **0** | the graph contains no f32 `MUL_MAT` node at all |

**The cause is one line in the profiler.** The per-type accumulator took
`node->op == GGML_OP_MUL_MAT || … MUL_MAT_ID || … GGML_OP_GET_ROWS`, keyed on `src[0]->type`, so every
f32 `GET_ROWS` was booked as an f32 matmul. On this model that is **174 f32 gathers**, and the tell is
the timing: the prep's "MUL_MAT f32 **5.758 ms**" and "GET_ROWS **5.791 ms**" are the same work measured
twice, to within 0.6%. The device table tonight reproduced it exactly (`MATMUL f32 5.874 ms` vs
`GET_ROWS 5.978 ms`).

**Fixed and verified.** The condition now admits only `MUL_MAT` / `MUL_MAT_ID`. Before -> after on the
same host run, same model:

```
before:   MATMUL f32  5.874 ms/token  184 calls (mm=48 mmid=0)   ...
after:    (no f32 row at all - the graph has no f32 MUL_MAT node)
          GET_ROWS    6.025 ms/graph  1044 calls
```

**Consequences, stated plainly:**

* **Row 2 stays unestablished, and worse than before** — the row it was going to certify does not exist.
  Any Phase-4 arithmetic that reserved 130 f32 matmuls (or 5.76 ms) is budgeting for gathers that are
  already counted under `GET_ROWS`. Removing the double count lowers the "matmul" budget by ~5.8 ms of
  a ~147 ms token, i.e. the per-type table was ~4% pessimistic overall.
* **The three-way "counts differ" verdict would have misfired.** It compares per-type call counts across
  engines, and with `GET_ROWS` folded in, `f32` compares ours (174 gathers) against ik (14) — a 12x
  count difference that has nothing to do with matmul kernels.
* **No quoted number tonight depends on the old table.** The device numbers in §3 are t/s from the arm
  protocol and the PPL pair; the `MUL_MAT_ID` cells used by the D1 confirmation are a per-op row, not a
  per-type row. The device libraries were **not** rebuilt with this fix, so tonight's measurements are
  exactly as run; the fix applies to the next build.
* The f32/gather conflation also means the `us/call` column for `f32` (31.9 µs) was a gather latency, not
  a matmul latency — one more reason not to read kernel quality from that row.

### 6.1 Every instrument defect found this window

| # | defect | how it surfaced | fix |
|---|---|---|---|
| 1 | per-type table booked `GET_ROWS` as matmul, so "f32 130 calls / 5.758 ms" was gathers twice | Row 2 counted directly: graph has 0 f32 `MUL_MAT` | condition restricted to `MUL_MAT`/`MUL_MAT_ID`; verified before/after |
| 2 | Stage 9 PK hook called outside its `__aarch64__` guard -> **x86 build did not compile** | first census build failed | guarded identically; aarch64 output unchanged |
| 3 | `phase1b_confirm` passed `$D1_ENV` as **flags**, so the D1 arm died before warmup | canary refused a pair with no `S8THR` | `gated_pair` gained `envB`; arm runs with the env in the env |
| 4 | `s9_d1_confirm.py` **crashed** on a dump-less log | traceback instead of a verdict | raises `NotMeasured`, prints `INVALID (no budget)`, exit 2 |
| 5 | `ARMS` containing C1/C2/D1 is **appended again**, giving 11 arms/pass and overwritten logs | driver banner showed the doubled list | relaunched with `ARMS="A0 AP2 B0 B1 B2"`; aborted logs preserved |
| 6 | device PPL run **streamed to the host**; a host power loss destroyed it at 12/32 chunks | measurement lost | every run now writes on the device first, pull after |

**The pattern is the same in all six: the instrument reported something plausible and wrong.** None of
them was found by reading the code first; five were found because a gate refused to pass, and one
because a control experiment (identical config -> 0.000e+00) said the noise was not noise.

---

## Summary

| step | outcome |
|---|---|
| **1. preflight** | host gate **0/27 failures**, analyzer selftest PASS; device arrived Dozing with the shade focused, corrected to Awake/ON/unlocked/focused, caps `3532800`/`3840000` |
| **2. host census** | **174 executed `GET_ROWS`** enumerated by tensor name (175 built, one `COMPUTE=0`); prep's 96-QSA attribution **corrected to 48**, with **72 recurrent-state gathers (113 MB/token = 94% of the 5.79 ms)** that the prep missed; `ik chunks_per_expert` verified (6/expert at `-t 6`, ~107 rows, vs our 16-row blocks); **x86 build unbroken**; **f32 row exposed as `GET_ROWS` double-count** |
| **3. on-device Phase 1** | identity **12/12 PASS, zero arms dropped**; 32/32 timed arms **valid** (`6/6`, decode `throttled=0`); **`D1` −5.99% / −0.50%** and **`C2` −3.53% / −5.00%** — both negative in both passes; ik gap re-measured at **+37.7% / +44.3% decode, +36.1% / +36.3% prefill** |
| **4. ship decision** | **nothing ships.** No default flipped, no rebuild, no `jniLibs` sync, Stage 7 APK untouched, patch not regenerated (its staleness recorded). The unreviewed `ChatScreen.kt` edit is therefore **not in any APK** |
| **5. `P3a` Gate 3(b)** | device pair complete, **32/32 both arms**: mode 1 **2.9049**, mode 2 **2.9063**; paired **+0.0529%**, **t = +0.305**, **95% CI [−0.301%, +0.407%]** (0 inside) — but max per-chunk drift **2.47e-2 > 5e-3**, so the literal rule says FAIL. Controls show the identical config is **bit-reproducible (0.000e+00)** and the kernel is **as accurate as stock at `nb=40/10`**, so the drift bound is miscalibrated (zero for the null, ~1e-2 for any true alternative). Gate stays **formally OPEN**; `MOE_X4=1` kept |
| **6. instrument** | six defects found and five fixed, listed in §6.1 |

**Device left clean:** no harness engine running, port 8299 free, `stay_on_while_plugged_in` reverted to
`0`, app foregrounded. The only engine process is the app's own (PID 5073), which is its normal state.

