# Stage 8 daytime prep (host-only)

Date: 2026-10-18 (daytime). **The phone is unplugged for a 12-hour shift: no `adb` command was run
in this session.** Every claim below is a host-side measurement or a static read of the binaries and
sources.

This file is written incrementally, one section per task, in the order the session override
specified: **Task 2 -> Task 3 -> Task 1 -> Task 4**.

| step | scope | status |
|---|---|---|
| Task 2 | `BONSAI_PROF` in `ik_llama.cpp`: true thread-0 wall-clock per op and per weight type | done |
| Task 3 | in-process post-warmup cooldown protocol (`s8_arm_ll.sh`, `s8_shootout.sh`) | done |
| Task 1 | static disassembly comparison, `ik` vs ours, 6 weight types | done |
| Task 4 | prototype gated on Task 1: the 4-row x 4-token tile, arm `C1` | done (unmeasured) |

---

## Task 2: `BONSAI_PROF` in `ik_llama.cpp` now measures true wall clock, per op and per weight type

### 2.1 What was wrong, stated precisely

Stage 6 §2.5 and Stage 7 §4 both had to add a caveat to every ik number, and the caveat was the same
each time: the ik profiler timed **every OpenMP worker** around its own
`ggml_compute_forward` call and summed the per-worker rows at dump time. Three consequences:

1. the dump reported the sum of *CPU-time shares*, not wall time: measured tonight on the host with 4
   threads, that inflation is **3.55x** (36.844 ms of worker rows against 10.366 ms of wall time);
   on the device with 6 workers it is the ~537 ms/graph that Stage 6 recorded for a graph whose real
   wall time was a fraction of that;
2. it called `clock_gettime` **6x per node**, which perturbs the very thing it measures;
3. the per-op shares could not be compared against our fork at all, because our fork's `BONSAI_PROF`
   is wall-clock and its totals agree with its own measured t/s (Stage 6 §2.5 verified that
   explicitly: 188.633 ms/token against 5.30 t/s).

### 2.2 What it does now

`ik_llama.cpp/ggml/src/ggml.c`, `ggml_graph_compute_thread`:

* **Only `state->ith == 0` takes timestamps.** `t0` immediately before `ggml_compute_forward`, and
  `dt` immediately **after `ggml_barrier(state->shared)`**. Because every worker must reach that
  barrier, `dt` is the node's wall-clock time on the whole pool, not one worker's share - which is
  exactly the semantics our fork uses.
* Two `clock_gettime` calls per node total, not 6x that.
* Rows: per op (`g_bp_ns`, `g_bp_calls`) and per weight type for **`MUL_MAT`, `MUL_MAT_ID` and
  `MOE_FUSED_UP_GATE`** (`g_bp_type_ns`, `g_bp_type_calls`) keyed on `src0->type`, which covers all
  eight types varH uses (`q2_0`, `q3_K`, `q4_K`, `q5_K`, `q8_0`, `iq4_xs`, `bf16`, `f32`).
* Thread 0 also accumulates the whole node loop's wall time (`g_bp_wall_ns`), so the sum of the op
  rows can be checked against the graph's real time instead of being taken on trust.
* The old per-worker sum survives as an **opt-in diagnostic** (`BONSAI_PROF_WORKERS=1`) because it is
  the thing that produced the bad Stage 6/7 numbers and it is useful to be able to show the
  inflation factor rather than assert it.

### 2.3 The output is byte-compatible with our fork, so one parser reads both engines

The dump block is now identical to the fork's (`ggml/src/ggml-cpu/ggml-cpu.c`, `g_prof_dump`):

```
=== CPU OP PROFILE after 41 graphs minflt=… majflt=… ===
  MUL_MAT                     4.74 ms total    0.226 ms/graph        42 calls
  MATMUL q2_0                 3.83 ms total    0.183 ms/graph
BONSAI_PROF WALL graphs=41 wall_ms=10.366 sum_op_ms=10.348 sum_type_ms=9.964
```

The `BONSAI_PROF WALL` / `BONSAI_PROF WORK` lines are diagnostics that `s4_prof.py` ignores. **The
fork's parser was not modified**: running it on captured ik output produces the full table, which is
the proof that the two engines' budgets are now directly comparable:

```
$ python3 MoeDroid/tools/s4_prof.py .stage8/ik-prof-capture.txt decode 20
# 33 dumps; window = graphs 14..33 (20 graphs)
op                       ms/token  calls/tok   share
MUL_MAT                     0.210        2.0   96.6%
ADD                         0.003        2.0    1.4%
RMS_NORM                    0.003        1.0    1.4%
SCALE                       0.002        1.0    0.7%
TOTAL                       0.218
## matmul time by weight type (includes MUL_MAT and MUL_MAT_ID)
q2_0              0.171   78.6%
```

### 2.4 Verification: the per-op rows do add up to the graph's real execution time

`MoeDroid/tools/s8_prof_verify.c` builds a 7-node graph (three `mul_mat` over `q2_0` / `q8_0` /
`bf16`, two `add`, `rms_norm`, `scale`), plans it **once**, and times 40 `ggml_graph_compute` calls
from the outside. The profiler independently times the same graphs. Compared over exactly the same
40 graphs (cumulative dump at graph 41 minus cumulative dump at graph 1):

| quantity | value | vs external |
|---|---|---|
| external wall of 40 graphs (the reference) | **10.494 ms** | - |
| profiler: sum of the per-op rows | **9.806 ms** | **-6.56%** |
| profiler: thread-0 node-loop wall | 9.820 ms | -6.42% |
| profiler: sum of the per-type rows | 9.466 ms | matmuls are 97% of op time |
| old per-worker sum (diagnostic) | 36.844 ms | **3.55x inflated**, 4 threads |

Reading it: **the per-op rows account for 99.86% of the node loop's wall time** (9.806 vs 9.820), and
the node loop accounts for **93.6%** of `ggml_graph_compute`. The missing 6.4% is the OpenMP parallel
region's fork/join per graph, which happens outside every node and therefore belongs to no op - it is
not a measurement error and it is now a known, bounded quantity rather than an unexplained gap.

Per-type coverage was checked one type at a time, because the multi-op synthetic graph turned out to
be an awkward witness (one of its three matmuls is not selected for timing, a property of that
synthetic graph rather than of the profiler). Every type varH uses records correctly, and for each
one the `MUL_MAT` total equals the `MATMUL <type>` total:

| type | `MUL_MAT` (3 calls) | `MATMUL <type>` | equal? |
|---|---|---|---|
| `q2_0` | 0.35 ms | 0.35 ms | yes |
| `q3_K` | 0.19 ms | 0.19 ms | yes |
| `q4_K` | 0.24 ms | 0.24 ms | yes |
| `q5_K` | 0.26 ms | 0.26 ms | yes |
| `q8_0` | 0.29 ms | 0.29 ms | yes |
| `iq4_xs` | 0.09 ms | 0.09 ms | yes |
| `bf16` | 0.11 ms | 0.11 ms | yes |

### 2.5 Builds

| build | result |
|---|---|
| `ik_llama.cpp/build-arm64-ik-prof` (`-DBONSAI_PROF=1`) | rebuilt, `libggml.so` 321,929,352 B; carries the new format strings and **none** of the old `BONSAI_PROF OP` ones |
| `ik_llama.cpp/build-arm64-ik` (shipping, no profiler) | rebuilt for consistency; `CPU OP PROFILE` absent, as it must be |
| `.stage6/host-ik-build` (x86, `GGML_NATIVE=ON`) | rebuilt with `-DBONSAI_PROF=1` for the host verification above |

Two defects in the *test harness* were found and fixed while doing this, and both are worth recording
because they would have produced a false "verified" verdict:

1. the filler wrote 34 bytes past the end of every quantized row, corrupting the heap;
2. the test called `ggml_graph_compute_with_ctx` in a loop, and that function allocates a
   work-buffer object **inside the context** on every call, so the 34th call exhausted the context and
   crashed in the error path. The test now plans once and reuses the plan, which is also the
   canonical usage and makes the external measurement pure compute.

---

## Task 3: the in-process post-warmup cooldown protocol (`s8_arm_ll.sh`, `s8_shootout.sh`)

### 3.1 The problem it fixes, in Stage 7's own numbers

Stage 7 §2.0 measured the thermal trip at **~155 s into every arm**: caps fell from
`3532800/3840000` to `2400000/2841600` at ~105 C, and because the 12 warmups alone take ~156 s the
trip landed inside the warmup phase, so **every measured window in Stage 7 ran throttled**. The
engine's real rate (~6.4-6.5 t/s) was only visible in the warmup series.

The obvious fix - cool the phone between arms - does not work: the trip is caused *inside* each arm
by its own warmups, and Stage 7 confirmed caps were already back to the good profile after every
120 s inter-arm cooldown. What is needed is a cooldown **between the warmups and the measurement,
while the server keeps running**, because the point of 12 warmups is to build the ~13.5 GB expert
page cache and restarting the engine (or letting it idle long enough for the kernel to evict) throws
that away.

### 3.2 What `s8_arm_ll.sh` does now

New defaults and new output, all in `MoeDroid/tools/s8_arm_ll.sh`:

| element | behaviour |
|---|---|
| `WARM_COOL_S` | **90 s** default, in-process, between warmup #12 and measured request #1. `0` disables it |
| `S8COOL` | `begin cool_s=… rss_kb=… mem_avail_kb=… cached_kb=…`, then `end … caps_hist=<histogram across the cooldown>` |
| `S8CACHE` | `rss_before= rss_after= mem_avail_before= mem_avail_after= cached_before= cached_after=` |
| `S8FLAG` | emitted when the server's RSS drops **> 1%** across the cooldown, i.e. the page cache did *not* survive and the arm's numbers are suspect |
| `S8CAP gate=POSTCOOL` | the **cap gate is re-run after the cooldown and before request #1**, still requiring `cpu0=3532800` and `cpu6=3840000` for 3 consecutive 10 s polls; a timeout emits `S8FLAG measured_window_caps_not_good=1` |
| per-request caps | every measured request records `cap0a/cap6a/cap0b/cap6b` (either side of the request) and a `thr=` bit |
| `S8THR` | `measured=6 throttled=<n>`, and `S8FLAG thermal_in_measured_window=1` when `n > 0` |

The gate was factored into a function so the *pre-arm* and *post-cooldown* gates are the same code,
which is the only way the two can be trusted to mean the same thing. The old pre-arm gate line is now
`S8CAP gate=PRE` and the post-cooldown one `S8CAP gate=POSTCOOL`, so the log distinguishes them.

**Why RSS is the right cache probe:** the model is `mmap`ed, so its resident file-backed pages are
counted in the server's `VmRSS`. If the kernel were to evict that working set during the cooldown,
RSS would fall with it; `MemAvailable`/`Cached` are recorded alongside because they show whether the
machine had pressure at that moment. A 1% threshold is deliberately tight: 1% of 13.5 GB is 135 MB,
far more than any legitimate fluctuation, and far less than a real eviction.

The per-request `thr` bit is the second half of the same defence: even with the cooldown and the
post-cool gate, a *long* measured window can re-trip the throttle, and Stage 7 could not tell that
from a slow kernel. Now the arm counts it and flags itself.

### 3.3 `s8_shootout.sh`: Phase 1 is the profiler comparison

`MoeDroid/tools/s8_shootout.sh [1|2|3|all]`, and the phase order is the point:

| phase | arms | why it is in this position |
|---|---|---|
| **1** | `prof-ours`, `prof-ik`, `prof-ik-nofmoe` - our `build-arm64-prof` and ik's `build-arm64-ik-prof`, **no `--poll` anywhere**, cooldown protocol on | the override's requirement: the first ~20 minutes on the phone must produce the definitive un-throttled **ms-by-op and ms-by-weight-type** tables for both engines - the thing Stage 6 and Stage 7 both failed to produce, for two different reasons (worker-sum inflation, then clock confound). Each arm's engine stderr is pulled immediately (`prof-logs/s8-<tag>.log`) so the dumps exist before the next arm runs |
| 2 | 8 identity arms (`id-A0`, `id-A5`, `id-AP2`, repeats, and the two 262-token-prefill families) | cheap, decisive, and needed before anything ships. `AP2` is now the **tile-OFF control**, because Stage 8's shipped default has the tile ON |
| 3 | mirrored decode + prefill passes over `A0 A5 AP2 B0 B1 B2` (+`C1` when `C1_ENV` is set) | the timing dataset, with the cooldown protocol, both directions |

`C1` is the Stage 8 candidate slot: `C1_ENV="BONSAI_…=1" s8_shootout.sh 3` adds it to phase 3 and
`C1_ENV=… s8_shootout.sh 2` adds its identity gates. Phase 1 does not include it on purpose - the
budgets must be measured on the shipping kernels, not on an unvetted candidate.

Note what changed about `AP2`: in Stage 7 it was "enable the tile". Now the tile is the shipped
default, so `A0` *is* the tile and `AP2` (`BONSAI_MOE_X4_PREFILL=0`) is the control that measures what
it is worth today. The named contrast in the analyzer was flipped accordingly.

### 3.4 `s8_ll_table.py`: the analyzer now reports trustworthiness, not just speed

A new section answers "was this number measurable at all", per arm:

```
### did the cooldown keep the page cache, and was the measured window un-throttled?

| phase | pass | cfg | cooldown s | rss before -> after (kB) | cache | mem avail before -> after | throttled requests | flags |
| decode | 1 | A0  | 90 | 13498000 -> 13490000 | held -0.06% | 1250000 -> 1290000 | 0/6 | - |
| decode | 1 | A5  | 90 | 13498000 -> 12100000 | LOST -10.4% | 1250000 -> 1290000 | 2/6 | rss_dropped_pct=10.35 … |
```

The main table's `throttled` column is now the per-request count (`0/6`) rather than a cap-sample
percentage, and `cap gate` shows `PRE`/`POSTCOOL`/`POSTCOOL-TIMEOUT`. The synthetic-log selftest was
rewritten for Stage 8 and covers the new telemetry, including a deliberately cache-losing and
throttled arm so the flag path itself is tested rather than assumed:

```
PASS  A5 vs A0 contrast is -4.00%
PASS  shipped tile ON beats OFF on prefill
PASS  C1 candidate contrast present
PASS  cooldown/cache/throttle table present
PASS  healthy arm shows cache held
PASS  cache-loss arm is flagged
PASS  throttled request count surfaced
PASS  arm flag surfaced
SELFTEST PASSED: every injected delta AND every Stage 8 telemetry line is recovered
```

### 3.5 Host-side gate, runnable with the phone still away

`HOST_CHECK=1 s8_shootout.sh 0` - **19 checks, 0 failures**, no `adb`:

```
OK ship libggml-cpu.so            OK ik-profile has the Stage 8 format
OK prof libggml-cpu.so            OK ik-profile dropped the old per-worker-only format
OK ik libggml/libllama/server     OK ik ship build has no profiler
OK ik-prof libggml.so             OK our prof build has BONSAI_PROF
OK our ship build has no profiler OK libc++_shared.so / libomp.so
OK device arm script              OK arm script needs a cooldown   (WARM_COOL_S + cap_gate POSTCOOL)
OK analyzer parses S8 telemetry   OK 4/4 Stage 7 kernel symbols in the ship build
OK ik build has batch pinning
=== host-side prerequisites: 0 failure(s) ===
```

### 3.6 What this protocol does NOT fix

1. **A cap gate is a proxy for temperature, not the temperature.** `scaling_max_freq` returning to
   `3532800` means the thermal engine has released its limit; the core may still be at a lower OPP
   for a moment, and the governor (`walt`) is free to choose. The per-request `thr` bit catches a
   re-trip, but a partially-throttled window can still be a few percent slow.
2. **90 s is a guess that the first run must validate.** If `S8THR` is non-zero or `S8CACHE` shows a
   loss, the answer is a longer `WARM_COOL_S`, and the arm now says which of the two happened.
3. **The cooldown costs ~90 s per arm** (~+25 min on a 28-arm night) and it does not remove the
   need for mirrored passes: whatever residual drift remains is still what the position control
   measures.

---

## Task 1: static disassembly comparison, `ik` vs ours, 6 weight types

Method: `llvm-objdump -d` on `ik_llama.cpp/build-arm64-ik/ggml/src/libggml.so` and
`upstream-llama/build-arm64-ship/bin/libggml-cpu.so`, plus the kernels' own source, read with
`.stage8/loopcount.py`. No QEMU TCG plugins, no device.

The tool selects the function's hottest loop (the backward branch whose body contains the most
integer-dot / fp-accumulate instructions) and counts instruction classes in that body. **The metric
is "what clang emitted for that loop body", not "cost per weight"** - the loop bodies do not all
cover the same amount of work, so the per-type normalisation is stated explicitly where it is known
and marked where it is inferred.

### 1.1 What each engine actually dispatches

`ik` compiles a **specialised kernel per token-row count**: `IQK_SET_MUL_MAT_FUNCTIONS` fills a table
with `kernel<1> … kernel<8>` (`iqk_common.h:166-174`), and the second dimension, the number of
*weight* rows per call, is a runtime argument (`int nrc_x`) with a per-family cap encoded in the
kernel name (`x4` = 4, `r4` = 4, `r8` = 8). All of that is visible in the symbol table: each family
has exactly 8 instantiations, `<1>` … `<8>`.

Our fork has **one kernel per weight type, `nrc = 1`**, enforced by
`type_traits_cpu[type].nrows = 1` and `assert(nrc == 1)` inside each `ggml_vec_dot_*`, plus the
Stage 6/7 additions for `q2_0` (4 rows) and `q3_K` (4 rows, rejected on speed).

| type | `ik` symbol (`nrc_y = 1`) | `ik` rows per call | our symbol | our rows per call |
|---|---|---|---|---|
| `q2_0` | `mul_mat_q2_0_q8_0_x4<1>` | **≤4 weight rows × ≤8 token rows** | `ggml_vec_dot_q2_0_q8_0` (+ `bonsai_vec_dot_q2_0_q8_0_x4t2`) | 1 (+ 4 rows × 2 tokens) |
| `q3_K` | `mul_mat_q3_k_r4_q8_k<1>` | **≤4 × ≤8** | `ggml_vec_dot_q3_K_q8_K` (+ `bonsai_vec_dot_q3_K_q8_K_x4`, rejected) | 1 (+ 4 × 1) |
| `q4_K` | `mul_mat_q4_k_r4_q8_k<1>` | **≤4 × ≤8** | `ggml_vec_dot_q4_K_q8_K` | **1** |
| `q5_K` | `mul_mat_q5_k_r4_q8_k<1>` | **≤4 × ≤8** | `ggml_vec_dot_q5_K_q8_K` | **1** |
| `q8_0` | `mul_mat_q8_0_r8_q8_0<1>` | **≤8 × ≤8** | `ggml_vec_dot_q8_0_q8_0` | 1 (`nrows = 2` under i8mm) |
| `iq4_xs` | `mul_mat_iq4_xs_r8_q8_k<1>` | **≤8 × ≤8** | `ggml_vec_dot_iq4_xs_q8_K` | **1** |

So the structural gap is real and it is two-dimensional: **`ik` amortises the weight expansion across
up to 8 token rows and amortises the call overhead across up to 4-8 weight rows; we do neither except
for `q2_0`.** `q4_K`, `q5_K` and `iq4_xs` have no multi-row kernel at all in our fork.

### 1.2 How `ik` avoids the register-pressure trap - read from the source, not inferred

This is the part that matters most, because Stage 7's `MOE_X4=3` was built on the opposite
assumption and lost 3.6%. `iqk_gemm_legacy_quants.cpp:4511-4600`, the `q2_0` kernel, in full:

```cpp
template <int nrc_y>                                  // token rows: compile-time, 1..8
static void mul_mat_q2_0_q8_0_x4(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Q8<nrc_y, block_q8_0_x4> q8(info);                // activation repacked, contiguous
    int8x16_t qx[8];
    float32x4_t acc[nrc_y] = {};                      // ONE ACCUMULATOR PER TOKEN ROW
    for (int ix = 0; ix < nrc_x; ++ix) {              // weight rows: SEQUENTIAL
        for (int ib = 0; ib < nb/2; ++ib) {           // block pair = 128 weights
            /* 2 x vld1q_u8 + 8 x vqtbl1q_u8 + 8 x (shl, and, sub)  -> qx[0..7]  */
            for (int iy = 0; iy < nrc_y; ++iy) {      // token rows: UNROLLED by nrc_y
                /* 4 x vld1q_s8_x2 (32 B each) + 8 x vdotq_s32 + 3 x vpaddq_s32 + 4 x fma */
            }
        }
    }
}
```

Three properties, and each of them is the opposite of what our 4-row kernel does:

1. **The accumulator count is set by the token-row count, not the weight-row count.**
   `acc[nrc_y]` is 1 at decode. Weight rows are walked *sequentially*, so `nrc_x` costs no registers
   at all. Our `bonsai_x4_rows4_mode1` keeps two accumulators per weight row live (even/odd block
   alternation) = 8 accumulators, and that is the direct cause of its 55 stack accesses: Stage 7's
   mode 3 tried to buy them back by halving the rows per pass, which doubled the activation loads and
   lost 3.7% net.
2. **The 2-bit expansion is amortised across token rows, not across weight rows.** One `qx[0..7]`
   unpack serves all `nrc_y` token rows. Our `x4`/`x4t2` amortises it across weight rows instead.
   Both are legitimate; ik's version costs nothing in registers, ours costs four rows' worth.
3. **The activation is read packed and contiguous.** `Q8<nrc_y, block_q8_0_x4>` means the quantised
   activation is `block_q8_0_x4`-shaped: 128 contiguous int8 quants plus 4 fp16 scales per 128
   weights, loaded with `vld1q_s8_x2` (32 B per instruction). Ours reads plain `block_q8_0` at a
   34-byte stride, two 16 B loads per 32 weights - **twice the load instructions per activation byte**,
   with every second load straddling a 16-byte boundary. This is exactly the repack Stage 7
   prototyped (`BONSAI_Q8_0_X4`) - and it is why that prototype's *null* result is informative: we
   added the packing but kept our 4-row/1-token shape, so we paid the repack traffic without gaining
   ik's token-row amortisation.

### 1.3 Measured loop bodies

`.stage8/loopcount.py`, per the hot loop of each function. `vload` counts vector loads, `dotprod`
counts `sdot`/`udot`, `i8mm` counts `smmla`/`usmmla`, `stack` counts `[sp]` accesses (a diagnostic
only - Stage 7 proved a zero-spill kernel can be slower).

| type | kernel | loop insns | vload | sdot | i8mm | tbl | shift/mask | stack | loop scope |
|---|---|---|---|---|---|---|---|---|---|
| `q2_0` | ours `ggml_vec_dot_q2_0_q8_0` | 87 | 7 | 8 | 0 | 8 | 24 | 0 | 1 block = 64 w, 1 row |
| `q2_0` | ours `bonsai_x4_rows4_mode1` (A0 default) | **329** | 32 | 32 | 0 | 32 | 93 | **27** | 2 blocks = 128 w, **4 rows** |
| `q2_0` | ours `bonsai_vec_dot_q2_0_q8_0_x4t2` (shipped prefill) | **224** | 27 | 24 | 0 | 12 | 48 | 0 | 128 w, **4 rows x 2 tokens** |
| `q2_0` | ik `mul_mat_q2_0_q8_0_x4<1>` | 142 | 11 | 12 | 0 | 12 | 28 | 0 | 128 w, ≤4 rows, 1 token |
| `q2_0` | ik `mul_mat_q2_0_q8_0_x4<2>` | 227 | 26 | 24 | 0 | 12 | 39 | 2 | 128 w, ≤4 rows, 2 tokens |
| `q3_K` | ours `ggml_vec_dot_q3_K_q8_K` | 223 | 7 | 16 | 0 | 0 | 79 | 0 | 1 superblock = 256 w |
| `q3_K` | ours `bonsai_vec_dot_q3_K_q8_K_x4` (rejected) | 432 | 30 | 32 | 0 | 0 | 150 | 12 | 256 w, 4 rows |
| `q3_K` | ik `mul_mat_q3_k_r4_q8_k<1>` | 289 | 13 | 32 | 0 | 0 | 133 | 2 | 256 w, ≤4 rows |
| `q4_K` | ours `ggml_vec_dot_q4_K_q8_K` | 144 | 16 | 16 | 0 | 0 | 38 | 5 | 1 superblock |
| `q4_K` | ik `mul_mat_q4_k_r4_q8_k<1>` | 245 | 23 | 32 | 0 | 0 | 80 | 11 | 256 w, ≤4 rows |
| `q5_K` | ours `ggml_vec_dot_q5_K_q8_K` | 168 | 12 | 16 | 0 | 0 | 65 | 0 | 1 superblock |
| `q5_K` | ik `mul_mat_q5_k_r4_q8_k<1>` | 360 | 39 | 32 | 0 | 0 | 137 | 16 | 256 w, ≤4 rows |
| `q8_0` | ours `ggml_vec_dot_q8_0_q8_0` | 31 | 6 | 4 | **4** | 0 | 2 | 0 | 1 block = 32 w |
| `q8_0` | ik `mul_mat_q8_0_r8_q8_0<1>` | 285 | 70 | **80** | **0** | 0 | 25 | 11 | 8 rows x 1 token |
| `iq4_xs` | ours `ggml_vec_dot_iq4_xs_q8_K` | 148 | 16 | 16 | 0 | 16 | 40 | 13 | 1 superblock |
| `iq4_xs` | ik `mul_mat_iq4_xs_r8_q8_k<1>` | 245 | 26 | 32 | 0 | 32 | 66 | 2 | 256 w, ≤8 rows |

**The numbers that matter, and the ones that do not.** Raw loop size is *not* comparable across a
type boundary and only partly comparable within one, because a body may cover one block or two. What
is unambiguous:

* **`sdot` is the engine on both sides.** `ik` uses **zero** i8mm (`smmla`/`usmmla`) in every one of
  these six kernels; our library has 260 i8mm instructions overall against ik's 92, and specifically
  our `q8_0` kernel uses 4 `smmla` where ik's `q8_0` kernel uses 80 `sdot` and no i8mm at all. So on
  `q8_0` the *instruction-selection* advantage is ours and the *row-batching* advantage is ik's.
* **Our 4-row `q2_0` kernel spends 2.9x the instructions of ik's per 128 weights per row**
  (329/4 = 82 against 142/1), and 27 of its 329 are `[sp]` accesses. It is slower per row and it wins
  in practice only because it serves four rows from one activation load. Our 4-row x **2-token** tile
  is the one that beats it: 224 insns for 8 row-tokens = **28 per row-token**, which is the best
  per-row-token figure on this table and is why it shipped.
* **`q4_K`/`q5_K`/`iq4_xs` have no multi-row kernel on our side at all**, and their `ik` counterparts
  are 1.7-2.1x larger per loop body for 4-8 rows of work.
* **`stack` is not a quality signal** (Stage 7): ik has 11-16 spill accesses in `q4_K`/`q5_K`/`q8_0`
  and is still faster there.

### 1.4 Verdict handed to Task 4

The clear, self-contained structural gap is: **a weight-row × token-row tile for the types that have
none, with the token-row dimension as the amortisation axis** (ik's `acc[nrc_y]` shape), rather than
the weight-row dimension (our `x4` shape, which is what `MOE_X4=3` doubled down on and lost).

Two candidate versions exist:

1. **Extend the shipped `x4t2` tile from 2 to 4 token columns** - the kernel
   `bonsai_vec_dot_q2_0_q8_0_x4t4` is already written, already **bit-exact** (Stage 7 §B.5), and is
   dead code because it was never wired. This is the cheapest possible test of ik's structure: same
   arithmetic, same bit-exactness, only the token-column count changes. **Chosen for Task 4.**
2. Add a packed-activation variant on top (`BONSAI_Q8_0_X4` + 4 tokens). Deferred: Stage 7 measured
   the repack alone as null-to-negative, and stacking two unmeasured changes in one arm is how this
   project has produced its worst measurements.

What Task 4 must NOT do, per the override: treat "zero spills" as the acceptance criterion. It is
recorded above as a diagnostic only.

---

## Task 4: the candidate - `BONSAI_MOE_X4_PREFILL=2`, the 4-row x 4-token `q2_0` tile (C1)

### 4.1 Why this candidate and not something else

Task 1's conclusion was that the gap is the **token-column amortisation axis**, not the weight-row
axis. That is not a guess: Stage 7 measured both directions of the same idea.

| shape | what it amortises | Stage 7 result |
|---|---|---|
| 4 weight rows x 1 token, rows sequential (our `MOE_X4=3`) | accumulator pressure down, **activation loads doubled** | **-3.62%** (four pairs, both phases) |
| 4 rows x 2 tokens (`BONSAI_MOE_X4_PREFILL=1`, shipped) | weight expansion across tokens | **+3.31% prefill**, both passes |
| 4 rows x 4 tokens (`x4t4`) | same axis, twice as far | **not measured until tonight** |

And the kernel for the third row of that table already existed: `bonsai_vec_dot_q2_0_q8_0_x4t4` was
written in Stage 7, verified bit-exact, recorded as "the 4-token tile cannot be made spill-free" and
then **left as dead code**. Task 4 wires it. This is the cheapest possible test of ik's structure -
identical arithmetic, identical bit-exactness, only the token-column count changes - and it does not
stack a second unmeasured change (the packed activation) on top.

**Spill count is a diagnostic here, not a criterion**, as the override requires: `x4t4` has a 256 B
frame with 9 stores and 15 loads to `[sp]`, against `x4t2`'s 0 B frame; Stage 7 proved that signal
wrong in both directions (`MOE_X4=3` had 0 spills and lost 3.6%; `MOE_X4=1` had 55 and won +4.5%).

### 4.2 What changed

1. **`bonsai_vec_dot_q2_0_q8_0_x4t4` now takes four explicit activation pointers and four explicit
   result pointers** - the same change `x4t2` needed in Stage 7 and for the same reason: in
   `mul_mat_id_one_chunk` four adjacent routed rows are four arbitrary rows of `wdata`, not four rows
   a fixed byte stride apart, and `dst_col` is permuted by the same router mapping. A stride API here
   would read the wrong data silently, which is exactly the bug class the identity gate exists for.
2. **`bonsai_q2_0_x4_prefill_mode()` now returns 0 / 1 / 2**: `0` = stock loop,
   `1` = 4 rows x 2 tokens (**default, unchanged**), `2` = 4 rows x 4 tokens (the candidate).
   `BONSAI_MOE_X4_PREFILL=2` selects it; the shipped default is untouched.
3. **Wired into `ggml_compute_forward_mul_mat_id_one_chunk`** as a 4-token pre-loop that runs before
   the existing 2-token loop, so the leftovers (0-3 tokens) fall through to the 2-token path and then
   to the stock single-token loop. The expert path is the right place for the prototype: it is 61.9%
   of prefill. The dense path keeps the 2-token tile.

### 4.3 Verification under `qemu-aarch64-static` (real ARM64, no device)

| level | check | result |
|---|---|---|
| kernel | `x4t4` with the four-pointer API vs `ggml_vec_dot_q2_0_q8_0`, 16 (row, token) pairs x `nb in {1,2,3,5,8,16}` | **BIT-EXACT, `0.000e+00`, all 96 pairs** |
| kernel | `x4t2`, re-run as a control after the same edit | **BIT-EXACT, `0.000e+00`** |
| graph | `BONSAI_MOE_X4_PREFILL=2` vs stock, 6 synthetic `MUL_MAT_ID` shapes, 8 threads and 1 thread | **byte-identical on all 6** |
| graph | `=2` vs `=2` on a repeat run | **stable** |
| coverage | tile calls per shape | 3200 / 768 / 2560 / 2176 / 0 / 0 - the same coverage as the 2-token tile, i.e. the multi-token shapes really do enter the 4-token loop, and the no-pair and single-token shapes correctly do not |

The coverage row is what makes the identity result meaningful rather than vacuous: a "byte-identical"
verdict on a path that never executed would be worthless, and this project has already been bitten by
exactly that once (Stage 7, `MUL_MAT_ID` case 4).

### 4.4 It is arm `C1` in tonight's shootout

`s8_shootout.sh` now defaults `C1_ENV=BONSAI_MOE_X4_PREFILL=2`, so

```
MoeDroid/tools/s8_shootout.sh 2     # identity gates incl. id-C1 and id-pf-C1
MoeDroid/tools/s8_shootout.sh 3     # timing: A0 A5 AP2 B0 B1 B2 C1, both phases, mirrored
```

adds the candidate without anyone having to remember the variable; `C1_ENV="" s8_shootout.sh 3`
skips it. The analyzer already carries the contrast (`Stage 8 candidate, decode/prefill (C1 vs A0)`)
and the `C1` label, and its selftest asserts that contrast is present.

**The decision rule, fixed before the data exists:** `C1` replaces `A0`'s tile as the default only if
it beats `A0` in **both** mirrored prefill passes and is neutral-or-better on decode. Two positive
passes is the standard Stage 7 applied to the 2-token tile, and the noise floor that run measured
(`AP2`'s own replication spread, 0.39% on prefill) is small enough that a +3% effect is detectable.

### 4.5 What this prototype is not

* **Not a claim.** No speed number for `C1` exists in this document and none can: the phone is away.
  The only measured facts are bit-exactness and that the path runs.
* **Not the full ik structure.** ik also packs the activation (`block_q8_0_x4`) and template-unrolls
  the token loop at compile time for 1..8 rows (`func<T>`), with `acc[nrc_y]` in registers. This
  candidate tests the token-column axis with our plain `block_q8_0` activation and a runtime-unrolled
  count of 4. If `C1` > `A0`, the next step is the packed activation *on top of a winning tile*, which
  is the combination Stage 7 never tested (it tested the pack alone, on a losing tile, and measured
  null).
* **Not wired in the dense path**, so a `C1` win understates itself slightly: prefill's expert share is
  61.9% of prefill time, and the dense `q2_0` projections keep the 2-token tile in this prototype.


---

## Tonight's run, in the order it should be typed

Everything below was validated on the host with no phone attached. `HOST_CHECK=1` is the gate that
says so.

```sh
cd /home/kingfish600/workspace

# 0. 5 seconds, no phone needed: every artifact, symbol, format string and selftest
HOST_CHECK=1 MoeDroid/tools/s8_shootout.sh 0          # expect: 19 checks, 0 failures

# 1. FIRST 20 MINUTES: the un-throttled per-op / per-type budgets for BOTH engines.
#    This is the deliverable Stage 6 and Stage 7 each failed to produce for a different reason.
MoeDroid/tools/s8_shootout.sh 1
python3 MoeDroid/tools/s4_prof.py .stage8-logs/prof-logs/s8-prof-ours.log decode 48
python3 MoeDroid/tools/s4_prof.py .stage8-logs/prof-logs/s8-prof-ik.log   decode 48
python3 MoeDroid/tools/s4_prof.py .stage8-logs/prof-logs/s8-prof-ik-nofmoe.log decode 48

# 2. identity gates (10 arms), then 3. the mirrored timing passes with the C1 candidate
MoeDroid/tools/s8_shootout.sh 2
MoeDroid/tools/s8_shootout.sh 3

# analysis: medians, within-pass deltas, the named contrasts, AND the trustworthiness columns
python3 MoeDroid/tools/s8_ll_table.py .stage8-logs
```

Read the tables in this order: **(a)** the cooldown/cache/throttle table first - if any arm shows
`LOST` on cache or a non-zero `throttled` count, its numbers are suspect and the fix is a longer
`WARM_COOL_S`, not a smaller delta; **(b)** then the within-pass deltas; **(c)** then the contrasts.
A `C1 vs A0` prefill delta that is positive in both passes is a ship candidate under the rule in
§4.4; anything else keeps `BONSAI_MOE_X4_PREFILL=1`.

## What is verified, and what still needs the phone

| claim | status |
|---|---|
| ik's profiler now measures thread-0 wall clock per op and per type, in the fork's exact dump format | **verified**: `s4_prof.py` parses it unmodified; per-op rows account for 99.86% of the node loop; 3.55x worker-sum inflation demonstrated and removed |
| the per-type rows cover every type varH uses | **verified**: `q2_0`, `q3_K`, `q4_K`, `q5_K`, `q8_0`, `iq4_xs`, `bf16` each record, `MUL_MAT` total == `MATMUL <type>` total |
| the cooldown protocol, cache-survival proof, post-cool cap gate and per-request throttle flags | **verified** on the host: shell logic unit-tested, analyzer + selftest cover every new line, 19-check host gate |
| the static comparison of the six types | **measured** from the shipped binaries + source; no speed claim |
| the C1 candidate (4 rows x 4 tokens) | **bit-exact** at kernel level (96/96 pairs) and byte-identical at graph level (6/6 shapes), and it demonstrably runs - but **its speed is not measured** |
| whether the cooldown actually restores the un-throttled state on this device | **not verifiable on the host** - that is what `S8THR`/`S8CAP gate=POSTCOOL` will answer tonight |
| tonight's arm speeds, the C1 decision, the ik-vs-ours millisecond budget | **not measured** - needs the phone |

---

## Addendum: the hard profiler clock-invariance gate

The Stage 7 §4 profiler pass produced a table that said the zero-spill kernel was 9.4% *faster* while
the same arm's wall clock said it was 5.4% *slower*. The cause was not subtle once measured: every op,
including `GET_ROWS` and `GATED_DELTA_NET` which a matmul change cannot touch, moved by 6-14%
together, i.e. the two arms ran at different clocks. That failure is now a **gate**, not a paragraph
of caveats.

### A.1 What the gate checks

`python3 MoeDroid/tools/s4_prof.py gate <log_a> <log_b> [--same-engine] [--tol=2.5]`

| pair | canary ops required to agree | ops explicitly exempt |
|---|---|---|
| **cross-engine** (`prof-ours` vs `prof-ik`) | `GET_ROWS`, `RMS_NORM`, `CPY` - stock `ggml` code in both engines | **`GATED_DELTA_NET`**: ik dispatches its own NEON `DELTA_NET` (`iqk_mul_mat.cpp:1791`) while we run the generic op, so a difference there is a real kernel delta and only interpretable once `GET_ROWS` proves the clocks matched |
| **same-engine** (`prof-ours` base vs candidate, or `prof-ik` vs `prof-ik-nofmoe`) | `GET_ROWS` **and** `GATED_DELTA_NET`/`DELTA_NET` - neither is touched by a matmul change | nothing |

Limit **2.5%**. Verdict line `S8PROF_GATE=PASS|FAIL`, exit code 0/3, and on failure the explicit
sentence `CLOCK MISMATCH - INVALID: discard this comparison and re-run the pair`.

### A.2 A flaw in my first version, and the fix

The first implementation compared canaries purely as a percentage and **failed a pair of identical
configurations**: in a small synthetic graph `GET_ROWS` costs 4 µs/token, where ±0.2 µs of timer
quantisation is 5%. A percentage is meaningless below the noise floor of the thing being timed.

So a canary must clear an **absolute cost floor** (`--min-canary-ms`, default **0.1 ms/token**) to be
used at all; below it the op is reported as `skipped < floor` and does not vote. On the device
`GET_ROWS` is ~6.6 ms/token, i.e. 66x the floor, so 2.5% there is ~165 µs - a real signal. And if
**no** canary clears the floor the verdict is **INVALID**, never PASS: an unvalidatable pair is
discarded, which is the conservative direction.

### A.3 The gate is itself tested

`s4_prof.py gate --selftest` builds five fixture pairs and asserts each outcome:

```
PASS  matched clock + matmul-only change -> PASS   [worst canary |delta| = 0.00% (limit 2.5%)]
PASS  canary drift 4% -> FAIL                      [worst canary |delta| = 4.05% (limit 2.5%)]
PASS  canary below the cost floor -> INVALID, not PASS
PASS  cross-engine: GATED_DELTA_NET difference is NOT a clock failure
PASS  cross-engine: GET_ROWS mismatch IS a clock failure
S8PROF_GATE SELFTEST PASSED
```

and validated against the real thing it exists to catch - the actual Stage 7 logs:

```
$ python3 MoeDroid/tools/s4_prof.py gate .stage7-logs/s7-prof-ours-A0.log \
                                        .stage7-logs/s7-prof-ours-A5.log --same-engine
GET_ROWS            6.6371   6.2496    -5.84%
GATED_DELTA_NET     6.2685   5.3908   -14.00%
# worst canary |delta| = 14.00% (limit 2.5%)
S8PROF_GATE=FAIL
# CLOCK MISMATCH - INVALID: discard this comparison and re-run the pair
```

That is the pair whose published reading was "A5 is 9.4% faster". The gate rejects it.

### A.4 It is wired into the shootout, with automatic re-runs

`s8_shootout.sh` phase 1 now runs the three profiler arms through the gate. On failure the pair is
**discarded and re-run after an extended cooldown** (`PROF_RETRY_COOL_S=240`, `PROF_GATE_ROUNDS=2`),
because publishing a clock-confounded budget is the error this exists to prevent:

* **pair A** (`prof-ours` vs `prof-ik`, cross-engine) - both arms re-run on retry;
* **pair B** (`prof-ik` vs `prof-ik-nofmoe`, same-engine) - only the measuring arm re-run, since pair A
  already proved `prof-ik`'s clock.

The verdicts land in `$LOGDIR/prof-gate.txt` (`pair_A(…)=PASS|FAIL`, `pair_B(…)=…`, the tolerance and
the round count), and `s8_ll_table.py` now prints them **at the top of the report** with an explicit
warning that a FAIL pair's per-op table is not publishable.

The retry loop is factored into `gated_pair()` / `gated_second()` so it can be tested without a phone;
`.stage8/test_gated_pair.sh` extracts the shipped functions verbatim and drives them with stubs:

| case | arms actually run | rc | meaning |
|---|---|---|---|
| gate passes on attempt 1 | `A B` | 0 | no retry, pair publishable |
| FAIL then PASS | `A B A B` | 0 | one automatic re-run, then publishable |
| always FAIL | `A B A B` | 1 | `S8PROF_GATE=FAIL-FINAL`, nothing publishable |
| `gated_second` FAIL then PASS | `B B` | 0 | only the measuring arm is repeated |

`HOST_CHECK=1 s8_shootout.sh 0` now runs **23 checks and all pass**, including the gate selftest, the
wiring check, the "refuses a real confounded pair" check, and the retry-loop test.

### A.5 What this does not claim

The gate proves the *clock matched*. It does not prove the numbers are right - a pair can agree on
`GET_ROWS` and still be measured badly (page-cache loss, a mis-set env var). It is one necessary
condition, added because on this project it is the one that has already failed twice (Stage 6's
worker-sum inflation, Stage 7's cross-arm clock drift), and the phase-1 cache/throttle telemetry in
Task 3 covers the other conditions.

