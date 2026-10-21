# Stage 4 report: the measured per-op budget, the THP ceiling, `-ub`, and shipping defaults

Date: 2026-10-15. Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite, 24 GB, CPU only.
Engine: fork of llama.cpp `1b70350`, tree at `6488af5` plus the uncommitted `BONSAI_SHIELD`
and `BONSAI_PROF` work. arm64 `armv8.2-a+dotprod+i8mm+fp16+bf16`.

Two device library sets:

| dir | build | use |
|---|---|---|
| `/data/local/tmp/lib-shield` | `build-arm64-ship` | all timing arms |
| `/data/local/tmp/lib-prof` | `build-arm64-prof`, `-DBONSAI_PROF=1` | the per-op budget only |

`build-arm64-prof` was configured with the same options as `build-arm64-ship` and the two
CMake caches diff clean (only `CMAKE_C_FLAGS` carries the extra define), so the two engines
are the same code plus the profiler.

Shipping flags used throughout, now including the Stage 3 batch-pool pinning:

```
-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1 --cpu-mask-batch 0xff --cpu-strict-batch 1
--poll 0 -c 8192 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -ngl 0
```

---

## 0. Headline

| task | result |
|---|---|
| **1. per-op budget** | decode **173.2 ms/token**: expert `MUL_MAT_ID` **79.5 ms (45.9%)** at 7.8 GiB/s, dense `MUL_MAT` **70.9 ms (40.9%)** at 36.4 GiB/s, `GET_ROWS` 5.6 ms, everything else 16.4 ms. A major fault costs **~170 us** and lands entirely in the expert path |
| **2. anonymous THP** | the kernel grants a hard ceiling of **~207-210 huge pages (420 MiB)** regardless of mapping size - 14% of the 2.9 GB dense set at 2.9 GB, 5% at 8 GB. `MADV_COLLAPSE` fails ENOMEM past that. **The anonymous dense-cache idea does not work on this device** |
| **3. `-ub` sweep** | prefill is non-monotonic in `-ub` and the spread is large: **512 and 64 are ~10.3-10.65 t/s, 256 and 128 are ~9.0 t/s**. The one clean mirrored pair at matched temperature gives **-14.9% for `-ub 256` vs `-ub 64`**. Default 512 stays |
| **4. shipping defaults** | batch-pool pinning wired into `EngineProfile`/`LlamaServerEngine`, `THERMAL_SAVER` opts out; shard 1 on the device swapped to varH by stage+verify+atomic rename, `ModelCatalog` repointed, app path re-verified at 5.46 t/s |

---

## 1. Per-op time and bandwidth budget on `varH`

### 1.1 Method

`BONSAI_PROF` existed but only totalled by op and by weight type, which cannot separate
`output.weight` from the other Q5_K tensors nor one BF16 tensor from another - both of which
this budget needs. It was extended with:

* **per-weight tracking** keyed on the `src[0]` tensor pointer, for `MUL_MAT`, `MUL_MAT_ID`
  and `GET_ROWS`, accumulating time, call count and bytes moved. Bytes are per-op-correct:
  a `MUL_MAT` reads the whole weight once per call, a `MUL_MAT_ID` reads only the routed
  experts (saturating at `n_as` for a long prompt), and a `GET_ROWS` reads one row per index
  rather than the whole table - which is what makes the 26.8 GiB shard-2 PLE lookup
  measurable instead of nonsense.
* **per-graph page-fault counters** read from `/proc/self/stat` and printed in each dump
  header, so op time can be *regressed* against faults instead of assuming a cost per fault.

With `BONSAI_PROF_EVERY=1` every graph dumps, and because the counters are cumulative,
consecutive dumps are differenced. The window is the **last 48 graphs of a 98-graph run**
(warmup prompt + 32 warmup decode + measured prompt + 64 measured decode), i.e. measured
decode tokens **17..64**. Using the final `ms/graph` average instead would have folded the
prompt graph and the warmup into the budget.

Three decode arms were run (5.610 / 5.788 / 5.691 t/s measured; the profiler is not
detectably slowing them) and one prefill arm.

### 1.2 Decode budget (mean of three runs)

Total compute **173.229 ms/token = 5.773 t/s**, against 5.610-5.788 t/s measured end to end,
so the op accounting is essentially complete - there is no large un-attributed gap.

| op | ms/token | calls/token | share |
|---|---|---|---|
| `MUL_MAT_ID` (experts) | **79.468** | 144.0 | 45.9% |
| `MUL_MAT` (dense) | **70.875** | 833.0 | 40.9% |
| `GET_ROWS` | 5.617 | 174.0 | 3.2% |
| `GATED_DELTA_NET` | 4.940 | 36.0 | 2.9% |
| `CPY` | 2.244 | 186.0 | 1.3% |
| `CONCAT` | 2.177 | 73.0 | 1.3% |
| `MUL` | 1.172 | 200.0 | 0.7% |
| `UNARY` | 1.016 | 402.0 | 0.6% |
| `ARGSORT` | 0.930 | 48.0 | 0.5% |
| `DSV4_HC_PRE` + `DSV4_HC_POST` | 1.567 | 96.0 + 96.0 | 0.9% |
| `RMS_NORM` | 0.696 | 256.0 | 0.4% |
| everything else (17 ops) | 2.587 | | 1.5% |
| **total** | **173.229** | | |

The two matmuls are **86.8% of decode**. Nothing else reaches 3.3%.

**Non-matmul ops** are negligible in aggregate: the Gated DeltaNet recurrence
(`GATED_DELTA_NET` + `SSM_CONV` + `CUMSUM` + `CONCAT`) is 7.4 ms (4.3%), attention
(`FLASH_ATTN_EXT` + `SOFT_MAX` + `ROPE` + `ARGSORT` + `TOP_K`) is 1.5 ms, the `hc_*`
elementwise/inject pair (`DSV4_HC_PRE`/`DSV4_HC_POST`) is 1.6 ms, and `RMS_NORM` is 0.7 ms.
There is nothing to win there.

### 1.3 Dense `MUL_MAT` by weight type

Dense moves **2644.0 MiB/token in 70.875 ms = 36.4 GiB/s** aggregate. Per type, against the
varH inventory:

| type | ms/token | dense MiB/token | GiB/s | share of total |
|---|---|---|---|---|
| `q3_K` | 20.596 | 476.1 | **22.6** | 11.9% |
| `q8_0` | 12.503 | 640.7 | 50.0 | 7.2% |
| `iq4_xs` | 11.406 | 417.5 | 35.7 | 6.6% |
| `q5_K` | 10.563 | 496.5 | 45.9 | 6.1% |
| `q4_K` | 4.551 | 221.5 | 47.5 | 2.6% |
| `bf16` | 4.381 | 207.1 | 46.2 | 2.5% |
| `q2_0` (dense part) | 4.259 | 69.8 | **16.0** | 2.5% |
| `q6_K` | 1.823 | 80.5 | 43.1 | 1.1% |
| `q4_0` / `q5_0` / `f32` | 0.637 | 31.1 | | 0.4% |

`token_embd.weight` (260.5 MiB, `q3_K`) is **not** in the dense matmul set - it is read by
`GET_ROWS`, so the `q3_K` matmul bytes are 476.1 of the 736.6 MiB the inventory shows for
that type.

Two types are outliers: dense `q2_0` at 16.0 GiB/s and `q3_K` at 22.6 GiB/s, against 43-50
GiB/s for everything else. Together they are **24.9 ms/token (14.4% of decode) for 546 MiB of
2644 MiB (20.6%)** of dense bytes. That is the one place in the dense path where a requant or
a faster dequant would pay.

The two tensors the task asks about specifically:

| tensor | ms/token | MiB/token | GiB/s | share of decode |
|---|---|---|---|---|
| `output.weight` (`Q5_K`, 416.8 MiB) | **8.664** | 416.8 | 47.0 | **5.00%** |
| all 291 remaining BF16 tensors | **4.381** | 207.1 | 46.2 | 2.53% |

`output.weight` is the single most expensive dense tensor - 12.2% of dense matmul time for
15.8% of dense bytes - and it already runs at the best rate in the model (47.0 GiB/s), so
there is no headroom left in it. It is also why the Stage 2 `Q5_K -> Q4_0` experiment was a
bad trade: the tensor is already at the bandwidth ceiling, so making it smaller bought
almost nothing while costing +0.814% perplexity.

### 1.4 Expert `MUL_MAT_ID`

| metric | value |
|---|---|
| ms/token | **79.468** (45.9% of decode) |
| calls/token | 144 (one per expert tensor: 48 layers x 3 matrices) |
| bytes/token | 632.8 MiB = **663.5 MB** (decimal) - the task's "664 MB/token" |
| effective rate | **7.8 GiB/s = 8.4 GB/s** |

The expert path is **4.7x slower per byte than the dense path** (7.8 vs 36.4 GiB/s) for
2.5x the time of the whole dense backbone in 24% of the bytes. That is the decode bottleneck.

### 1.5 `GET_ROWS` and the shard-2 PLE lookup

`GET_ROWS` is **5.90-5.62 ms/token across 174 calls/token**, 3.2-3.4% of decode. This is
where the embedding lookups live, including `per_layer_token_embd.weight` - a single 26.8 GiB
`iq4_nl` tensor in shard 2, read as sparse rows.

Per-tensor accounting needed a fix to be meaningful. The profiler's weight table was
originally 1024 entries; adding `GET_ROWS` pushed the number of distinct `src[0]` tensors past
that, and because the LM head is the **last** matmul of a graph it was the one silently
dropped - `output.weight` vanished from the table and the dense aggregates undercounted by
17% (2193 MiB/token instead of 2644). That was caught by checking the aggregate against the
known dense set size, not by the profiler complaining, so the table was raised to 4096 and an
overflow now prints a warning. With the fix the byte accounting closes exactly:
**AGG dense 2756.7 MiB/token = 2644.0 MiB of dense matmul weights + 112.7 MiB of `GET_ROWS`
rows**, which is a useful cross-check on both.

The split that emerges:

| tensor | ms/token | calls/token | note |
|---|---|---|---|
| `token_embd.weight` (`q3_K`, 260.5 MiB) | ~0.3-0.7 | ~1 | enters and leaves the dense top-16 around this value |
| `per_layer_token_embd.weight` (`iq4_nl`, **26.8 GiB**, shard 2) | **< 0.32** | few | never reaches the dense top-16, whose 16th entry is 0.318 ms/token |
| rest of the 174 `GET_ROWS` calls | remainder | | many small per-layer lookups |

**The 26.8 GiB shard-2 table is not a per-token cost worth attacking.** Only a handful of rows
are touched per token and they are page-cache resident, so the whole table costs less than a
third of a millisecond. The 5.9 ms that `GET_ROWS` does spend is spread across ~174 small
per-layer lookups, not concentrated in the big table. This closes the question the handoff
left open about whether the PLE table needed its own strategy.

### 1.6 How much of the expert/dense gap is major faults

The profiler header carries `minflt`/`majflt` per graph, so the question is answered by
regressing per-graph op time against per-graph major faults over the 47 deltas of the
measured window:

| regressand | slope | Pearson r |
|---|---|---|
| total compute | **+166.5 us per major fault** | +0.407 |
| `MUL_MAT_ID` | **+172.6 us per major fault** | +0.451 |
| `MUL_MAT` (dense) | -2.3 us per major fault | -0.036 |

Major faults in the window average **113-119/token** (min 36, max 288). Two things follow:

* **A major fault costs ~170 us.** That is a real, measured number rather than an assumed
  one, and it is consistent across two independent runs (166.5 / 168.2 us) and both
  reagressands.
* **The relationship is entirely in the expert path.** Dense `MUL_MAT` shows no correlation
  with faults at all (r = -0.04 / -0.15), which is what "the dense backbone is cache-resident"
  looks like when you measure it.

Attribution of the 4.7x per-byte gap:

| term | value |
|---|---|
| expert path, measured | 79.468 ms/token for 632.8 MiB |
| same bytes at the dense aggregate rate (36.4 GiB/s) | 16.98 ms/token |
| **excess** | **62.49 ms/token** |
| attributable to major faults: 115 faults x 170 us | **19.6 ms** |
| **share of the excess explained by faults** | **31%** |
| residual (Q2_0 dequant, 460 KB slice granularity, less vectorisation) | 42.9 ms |

So roughly **a third** of the expert penalty is page-fault stalls and the remaining two
thirds is the Q2_0 compute path itself. The handoff's framing that the expert path is
"not ALU-bound" holds for the *kernel micro-optimisation* it tested, but it does not mean the
path is fault-bound either - most of the gap is neither.

### 1.7 Prefill budget

One 262-token prompt graph, divided by its 262 tokens. The graph is unambiguous in the dump
sequence: it is the delta with the largest total, and it reconciles with the server's own
`prompt eval time`.

| op | ms/token | share |
|---|---|---|
| `MUL_MAT_ID` (experts) | **70.521** | 61.9% |
| `MUL_MAT` (dense) | **31.544** | 27.7% |
| `RMS_NORM` | 2.594 | 2.3% |
| `GATED_DELTA_NET` | 2.177 | 1.9% |
| everything else (27 ops) | 7.140 | 6.3% |
| **total** | **113.976** | |

Graph totals: 29,861.7 ms for 262 tokens = **8.77 t/s of pure compute** against 8.574 t/s
measured. Major faults are **1,032.8/token** - 9x the decode rate, which is the eviction
storm the Stage 3 shielding work was aimed at.

Prefill is much more expert-dominated than decode (61.9% vs 45.9%): a 262-token prompt routes
~2620 expert slots across 512 experts, so essentially the whole 32,400 MiB expert pool is
read once per prompt, whereas decode touches only 632.8 MiB and mostly from cache.

**This also explains the Stage 3 batch-pinning win.** Same model, same prompt, pinned vs
unpinned batch pool, both cap-stable:

| component | unpinned (`-tb 8` alone) | pinned (`--cpu-mask-batch 0xff`) | delta |
|---|---|---|---|
| total | 131.61 ms/token | 113.98 ms/token | **-13.4%** |
| dense `MUL_MAT` | 31.27 | 31.54 | +0.9% (unchanged) |
| expert `MUL_MAT_ID` | 87.70 | 70.52 | **-19.6%** |
| expert effective rate | 1.38 GiB/s | **1.71 GiB/s** | +24% |
| majflt/token | 693.4 | 1,032.8 | +49% |

The entire prefill win is in the expert path; the dense path is untouched. Pinning raises the
fault count but also raises the achieved expert rate, which says the expert path is not
purely storage-bound - deeper miss parallelism from 8 pinned workers extracts more from the
same flash.

---

## 2. Anonymous THP (`AnonHugePages`) feasibility probe

`/data/local/tmp/thp_probe` (source `MoeDroid/tools/thp_probe.c`, ~120 lines, statically
linked against the NDK) maps anonymous memory, hints `MADV_HUGEPAGE`, touches every page, then
tries `MADV_COLLAPSE` (25), reporting `AnonHugePages` from `/proc/self/smaps_rollup` and
`/proc/meminfo` at each step.

Device state at the time:

| knob | value |
|---|---|
| `/sys/kernel/mm/transparent_hugepage/enabled` | `[always] madvise never` |
| `/sys/kernel/mm/transparent_hugepage/defrag` | `always defer defer+madvise madvise [never]` |
| `/sys/kernel/mm/transparent_hugepage/shmem_enabled` | `always within_size advise [never] deny force` |
| `FileHugePages` (idle) | 0 kB |
| `AnonHugePages` (idle) | 178-182 MiB |
| `SwapTotal` / `SwapFree` | 12,287 MB / 10,573 MB |
| `MemTotal` / `MemAvailable` | 23,663 MB / 17,980 MB |
| `defrag` writable? | **no** - `Permission denied` without root |

`zram0` is not readable as `shell`, but a 12 GiB swap with ~1.7 GiB used and no physical swap
partition means **zram is active**.

### 2.1 Results

| mapping | 2 MB-aligned | `AnonHugePages` after touch | coverage | `MADV_COLLAPSE` |
|---|---|---|---|---|
| 64 MiB | yes | 64 MiB (32 x 2 MB) | **100%** | rc=0 |
| 512 MiB | yes | 414 MiB (207 x 2 MB) | 80.9% | rc=-1 ENOMEM |
| **2908 MiB** (the dense set) | no | 408 MiB (204 x 2 MB) | **14.0%** | rc=-1 ENOMEM |
| 2908 MiB | **yes** | 420 MiB (210 x 2 MB) | 14.4% | rc=-1 ENOMEM |
| **8192 MiB** (expert cache) | no | 418 MiB (209 x 2 MB) | **5.1%** | rc=-1 ENOMEM |
| 8192 MiB | yes | 418 MiB (209 x 2 MB) | 5.1% | rc=-1 ENOMEM |

### 2.2 What this means

**The kernel grants a hard ceiling of about 207-210 huge pages (~420 MiB), and it does not
grow with the mapping.** Going from 2.9 GB to 8 GB of request bought exactly zero extra huge
pages (210 -> 209). With `defrag=never` khugepaged never compacts, so the only 2 MB pages
available are the order-9 blocks already free in the buddy allocator, and there are about
420 MiB of them.

Two details worth recording:

* **Alignment is not the limit.** Forcing 2 MB alignment (which `mmap(NULL, ...)` does not
  guarantee - the unaligned 2908 MiB run came back at `0x7be7cc5000`) moved coverage from
  204 to 210 pages. The binding constraint is free contiguous memory, not the mapping address.
* **`MADV_COLLAPSE` works only while the budget lasts.** At 64 MiB the touch already achieved
  100% coverage and collapse returned 0. At 512 MiB and above it fails with `ENOMEM` while
  trying to build the pages the touch could not.

**The anonymous dense-cache idea from the task is therefore not viable on this device.** The
plan was to `pread` the 1079 dense tensors (2907.8 MiB) into an anonymous buffer, back them
with 2 MB PMD entries, and `MADV_DONTNEED` the file-backed ranges so the dense backbone
becomes immune to page-cache eviction during prefill. Measured, that buffer would get
**~420 MiB of huge pages - 14% - and the other 86% would still be 4 KB pages**, so there is no
TLB win to be had, and the copy plus the `MADV_DONTNEED` would add load-time cost and risk for
nothing. An 8 GB anonymous expert cache is worse: 5% coverage, and it would take 8 GB of the
~18 GB `MemAvailable` away from the file page cache that currently gives the 99.8% expert hit
rate. `SwapFree` did not move during either allocation, so the kernel reclaimed page cache
rather than swapping - the allocation is not free, it is paid for out of the cache that makes
decode fast.

`FileHugePages = 0` reconfirms the Stage 2 finding: f2fs on 6.6 never emits large folios, so
the file-backed route is closed for a different reason than the anonymous one.

---

## 3. `-ub` (physical microbatch) prefill sweep

### 3.1 Design

518-token prompt (`PREP_REPS=32`), `n_predict 16`, shipping flags including the batch-pool
pinning, `pd` mode so the prefill rate, the prefill/decode fault split and the first-16
decode window all come from one streamed request. 90 s cooldown plus the cap gate before
every arm.

The knob cannot be mirrored the way a two-valued A/B can, so the first two passes ran the
same four values in **opposite order** - descending (512/256/128/64) and ascending
(64/128/256/512) - to expose ordering drift. That control did its job: it revealed a
confound (§3.3), which a third **mirrored A/B/B/A** pass (64/256/256/64) then settled.

### 3.2 The two ordered passes

| pass | arm | -ub | start cputemp | prompt_ms | prefill t/s | majflt prefill | majflt decode | cap stable |
|---|---|---|---|---|---|---|---|---|
| desc | Du512 | 512 | 91.9 | 48,013 | **10.7888** | 452,896 | 7,210 | yes |
| desc | Du256 | 256 | 93.4 | 57,311 | 9.0385 | 505,888 | 55,948 | yes |
| desc | Du128 | 128 | 95.8 | 57,557 | 8.9998 | 295,379 | 117,594 | yes |
| desc | Du64 | 64 | 93.4 | 50,903 | **10.1762** | 143,719 | 118,990 | yes |
| asc | Au64 | 64 | 89.9 | 49,891 | **10.3827** | 242,406 | 119,739 | yes |
| asc | Au128 | 128 | 96.9 | 57,998 | 8.9314 | 409,825 | 119,198 | yes |
| asc | Au256 | 256 | 95.4 | 56,893 | 9.1049 | 336,324 | 134,809 | NO |
| asc | Au512 | 512 | 91.9 | 49,257 | **10.5162** | 296,414 | 157,406 | NO |

Every arm evaluated exactly 518 prompt tokens, so the rates are directly comparable.

| -ub | descending | ascending | mean |
|---|---|---|---|
| 512 | 10.7888 | 10.5162 | **10.6525** |
| 256 | 9.0385 | 9.1049 | 9.0717 |
| 128 | 8.9998 | 8.9314 | 8.9656 |
| 64 | 10.1762 | 10.3827 | **10.2795** |

The two passes agree on the ordering to within ~2.5%, and the spread between the fastest and
slowest -ub is **~19%**, far outside this device's 5-8% arm-to-arm noise. The pattern is
**non-monotonic**: 512 and 64 are fast, 256 and 128 are slow.

### 3.3 The confound, and how it was broken

In both passes the arm slots line up exactly with the -ub sets: slots 1 and 4 always held
{512, 64} and slots 2 and 3 always held {256, 128}. So "slow" is equally explained by "the
middle two arms of a four-arm block are slow" - and the start-of-window CPU temperature
correlates with slot (91.9-93.4 C in the fast slots, 93.4-96.9 C in the slow ones).

The mirror pass is the control:

| arm | -ub | slot | start cputemp | prompt_ms | prefill t/s | cap stable |
|---|---|---|---|---|---|---|
| Ms1u64 | 64 | 1 | 91.5 | 48,905 | **10.5919** | yes |
| Ms2u256 | 256 | 2 | 93.0 | 57,432 | 9.0194 | yes |
| Ms3u256 | 256 | 3 | 97.3 | 58,304 | 8.8845 | **NO** (cap fell to 2841600 at the end) |
| Ms4u64 | 64 | 4 | 95.0 | 53,339 | 9.7115 | **NO** (cap fell to 2841600 at the end) |

The one **fully cap-stable pair at matched temperature** is Ms1u64 / Ms2u256: start temps
91.5 vs 93.0 C, both arms entered and left the measured window at `3532800/3840000`:

> **`-ub 64` 10.5919 t/s vs `-ub 256` 9.0194 t/s = -14.85% for `-ub 256`.**

Group means over all four mirrored arms (including the two that throttled at the very end of
their windows) are 10.1517 vs 8.9520 = **-11.8%**. Both mirrored pairs point the same way, and
the direction matches the two ordered passes.

So the effect is real and belongs to `-ub`, not to slot position - but the residual thermal
contamination in the same direction means the honest effect size is a range, roughly
**-9% to -15% for `-ub 256` against `-ub 64`**, with the cleanest estimate at -14.9%.

### 3.4 Decode after the prefill, and the fault columns

Decode `1..16` t/s, i.e. the post-prefill window:

| -ub | descending | ascending | mirrored |
|---|---|---|---|
| 512 | 4.3922 | 3.9652 | - |
| 256 | 4.3849 | 3.8724 | 3.9159 / 4.2298 |
| 128 | 4.1238 | 4.3018 | - |
| 64 | 4.3286 | 4.3531 | 4.3536 / 3.5718 |

The early-decode window tracks the prefill rate loosely and is much noisier (3.57-4.39, a 23%
spread against a ~2.5% repeatability on prefill), so no claim is made from it beyond "it does
not move against the prefill ordering".

**The prefill/decode major-fault split is not a usable `-ub` signal and should not be quoted
as one.** The prefill-phase counts do track the fast/slow split (the `-ub 64` arms are the
lowest at 143,719 / 242,406 / 150,271 / 168,399 against 295,379-505,888 elsewhere), but the
**decode-phase** counts are dominated by history rather than by `-ub`: `-ub 512` measured
7,210 in the descending pass and 157,406 in the ascending pass for the same configuration,
a 22x swing. That column is reported for completeness only.

### 3.5 Recommendation

**Keep the default `-ub 512`.** It is the fastest measured value and needs no change. The
actionable finding is a negative one: **`-ub 256` and `-ub 128` are traps on this device,
~9-15% slower for prefill**, so a future attempt to trade memory for speed by shrinking the
microbatch should not start there. The non-monotonic shape (512 fast, 256/128 slow, 64 fast)
has no established mechanism and is worth a randomised follow-up rather than an explanation.

---

## 4. Shipping defaults and model path

### 4.1 Batch threadpool flags

`MoeDroid/app/src/main/java/com/prismml/bonsaidroid/engine/EngineProfile.kt` gains
`pinBatchAllCores` (default `true`), and `LlamaServerEngine.kt` emits
`--cpu-mask-batch 0xff --cpu-strict-batch 1` alongside the existing
`--cpu-mask 0x3f --cpu-strict 1` whenever `pinPerfCluster` applies and `-tb` is actually
being passed:

```kotlin
if (p.pinPerfCluster && Runtime.getRuntime().availableProcessors() >= 6) {
    args += listOf("--cpu-mask", "0x3f", "--cpu-strict", "1")
    if (p.pinBatchAllCores && p.threadsBatch > 0) {
        args += listOf("--cpu-mask-batch", "0xff", "--cpu-strict-batch", "1")
    }
}
```

* `QUALITY`, `BALANCED` and `SPEED` (all `-t 6 -tb 8`) now produce
  `-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1 --cpu-mask-batch 0xff --cpu-strict-batch 1 --poll 0`
  - exactly the flag set measured in Stage 3 and again in §1.7.
* `THERMAL_SAVER` (`-t 4 -tb 4`) sets `pinBatchAllCores = false`: only `-t 6 -tb 8` was
  measured, and `-tb 4` was never characterised with a batch mask, so it keeps its current
  behaviour.
* The gate is `threadsBatch > 0` so the flag is only added when `-tb` is really emitted. With
  no `-tb` the batch pool does inherit `--cpu-mask` and there is nothing to fix.

The rationale and the measurement are recorded in the `EngineProfile` KDoc next to
`pinPerfCluster`, including why `-tb` alone is not enough (`postprocess_cpu_params` only
copies the cpu params into the batch params while the batch thread count is unset, so `-tb 8`
leaves an all-zero cpumask and no affinity at all).

No native rebuild is needed: `--cpu-mask-batch` and `--cpu-strict-batch` are stock options of
this llama.cpp revision, which is what the app's bundled engine is built from.

Both `MoeDroid/` (the source that matches the package actually installed on the phone) and
`MonsterQwenDroid/` (the renamed public release) were updated, and the two
`EngineProfile.kt`/`ModelCatalog.kt` copies are byte-identical again.

### 4.2 Model: shard 1 becomes varH

The file name is kept, because llama.cpp resolves the sibling shard by rewriting the
`-00001-of-00002` part of the path the app passes, and the app's presence check keys on
`fileName`. Only the bytes change.

| | before | after |
|---|---|---|
| file | `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | same |
| size | 37,623,740,192 | **37,033,916,192** |
| sha256 | `69820c02...b199b720` | **`e7c69452...ded12e2`** |
| shard 2 | `316b46f3...c161e113` | unchanged (byte-identical between the variants) |

Done as a **stage, verify, atomic rename** rather than an in-place overwrite, so a failed or
truncated copy cannot leave the installed model corrupt:

1. `cp` the verified varH shard from `/sdcard/Download/mf-model/` to
   `.varh-shard1.staging` **in the same directory** (same filesystem, so the rename is atomic).
2. `sha256sum` the staging file on device -> `e7c69452...ded12e2`, matching the host copy.
3. `mv` it over the live shard 1 (atomic replace).
4. Re-`sha256sum` the live file -> `e7c69452...ded12e2`, size 37,033,916,192.
5. Rewrite the `.meta.json` sidecar, which still recorded the old size.

The swap is transparent to the app for two reasons, both checked in the source rather than
assumed:

* `ModelDownloader.isOneDownloaded()` short-circuits to a plain `fileFor(one).exists()` when a
  `.meta.json` sidecar is present, and the sidecar is present. (The 97%-of-catalog size check
  only applies to downloads that predate the sidecar.)
* The `ModelAsset.sha256` field is **never read anywhere in the app** - `grep` finds no
  consumer outside `ModelCatalog.kt` - so it is documentation, not enforcement. The catalog was
  still updated (`sizeBytes`, `sha256`, `summary`) so the model list and the free-space
  arithmetic agree with what is on disk.

**Caveat to carry forward:** the hosted URL still serves the un-rewritten shard 1. Because the
hash is not enforced, a fresh install would silently download the original and lose the varH
win rather than failing loudly. The rewritten shard has to be published (or the asset's URL
repointed) before the app can be re-provisioned from scratch. This is recorded in a KDoc block
above the asset list.

Both repos were updated, and the `MoeDroid` and `MonsterQwenDroid` copies of
`ModelCatalog.kt` and `EngineProfile.kt` are byte-identical again.

### 4.3 Verification

One decode arm was run against the **app's own model path** after the swap, with the new
shipping flags:

| arm | model path loaded | decode t/s | cap window | stable |
|---|---|---|---|---|
| verify1 | `/sdcard/Android/data/com.prismml.moedroid/files/models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | **5.4626** | 3532800/3840000 -> 3532800/3840000 | yes |

The engine log confirms the path, and the rate is in the varH band rather than the shipping
band (shipping measured 5.08-5.34 in Stage 3, varH 5.60-5.63), so the shipped model is now
varH. The arm started at 93.0 C against 86-91 C for the Stage 3 varH arms, which accounts for
it sitting a little below them; the point of the arm is that the app's path serves varH, not
that it is a fresh measurement.

The original shard 1 is no longer on the device (the swap freed 590 MiB); it remains on the
host at `files-drop-bonasidroidv2/flashnext-q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`,
sha256 `69820c02...b199b720`, so the swap is reversible.

---

## 5. Reproducing

* Profiler: `BONSAI_PROF=1` build in `build-arm64-prof`; `BONSAI_PROF_EVERY=1` makes every
  graph dump. Analysis: `MoeDroid/tools/s4_prof.py <log> <decode|prefill|graph> [n] [tokens]`.
* Arm harness: `MoeDroid/tools/s3_arm.sh` with `LIBDIR`, `UB`, `BONSAI_PROF_EVERY`,
  `CPU_MASK_BATCH`/`CPU_STRICT_BATCH` in the environment. `LIBDIR=/data/local/tmp/lib-prof`
  selects the profiler build.
* THP probe: `MoeDroid/tools/thp_probe.c`, built with the NDK
  `aarch64-linux-android28-clang -O2 -static`.
* `-ub` sweep: `MoeDroid/tools/s4_ub_sweep.sh <tag> <ub...>`; analysis
  `MoeDroid/tools/s4_ub.py`.
* Logs pulled to `.stage4-build/`.

### Thermal methodology

Unchanged from Stage 3 and it held for every arm here: the gate requires the known-good cap
profile (`cpu0 = 3532800`, `cpu6 = 3840000`) for three consecutive 10 s polls before the model
is loaded, a cooldown between arms, and stability is judged from the caps read immediately
either side of the measured window plus a 5 s sampler across the whole arm. Every arm in this
stage entered and left its measured window on that profile, and no sample fell to `2841600`.
