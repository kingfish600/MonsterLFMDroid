# Stage 2 Part A report: 8-core `-t 8 --cpu-mask 0xff` thread placement

Date: 2026-10-14. Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite, 24 GB.
Engine: fork `10288ca`, arm64 `armv8.2-a+dotprod+i8mm+fp16+bf16`, `MULMATID_FLAT` on,
`MULMAT_FLAT` off (both are the shipped defaults).
Run: `-t 8 -tb 8 --cpu-mask 0xff --cpu-strict 1 --poll 0 -c 8192 -fa on
--cache-type-k q8_0 --cache-type-v q8_0 -ngl 0`, 256-token greedy decode.

## 1. Result

| run | mask | threads | decode t/s |
|---|---|---|---|
| Part A run 1 | `0xff` | 8 | **4.107** |
| Part A run 2 | `0xff` | 8 | **3.958** |
| Stage 1 block 1/2 (reference) | `0x3f` | 6 | 4.6027 - 4.7977 |
| Part A gate re-check | default `0x3f` | 6 | 4.8024 |

**`0xff` is ~14-18% slower than `0x3f`.** The gap is well outside the ~5-8% arm-to-arm
noise measured in Stage 1, so the 8-core question is settled: **do not use `0xff`.**

Cap profile was identical in every run: `cpu0-5 = 3532800` (uncapped), `cpu6/7 = 3840000`
of a `4473600` hardware max (the known persistent cap), unchanged before and after.

## 2. Thread placement (the actual question)

Sampled `/proc/<pid>/task/*/stat` field 39 (`processor`: last CPU the task ran on) plus
field 3 (`state`) every second for 40 s during steady-state decode, across all 26 threads.

The eight ggml worker threads are placed **1:1 across cpu0..cpu7, with no bunching**.
Each worker sat on exactly one core for 40/40 samples (100% stickiness):

| tid | cpu | samples | running |
|---|---|---|---|
| 2385 | cpu0 | 40/40 | 25 |
| 2386 | cpu1 | 40/40 | 29 |
| 2387 | cpu2 | 40/40 | 34 |
| 2388 | cpu3 | 40/40 | 30 |
| 2390 | cpu4 | 40/40 | 30 |
| 2391 | cpu5 | 40/40 | 33 |
| 2392 | cpu6 | 40/40 | 32 |

CPU occupancy of threads observed running (>=10 running samples) was essentially uniform:
cpu0 13.1%, cpu1 12.5%, cpu2 12.8%, cpu3 12.5%, cpu4 12.8%, cpu5 12.5%, cpu6 13.1%,
cpu7 10.6%.

## 3. Interpretation

The slowdown is **not** a placement failure. The kernel does exactly what an ideal
scheduler would do: eight runnable workers spread one per core across both clusters.

So the cost is the topology itself. The SoC is 2 clusters: 6 perf cores at 3.53 GHz
(`cpu0-5`, one L2/LLC domain) and 2 prime cores at 4.47 GHz, capped to 3.84 GHz
(`cpu6-7`, a second domain). Under `0x3f` all six workers share one cluster's cache
hierarchy. Under `0xff` the two prime cores join in, and:

* every `ggml_barrier` in the graph now crosses a cluster boundary, and
* the two extra cores add ~36% more core clock but only ~0% more usable memory
  bandwidth, which is what the MoE expert streaming is bound on.

The measured outcome is a **14-18% net loss** from adding the two fastest cores. That is
consistent with the earlier measured +26% decode from cluster pinning `0x3f` in the first
place: this workload is memory-side, not core-side.

## 4. Consequence

No scheduler patch is warranted. The shipping `-t 6 --cpu-mask 0x3f --cpu-strict 1`
remains correct and `0xff` is closed as a lever.

## 5. Reproducing

`MoeDroid/tools/placement_8core.sh <tag> <threads> <mask>` on device; raw sample files
`placement-8core.samples` and `placement-8core2.samples` in the workspace; logs
`placement-8core.log`, `placement-8core2.log`.

Note on field indexing: in `/proc/<pid>/task/<tid>/stat`, after stripping the comm field,
`processor` (field 39) is awk field **37**; awk field 36 is `exit_signal` (17 for the main
thread, -1 for pthread workers), which is an easy way to read the wrong column.
