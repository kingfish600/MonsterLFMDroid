# Goal: make LFM2.5-24B-A2B the fastest and most useful on-device assistant on this phone

Standing objective: run **LFM2.5-24B-A2B QAD-Q4_0** on a RedMagic 10S Pro at the highest
throughput the hardware allows **without dropping below Q4** (owner constraint), and make it
genuinely useful as an assistant — web search, tool calling, and speech — all fully offline.

Format is the [goal-md](https://github.com/jmilinovich/goal-md) pattern. Every number here is a
measured result; detail and provenance live in `HANDOFF.md`.

> **This file previously contained MonsterQwenDroid's Qwen3.8-Flash-Next (varH) goal verbatim.**
> That is a different project with a different model and a CPU-only constraint. If you are
> looking for the Qwen varH work, it is `../MonsterQwenDroid/GOAL.md`.

Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite (SM8750), Adreno 830, 24 GB, 1 TB UFS.
Engine: stock llama.cpp `8642541`, arm64, NDK r28, `-O3 -march=armv8.2-a+dotprod+i8mm+fp16+bf16`,
`-DGGML_OPENMP=OFF`, and since 2026-10-29 `-DGGML_OPENCL=ON` (Adreno kernels).

---

## Fitness function

Throughput in **t/s**, always quoted with its regime. The device is required; there is no
host-side substitute.

```bash
# PRIMARY - sustained decode on the 24B, app path, perf mode ON:
#   5 x 300-token greedy requests against the app's own server; report the run-to-run spread.
#   Regime MUST be recorded (governor, GameSpace, free RAM/swap) - GameSpace alone is +49%.

# SECONDARY - controlled A/B, headless, llama-bench, >=3 reps:
llama-bench -m <model> -p <pp> -n <tg> -t 6 --cpu-mask 0xff --cpu-strict 1 -ngl <0|99> -r 3
```

Baselines to beat, all measured, perf mode ON, `--cpu-mask 0xff --cpu-strict 1`:

| | decode | prefill |
|---|---|---|
| LFM2.5-24B-A2B QAD-Q4_0 (CPU) | **35.8 t/s** | 106.7 t/s |
| LFM3-1.7B QAD-Q4_0 (CPU) | 43.5 t/s | 228.8 t/s |
| 24B, Adreno OpenCL (`-ngl 99`) | 20.75 (tg64) | **76.86** (pp256) |
| LFM3-1.7B, Adreno OpenCL | 28.97 (tg128) | **330.72** (pp512) |
| parent app (Qwen 176B) | 6.945 t/s | — |

Note the two protocols are not interchangeable: the app-path 35.8 t/s is a short-context
figure, the llama-bench 20.75 is tg64 after a 256-token prompt. Do not compare them.

---

## Improvement loop

1. Take the device lock (`/mnt/nvme4tb/shared-test/device.sh lock MonsterLFMDroid "..."`).
2. Record the regime before and after. A run whose regime is unknown is discarded.
3. Change one thing. Measure ≥3 reps. Report mean ± stddev with the exact command line.
4. If it wins, keep it and record the number here. If it loses, add it to the do-not-retry
   table in `HANDOFF.md` §6 with the number that killed it.
5. Clean up device processes, release the lock.

---

## Action catalog

Ordered by expected value. Everything below is either untried or was left unfinished.

1. **Verify GPU offload through the app UI.** The engine path is proven with the app's exact
   arguments; the app's own Load button has not been exercised since the swap. Highest
   priority — it is the difference between a shipped feature and a tested one.
2. **Tune the prefill/offload split.** `-ngl 99` is all-or-nothing. The Adreno shares LPDDR5X,
   so a partial offload (attention on GPU, some layers on CPU) might beat both extremes. The
   GPU wins big on compute-bound prefill and barely on bandwidth-bound decode; aim there.
3. **Decide whether offload is user-toggleable.** It costs a 15 MB lib and helps prefill 2.5×
   but decode only 1.2×. A switch in Settings may be the honest answer.
4. **Context-length decode curve.** Decode falls ~31 t/s @500 tok → ~25 @4K → ~15–17 @8K on the
   24B. Chart it properly; it is the real user-visible cost of long chats and the agent loop.
5. **Agent loop cost.** Each tool round re-processes the conversation. With prefill 2.5× faster
   this should be re-measured end to end, not per-token.
6. **Re-test LFM3-VL vision when upstream moves.** Blocked on `lfm3.cpp`/mtmd, not on us.

---

## Operating mode

- The parent repo `../MonsterQwenDroid` is **read-only by owner instruction**. Fork, never edit.
- Prefer measured numbers over plausible ones. Two conclusions in this project's history were
  wrong because a single bad data point was generalised (the "DDG blocks bots" verdict, and
  the "load-mode causes OOM" verdict). Both were corrected by re-testing the actual variable.
- Record rejected ideas with the number that rejected them. The do-not-retry table has already
  saved a repeat of the `--cache-ram` and `-ncmoe` mistakes.

---

## Constraints

- **Quant floor: Q4.** The owner will not accept lower. `QAD-Q4_0` is the shipped quant.
- **Fully offline.** Web search runs on-device (DuckDuckGo directly from the phone). Nothing
  routes through the host. An earlier SearXNG-over-Tailscale workaround was reverted for this
  reason — it worked, but searches left the device.
- **Leave the phone usable.** Do not leave stray servers holding swap; clean up after testing.
- **One phone, two projects.** MonsterQwenDroid shares this device. Take the lock.
