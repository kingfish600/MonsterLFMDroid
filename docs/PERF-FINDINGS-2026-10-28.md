# LFM performance findings — 2026-10-28

_Device: RedMagic 10s Pro (`91251291018D`), Snapdragon 8 Elite, 24 GB RAM.
Everything below is measured on-device unless marked host x86._

## The measurement protocol was wrong, and it mattered
Every benchmark this project ran before today used `adb shell` with nothing in the
foreground and bare `-t 8`. Two things were silently wrong:

1. **Core mask.** Snapdragon 8 Elite has **no little cores**: 2 prime (cpu6/cpu7, up
   to 4.47 GHz) + 6 performance (cpu0-cpu5, up to 3.53 GHz). Bare `-t 8` leaves
   placement to the scheduler. The app pins `--cpu-mask 0xff --cpu-strict 1`, and
   that alone is worth **1.65×** on Qwen3.6 (5.80 → 9.55 t/s tg128).
2. **Governor.** Observed as `powersave` at idle and `walt` under load. It is not
   deterministic from foreground state alone — later sampling showed `walt` even
   with the app force-stopped — so it must be **read and recorded with every
   measurement**, not assumed.

Correct protocol:
```sh
adb shell 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'   # record it
adb shell 'cd /data/local/tmp/lfm; LD_LIBRARY_PATH=. ./llama-bench -m <model> \
  -p 128 -n 128 -r 2 -ngl 0 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
  -t 8 -C 0xff --cpu-strict 1 -lm none'
```

## LFM2.5-24B-A2B QAD-Q4_0, re-measured
| config | pp128 | tg128 |
|---|---|---|
| mmap (`-lm auto`, the old default) | 78.88 | 30.53 ± 4.21 |
| **`-lm none` (new default)** | **96.52** | **36.63 ± 0.35** |
| bare `-t 8`, no mask (old methodology) | 14.73 | 18.44 ± **25.34** |

`-lm none` is +20% decode with 12× less variance, because the model fits resident
(12.64 GiB on ~23 GB). Now the fork's default — see commit `1e568b3`, which also
replaced the `mlockWeights` boolean (that could never work: Android caps
`RLIMIT_MEMLOCK` at 64 KB for a shell process).

**Sustained (not burst):** 4 consecutive 300-token generations held
**35.40 / 35.39 / 35.49 / 35.43 t/s** — flat, no thermal decay. This contradicts the
project's earlier note that sustained use "drops to ~11 t/s"; that note should be
treated as retired.

In-app the same config reports **27.1–27.5 t/s**. The app's real argv was read off
the device (`/proc/<pid>/cmdline`) and matches what is intended, including
`--load-mode none`. Threading is correct: 8 workers, each pinned to its own core
0-7 (`Cpus_allowed_list` per thread).

## Negative result: `--cache-ram` is innocent
The in-app/shell gap looked like the app's `--cache-ram 24576` (a 24 GB prompt
cache). An interleaved A/B — alternating arms so drift hits both equally — says no:

| arm | runs | mean |
|---|---|---|
| `--cache-ram` none | 23.53, 25.93, 20.25, 21.36 | **22.77** |
| `--cache-ram` 24576 | 22.59, 21.91, 22.71, 23.91 | **22.78** |

Identical. Worth recording because a single unpaired A/B had earlier suggested a
25% penalty — it was drift, not the flag.

## The actual cause of the 22-vs-36 t/s swing: the CPU is not boosting
Sampling `scaling_cur_freq` once per second *during* a generation:

| core | observed | max | % of max |
|---|---|---|---|
| cpu0 (performance) | 1996 MHz | 3532 MHz | 57% |
| **cpu6 (prime)** | **1017–1958 MHz** | **4473 MHz** | **23–44%** |

The prime cores — the whole point of the `0xff` mask — run at roughly a quarter to
a half of capability. Thermals are not the cause (42-54 °C, no throttling range).
Neither is memory (20.3 GB available, negligible paging) nor the page cache.

**RedMagic's per-app performance list is the likely cause:**
```
settings get global NubiaperformanceMode
  -> com.pocketpalai+300, com.prismml.moedroid+300,
```
`com.prismml.moedroid` is the **original MonsterQwenDroid**, and it carries a
**+300 boost**. None of the three forks are in that list. Related keys:
`db_game_strengthen_mode_list` (also names `com.prismml.moedroid`),
`db_game_strengthen_packagename=com.prismml.moedroid`, `game_mode_on_off=2`,
`nubia.perf.cpu.cpufreq.ctrl=1`.

**Implications, both directions:**
- The monster's 6.945 t/s was measured **with** the boost. Our forks' numbers are
  **without** it, so the comparisons are unfair to the forks — LFM's ~35 t/s is
  achieved unboosted.
- Conversely, getting the forks into that list is the single largest remaining
  speed lever, potentially recovering the 22 → 36 t/s gap seen in-app.

**Not yet done** (phone was unplugged): writing the package into
`NubiaperformanceMode` via `settings put global` (adb shell has
`WRITE_SECURE_SETTINGS`, so it may work) and re-measuring; alternatively the user
can add the app through RedMagic's Game Space UI, which is the supported path and
less likely to be overwritten.
