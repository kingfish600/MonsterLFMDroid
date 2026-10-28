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

## SOLVED: the clock boost is the whole story (and it is a 49% win)
The prime cores were running at a quarter speed because **RedMagic only boosts
apps listed in `NubiaperformanceMode`**. The original MonsterQwenDroid
(`com.prismml.moedroid`) was in that list; none of the forks were. Granting the
LFM fork the same `+300` entry:

```sh
adb shell settings put global NubiaperformanceMode \
  "com.pocketpalai+300,com.prismml.moedroid+300,io.github.kingfish600.monsterlfmdroid+300,"
```

| | before | after |
|---|---|---|
| cpu0 (perf, max 3532) | 1996 MHz | **3532 MHz (max)** |
| cpu6 (prime, max 4473) | 1017–1958 MHz | **2841–3840 MHz** |
| decode (single) | 24.1 t/s | **35.9 t/s** |
| **sustained (5 × 300 tok)** | — | **36.15 / 35.73 / 36.14 / 36.02 t/s** |

`+49%`, and it reproduces sustained: ~36 t/s flat across 1500 tokens with cpu6
holding 2.8–3.8 GHz. This single setting explains the entire 22-vs-36 t/s spread
that no amount of flag tuning could account for.

**Original value was backed up** to `~/.perfmode-backup.txt`
(`com.pocketpalai+300,com.prismml.moedroid+300,`). The supported equivalent is to
add the app through RedMagic's Game Space UI, which survives updates better than a
direct `settings put`.

**Implication for every earlier comparison in this project:** the monster's
6.945 t/s was measured **with** the boost; our forks were measured **without** it.
LFM's ~36 t/s is therefore achieved from a level playing field — the earlier
~25 t/s figures understated the fork by ~45%.

## Negative result: DSpark draft speculation loses on x86 (evaluated)
LiquidAI ships a purpose-built drafter for this exact target:
`LiquidAI/LFM2.5-24B-A2B-DSpark-GGUF` → `LFM2.5-24B-A2B-DSpark-F16.gguf`
(0.62 GiB, sha256 `0cd847ca…a1b1a0f`, verified). It is a sidecar: 5 attention
layers, rank-256 Markov head, confidence head, **block size 9**, sharing the
target's embeddings and LM head at load time.

**It loads cleanly on our engine revision** — the shared-tensor concern that
affects Flash-Next's `shared-` heads does not apply here:
```
adding speculative implementation 'draft-dspark'
- n_max=9, n_min=0, p_min=0.00
- block_size=9, mask_token_id=125017, n_extract=5, sample_from_anchor=true
```
Official invocation: `-md <sidecar> --spec-type draft-dspark --spec-draft-n-max 9
--spec-draft-n-min 0 -fa on`.

Host x86, same prompt, temp 0, 400 tokens (`LFM2.5-24B-A2B-Q4_K_M` target):

| config | decode | acceptance |
|---|---|---|
| **baseline (no draft)** | **36.40** | — |
| n_max=4 p_min=0.3 | 34.36 | 63.7% |
| n_max=4 p_min=0.5 | 31.72 | 78.1% |
| n_max=4 p_min=0.7 | 30.46 | 84.7% |
| n_max=6 p_min=0.7 | 30.32 | 81.7% |
| n_max=2 p_min=0.5 | 31.36 | 76.1% |
| n_max=9 p_min=0 | 19.29 | 18.4% |

**No configuration beats the baseline** — best case is 5.6% slower. Two notable
findings: the confidence threshold `p_min` transforms acceptance (18% → 85%), but
counter-intuitively *higher* acceptance is *slower*, so the drafter's own cost, not
acceptance, is the binding constraint. At `n_max=4` the sidecar must cost under 75%
of a target token to win; it reads 0.62 GB/token in F16 against the target's
~1.06 GB/token, and evidently does not clear that bar in practice.

This matches the project's only other speculation measurement (MTP on the monster:
0.52×). **Recommendation: do not ship speculation for LFM.** It costs RAM and
complexity for a measured loss. Retained here because the phone's regime
(bandwidth-bound) differs from x86 and deserved a direct check.

### And on the phone it is catastrophic, not merely slower
Same sidecar, same flags, on the device (boost active, baseline re-measured in the
same run at **35.95 / 35.58 t/s**):

| config | decode |
|---|---|
| baseline (no draft) | **35.9 t/s** |
| `n_max=4 p_min=0.3` + DSpark | **0.35 t/s** |

That is a **~100× slowdown**, not a few percent. The server log shows it grinding
for 14 minutes to produce 143 tokens, with `tg_3s` oscillating between 0.13 and
0.67 t/s. The drafter initialises correctly
(`block_size=9, n_extract=5, sample_from_anchor=true`), so this is a runtime cost
problem, not a load failure.

Worth recording *why this is surprising*: the mechanism should be cheap. DSpark
feeds the drafter `[id_last, <mask> × (block_size-1)]` and denoises the entire
block in a **single** forward pass (5 layers), then the target verifies the block in
one pass — so a round should cost ~1 drafter pass + 1 target pass for up to 9
tokens. Nothing in that should be 100× slow.

Not root-caused. Candidates, in order of suspicion: an op in the drafter's path
falling back to a scalar/generic implementation on ARM; the
`dflash requires ctx_other to be set` warning actually mattering at runtime (it
appears on x86 too, where the same config only lost ~6%); or F16/F32
cross-conversion in the shared embedding/LM head path. **It does not need to be
root-caused to make the decision** — x86 says "no win", the phone says "unusable",
so speculation is out either way. The sidecar stays on the device at
`/data/local/tmp/lfm/` if it is ever worth revisiting.

## RESOLVED: `-lm none` is the right default, and it does NOT swap
The concern was that `-lm none` puts 13 GiB in *anonymous* memory (swappable)
while `mmap` leaves it file-backed (droppable) — so under memory pressure `none`
might swap the model out and collapse. Measured directly, by loading each mode and
then creating ~10 GB of page-cache pressure with `dd`:

| | `-lm none` | `mmap` |
|---|---|---|
| swap free before load | 11.96 GB | 11.96 GB |
| swap free after load | **11.96 GB (unchanged)** | **6.04 GB (−5.9 GB!)** |
| clean decode | **36.15 t/s** | 30.71 t/s |
| decode under 10 GB pressure | **35.98 t/s** | 35.64 t/s |
| after unload | swap unchanged | recovering |

The result is the opposite of the worry: **`none` never touched swap at all**,
while `mmap` consumed 5.9 GB of it on load and was 5 t/s slower until its pages
warmed (its *pressured* decode, 35.64, beats its own clean decode, 30.71 — i.e. the
first tokens after a mmap load are the slow ones).

So the shipped default is correct and should stay: `-lm none` is faster, steadier,
and no worse a citizen. The earlier "swap filled to 832 kB" incident that motivated
the worry was accumulated leftovers from repeated test servers plus other apps, not
the load mode — killing the stale server restored swap instantly.

Note the pressure test also shows `none` is robust: 10 GB of competing page cache
cost it 0.17 t/s (36.15 → 35.98).

## In-app vs server: the gap explained, and a UI fix
The app reported ~29 t/s while the same server, driven directly with curl, gave
**36.9 t/s**. Chasing that:

| condition | decode |
|---|---|
| app's server, fresh short prompt (direct curl) | **36.87 t/s** |
| same server, 1223-token context | **32.42 t/s** (−12%) |
| through the app UI, real conversation | **30.9 t/s** (−5% more) |

So most of the gap is **context length, which is inherent** — decode genuinely
costs more with a long KV. Only ~5% was the UI.

**The UI cost was real though, and fixable.** `AppViewModel` appended to the turn
list once per token, and each append is a full Compose recomposition *plus* an
O(n) string rebuild of the answer so far — at 37 t/s that is 37 recompositions per
second and a quadratic rebuild, competing with the engine for the same cores. Now
buffered and flushed at ~12 Hz (`AppViewModel.startGeneration`). Visually
identical, verified rendering complete (answer, thinking block and t/s all present),
and it recovered ~1.2 t/s of the UI's share.

Also observed while measuring: during UI-driven generation the prime core sat at
1017 MHz (its minimum) for the entire run, whereas a direct curl to the same server
pushed cpu6 to 3513 MHz. Likely cause is UI backpressure — the app reads the
stream slowly, the server blocks on its socket, and the governor sees intermittent
rather than sustained load, so it never ramps. That is consistent with the throttle
helping, and suggests further UI-side optimisation would buy a little more. Not
pursued further; the remaining gain is small and the app is now close to what the
hardware does with a realistic context.

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
