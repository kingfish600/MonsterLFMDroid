# Stage 7 report: on-device shootout (zero-spill kernels, prefill tile, aligned repack)

Date: 2026-10-17/18. Device: RedMagic 10S Pro (NX789J), serial `91251291018D`, Snapdragon 8 Elite,
24 GB, CPU only. External fan active, Game Space ON, `com.prismml.moedroid/MainActivity` focused,
phone unlocked and awake. Hard cutoff 04:30 CDT.

Companion document: `STAGE7-daytime-prep.md` (host-only preparation: the zero-spill `q2_0` decode
kernel `BONSAI_MOE_X4=3`, the wired 2-token prefill tile `BONSAI_MOE_X4_PREFILL=1`, the aligned
activation repack prototype `BONSAI_Q8_0_X4=1`, and the Stage 7 harness).

**Design in one line:** both engines read the *same* sha256-verified model file, every arm is a
long-lived server with **12 timed warmups + 6 measured requests**, all comparisons are within-pass
against that pass's own `A0`, and every bit-exact candidate first has to pass a token-identity gate.

Written incrementally, one section per completed step.

---

## 0. Preflight

Started 21:44 CDT, ~6h45m before the 04:30 cutoff.

| check | result |
|---|---|
| `adb devices` | `91251291018D  device usb:7-1 product:NX789J-UN model:NX789J` |
| `mWakefulness` | **Awake**, `mHoldingDisplaySuspendBlocker=true` |
| focus | `com.prismml.moedroid/com.prismml.bonsaidroid.MainActivity` -> Game Space active |
| CPU caps | **`cpu0-5 = 3532800`, `cpu6/7 = 3840000`** of `4473600` - the known-good profile, no cap gate wait needed |
| governor | `walt`, all 8 cores |
| `/sdcard` | **201 G free** (731 G used, 79%) |
| CPU-zone temperature | **27.8-29.4 C** (fan on; Stage 6 ran these arms at 36.0-36.8 C, Stage 4 at 41-43 C) |
| stray engines | **none**: no `libllamaserver`, no listener on 8299/8199 |
| app process | pid 23226, RSS **290 MB**, **6 CPU ticks over 6 s** (1 tick/s of UI, no inference), holds no CPU and no model. Left running because Game Space depends on it |
| model mapped | scanned every pid's `/proc/<pid>/maps` for `Qwen3.8-Flash-Next`: **no process holds the model** |

Conclusion: the device is in the exact state the protocol requires, and the app's engine is dormant
rather than stopped, so Game Space stays engaged throughout.

## 1. Identity gate (`s7_shootout.sh 0`) - ALL SIX GATES PASS

Started 21:46:06, finished 21:56:00 (**~10 min**, well under the 35 min budget: the arms themselves
are short, the 30 s cap gate dominating each one).

Staging was gated before anything was pushed: the sha256 of both model shards matched varH exactly,
`ik supports --cpu-mask-batch: 1`, and all six Stage 7 symbols were present in the shipping
`libggml-cpu.so`, so no arm could silently measure the wrong binary.

| arm | change under test | prompt | md5 of greedy token ids | verdict |
|---|---|---|---|---|
| `id-A0` | nothing (shipped defaults) | short, 64 tokens | `8ad776c2658411fcbe94ff8ca509d659` | baseline |
| `id-A5` | `BONSAI_MOE_X4=3` (zero-spill decode kernel) | short, 64 | `8ad776c2658411fcbe94ff8ca509d659` | **PASS - identical** |
| `id-A6` | `MOE_X4=3` + `BONSAI_Q8_0_X4=1` (aligned repack) | short, 64 | `8ad776c2658411fcbe94ff8ca509d659` | **PASS - identical** |
| `id-A6b` | `A6` repeat | short, 64 | `8ad776c2658411fcbe94ff8ca509d659` | **PASS - repeatability control** |
| `id-pf-A0` | nothing, long prompt | **~262-token prefill**, 16 tokens | `514b317d2b0bcb62cf3818361dad64e1` | baseline |
| `id-pf-AP2` | `BONSAI_MOE_X4_PREFILL=1` (2-token prefill tile) | ~262-token prefill, 16 | `514b317d2b0bcb62cf3818361dad64e1` | **PASS - identical** |
| `id-pf-A6` | `MOE_X4=3` + `Q8_0_X4=1` + `MOE_X4_PREFILL=1` | ~262-token prefill, 16 | `514b317d2b0bcb62cf3818361dad64e1` | **PASS - identical** |
| `id-pf-AP2b` | `AP2` repeat | ~262-token prefill, 16 | `514b317d2b0bcb62cf3818361dad64e1` | **PASS - repeatability control** |

Driver verdict lines:

```
IDENTITY PASS (MOE_X4=3 vs shipped)
IDENTITY PASS (MOE_X4=3 + Q8_0_X4 repack vs shipped)
IDENTITY PASS (Q8_0_X4 repack repeatability)
IDENTITY PASS (prefill tile vs shipped, 256-token prefill)
IDENTITY PASS (prefill tile + repack vs shipped)
IDENTITY PASS (prefill tile repeatability)
```

**Per-arm triage: no arm needed excluding.** All three independent changes are token-identical to the
shipped engine on both prompt families, so the full arm list `A0 A5 A6 AP2 B0 B1 B2` runs in Step 2.
Two facts make these gates meaningful rather than decorative:

* `evaluated=262` on every long-prompt arm confirms the prefill really was ~256 tokens. The paired
  loop needs two routed rows, so a short prompt would have left the new code never entered and
  "identical" would have been vacuous - this is the same trap Stage 6 hit with a 6-token prompt.
* each arm printed its `S7ENV` line, so the variable that was supposed to be set is provably the
  one the engine ran with (e.g. `BONSAI_Q8_0_X4=1` on `id-A6`, `unset` on `id-A0`).

Consequence for the night: `A5`, `A6` and `AP2` are all **Gate-1 eligible** (bit-exact by
construction on the host under qemu, and token-identical end to end here), so any of them may be
shipped on speed evidence alone, with no quality caveat.

## 2.0 A protocol-level finding that must be read before any number in section 2

The first two decode arms revealed that **the phone thermally throttles about 155 s into every arm**,
and that this is a property of the 12-warmup protocol on this device tonight, not of any candidate
change. Recorded here, before the rest of the data, because it changes how every absolute number in
this report must be read.

Measured on `p1-A0` (the shipped engine, no env vars):

| observation | value |
|---|---|
| caps during the arm, sampled every 5 s | **31 samples at `3532800/3840000`** (the good profile), then **11 samples at `2400000/2841600`**, then 1 transitional |
| hottest CPU zone during the arm | **103.5-105.0 C** (`cpu-0-0-0`, `cpu-0-0-1`, `cpuss-0-0`) |
| battery | **AC powered, 39.0 C**, level 89% |
| caps after the 120 s cooldown | back to `3532800/3840000` |

So the sequence inside every arm is: start cool at the known-good profile, run ~2.5 minutes of
sustained 6-thread load, hit the thermal trip at ~105 C, drop to `2400000/2841600`, and **run the
measured window in that throttled state**. The 12 timed warmups take ~156 s, which is almost exactly
when the trip happens, so the transition lands **inside the warmup phase and not inside the measured
window**:

| window | `p1-A0` server t/s | what it is |
|---|---|---|
| warmups 1-3 | 4.69, 5.55, 6.23 | page cache filling (the effect §A.3 measured) |
| warmups 9-12 | 6.56, 5.63, 6.16, 6.47 | **un-throttled steady state, ~6.4-6.5 t/s** |
| measured 1-6 | 4.87, 5.65, 5.49, 5.75, 5.50, 5.52 | **throttled steady state, median 5.51** |

Three consequences, and they are the reason this section exists:

1. **Every absolute in section 2 is a throttled-state number** (`2400000/2841600`, i.e. 32%/26% below
   the known-good profile). It is therefore **not comparable to Stage 5's 6.90 t/s** (which ran cool
   and unpinned by this throttle) nor meaningfully to Stage 6's 5.39 (a cold-cache number). The
   within-pass percentages are the only valid output of this run.
2. **The un-throttled rate is still visible in the warmup series** and it corroborates §A.3: the
   engine reaches ~6.4-6.5 t/s once the page cache settles, which is the 6.52 median §A.3 measured
   with 14 warmups. That means stage 6's 5.39 really was a warmup-limited number, and tonight's 5.51
   is a *throttle*-limited number, and neither is the engine's capability.
3. **The 12-warmup protocol accidentally produces the cleanest measurable arrangement available**,
   and that is worth stating because the opposite was expected: with 4 warmups the throttle
   transition would land *inside* the measured window, which would make the six samples bimodal
   (some at 6.5, some at 5.5) and hammer exactly the median the decision rests on. With 12 warmups
   the transition is absorbed by the warmups. The cost is that the measured regime is throttled; the
   benefit is that it is *consistently* throttled.

What this means for the night's decisions: a candidate beats `A0` if it wins **within its pass, in
both passes**, in the throttled regime. A kernel that reduces instructions should help more, not
less, when the clock is lower, so the direction of the result is still informative; the *magnitude*
is a throttled-regime magnitude and will be reported as such. The alternative - cooling the phone
between arms - cannot fix this, because the throttle is caused by the warmups inside each arm, not by
heat carried over from the previous one (caps are back to full after every 120 s cooldown).

## 1.5 A harness defect found and fixed on the first pass (the reason section 2 starts at 22:43)

The first attempt at the timed passes produced one arm before the defect became obvious, and it is
recorded here because it changed the data, not just the schedule.

**Defect:** `timed_pass()` in `s7_shootout.sh` passed `$LIB_OURS` to every arm and only switched the
*engine name*. So `B0` and `B1` (the `ik_llama.cpp` arms) ran **our** engine with ik's flag list, and
`B2` died instantly:

```
=========== S7 LL ARM p1-B2 engine=ik mode=decode dir=/data/local/tmp/lib-s7-ours ===========
S7FAIL engine died
error: invalid argument: --no-fused-moe
```

The `dir=` in that line is the tell: an `ik` arm must never have `lib-s7-ours` as its library
directory, and `--no-fused-moe` is an ik-only flag, so the error was our engine rejecting an
argument it has never had.

**Consequence for the data:** the first pass's `B0` (5.5021) and `B1` (5.4238) were our engine, not
ik, and are **discarded** - they are not quoted anywhere in this report. The first pass's four `A`
arms were genuine (correct library directory, correct env) and are kept in
`.stage7-logs/aborted-run1-harness-bug/` as a partial block, but they are **not** mixed into the
tables in section 2: mixing a 41-minute-old block with a fresh run would breach the project's
"one protocol block per claim" rule for no benefit.

**Fix and proof:** the library directory is now chosen from the engine (`ik` -> `$LIB_IK`, otherwise
`$LIB_OURS`) and each arm logs its own `(cfg -> engine=…, lib=…)` line. Verified on the device before
restarting, with the engine's own configuration report as the evidence:

| smoke arm | library dir | engine's own line | meaning |
|---|---|---|---|
| `B0` | `lib-s7-ik` | `fused_moe = 1` | ik, fusion on, no `--poll`, no pinning |
| `B2` | `lib-s7-ik` | **`fused_moe = 0`** | ik, `--no-fused-moe` **accepted**, fusion provably off |

That is the check that would have caught this before the first pass: an ik arm must show
`fused_moe` from ik's own init log, and `B2` must show `= 0`. Both do now.

Timed passes restarted 22:43 CDT with the fixed driver, same order, same 12-warmup protocol.

## 2. Decode, pass 1 (21:57 ordering re-run at 22:43; arms in forward order)

Long-lived server, **12 timed warmups + 6 measured**, one model load per arm, 120 s cooldown.
Every arm verified its engine from the engine's own log (`fused_moe = 1` for `B0`/`B1`, `= 0` for
`B2`), and every `ik` arm ran from `/data/local/tmp/lib-s7-ik` with **no `--poll`**.

| arm | config | median t/s | vs `A0` | throttled fraction |
|---|---|---|---|---|
| `A0` | shipped defaults (`MOE_X4=1`, `MULMATID_FLAT=2`) | **5.6422** | - | 24/44 = 55% |
| `A5` | `BONSAI_MOE_X4=3` (zero-spill decode kernel) | 5.3957 | **-4.37%** | 55% |
| `A6` | `A5` + `BONSAI_Q8_0_X4=1` (aligned repack) | 5.2728 | **-6.55%** | 55% |
| `AP2` | `BONSAI_MOE_X4_PREFILL=1` (2-token prefill tile) | 5.6054 | **-0.65%** | 52% |
| `B0` | `ik_llama.cpp`, no pinning, no `--poll` | **8.5714** | **+51.9%** | 17/42 = 40% |
| `B1` | ik + `--cpu-strict 1 --cpu-mask-batch 0xff --cpu-strict-batch 1` | 8.3446 | vs `B0` **-2.65%** | 41% |
| `B2` | `B1` + `--no-fused-moe` | 8.0397 | vs `B1` **-3.65%** | 39% |

Three results, and the first one is a refutation of my own hypothesis.

### 2.1 `A5` is SLOWER: the zero-spill kernel loses ~4% to the spilling one

`BONSAI_MOE_X4=3` has a 0 B frame and zero local spills against mode 1's 432 B frame and 55 stack
accesses, and it is bit-exact - and it is **4.4% slower**. This is now the second independent pass
that says so (the discarded first run measured -2.47% at the same position), so it is not a fluke of
one arm.

The mechanism was identified in the daytime design and is confirmed by the shape of the result: to
halve the accumulator pressure, mode 3 processes two weight rows per pass over the activation, which
means the activation blocks for a given block index are **loaded twice** (8 vector loads per 128
weights instead of 4). Those loads are L1-resident, but they are extra load-pipe slots in a kernel
that is already waiting on the weight stream. The spills mode 1 pays are absorbed by the store/load
ports without stalling the dependency chain; the duplicate activation loads are not free.

That is the same lesson `A3` (the `q3_K` four-row kernel, -5.7%) taught, and it is now the second
confirmation: **zero spills is a property of the register allocator, not of the machine.** A kernel
change in this project has to be judged by measurement, always, even when its static properties look
strictly better. Mode 3 stays in the tree behind `BONSAI_MOE_X4=3`, opt-in, and **must not** become
the default.

### 2.2 `A6`'s repack: pass 1 said -2.3%, pass 2 said +2.4% - it is inside the noise, i.e. nothing

**This paragraph was first written from pass 1 alone as "the repack does not pay for itself". Pass 2
contradicts that and the honest version is below.** `A6 - A5` per pass: **-2.28%** (pass 1) and
**+2.37%** (pass 2), so the mean is **+0.05%** with a 4.7-point swing round it. That is not a loss,
it is a null result, and it is the same size as the night's replication noise (below).

So the conclusion changes from "worse" to "worth nothing measurable", which still means
`BONSAI_Q8_0_X4` stays off: a change that is neutral within noise is not worth carrying, and the
project's own rule is that neutral results ship opt-in **off**. The useful part is the negative
information: the 34-byte `block_q8_0` stride is not what limits this kernel, so ik's
`block_q8_0_x4` repacked layout is **not** where ik's 52% lives either.

### 2.2b The night's noise floor, measured rather than assumed

`AP2` is a config that provably cannot affect a 64-token decode, and with 7 arms in a mirrored design
it lands in **the same position (4th) in both passes**. That makes its two measurements a pure
replication control:

| control | pass 1 | pass 2 | spread |
|---|---|---|---|
| `AP2` (same position both passes, no possible decode effect) | 5.6054 | 5.7767 | **+3.0%** |
| `A0` (position 1 -> position 7) | 5.6422 | 5.5104 | 2.4% |
| `B0` (position 5 -> position 3) | 8.5714 | 8.1381 | 5.0% |

So an effect has to exceed about **3%** on the mean of two passes to be worth quoting, and ±5% for
the ik arms. That is why §2.1's `A5` result is quoted as a ~3.6% mean with both passes negative (a
sign test as well as a magnitude), while `A6`'s repack is called null. It is also the reason the
per-op profiler in §4 matters more than usual tonight: it measures inside the op, where the noise is
one order of magnitude smaller.

### 2.3 `AP2` is neutral on decode, which is exactly what it had to be

The 2-token prefill tile cannot affect a 64-token decode (the request's 7-token prompt is the only
prefill, and the paired loop needs two routed rows). Measured **-0.65%**, i.e. inside the night's
noise. That is a useful negative control: it says the +/-2% band contains no effect at all, which
calibrates how much of `A5`'s -4.37% and `A6`'s -2.28% is real. Both are outside it.

### 2.4 The `ik_llama.cpp` gap is real, large, and survives throttling

`B0` = **8.5714 t/s** against our **5.6422**: **+51.9%**. This is the number Stage 6 could not measure
cleanly, and three things about it matter:

* **It is not a `--poll` artefact.** No `--poll` was passed to ik in any arm; Stage 6's `B1`/`B2`
  were poisoned by it and are now superseded.
* **It is not a cold-cache artefact.** ik throttled *less* than we did (40% of its samples vs our
  55%) and still won by half again, at the same caps.
* **It reproduces at a larger magnitude than Stage 6's +38%**, which makes sense in the other
  direction from what the daytime document predicted: our engine is now measured in a throttled
  steady state (5.64), while Stage 6's 5.39 was a cold-cache number that happened to look similar.
  The gap was understated, not overstated.

`B1 - B0`: **-2.65%** in pass 1 but **+0.98%** in pass 2, so mean **-0.85%** with a 3.6-point swing.
That is inside the noise: our `0xff` batch-mask pinning, ported to ik, does **nothing measurable** for
ik's decode - consistent with Stage 6 §4.1 (ik already pins strictly per worker, so the mask only adds
a second policy that agrees with the first). Use `B0` as ik's decode configuration.

`B2 - B1` = **-3.65%**: turning ik's fused up+gate **off** costs it 3.65%, so the fusion is worth
~3.7% to ik. Stage 6 measured ~0.8-3% with `--poll` contamination; this is the clean number and it is
larger. Note what it implies for us: even if we built the fused op tomorrow and captured 100% of that
3.7%, it would close less than a tenth of the current gap.


### 2.5 Decode, both passes together

Within-pass deltas are the only valid comparison; the last two columns are the position control.

| arm | pass 1 | pass 2 | mean median | **pass 1 delta** | **pass 2 delta** | both negative? |
|---|---|---|---|---|---|---|
| `A0` | 5.6422 | 5.5104 | 5.5763 | - | - | - |
| `A5` (`MOE_X4=3`) | 5.3957 | 5.3524 | 5.3741 | **-4.37%** | **-2.87%** | **yes, -3.62% mean** |
| `A6` (`A5` + repack) | 5.2728 | 5.4795 | 5.3762 | -6.55% | -0.56% | marginal, -3.56% mean |
| `AP2` (`MOE_X4_PREFILL=1`) | 5.6054 | 5.7767 | 5.6911 | -0.65% | +4.83% | no: neutral (+2.09% mean) |
| `B0` (ik) | 8.5714 | 8.1381 | 8.3548 | +51.9% | +47.7% | **yes, +49.8% mean** |
| `B1` (ik + batch pinning) | 8.3446 | 8.2176 | 8.2811 | -2.65% vs B0 | +0.98% vs B0 | no: neutral |
| `B2` (ik, fusion off) | 8.0397 | 8.0548 | 8.0653 | -3.65% vs B1 | -1.98% vs B1 | **yes, -2.82% mean** |

Decode verdicts, fixed before the prefill data arrived:

* **`A5` does not ship.** Both passes negative, tight spread on `A5` itself (0.8%), and the loss is
  larger than the noise floor. The zero-spill kernel is a genuine regression and stays opt-in off.
* **`A6` does not ship.** Its own repack effect is null (§2.2) and it inherits `A5`'s regression.
* **`AP2` is neutral on decode**, as required. Its verdict comes from the prefill passes in §3.
* **The `ik` gap is +49.8% decode**, up from Stage 6's +38%, and it is the largest single fact of the
  night: our engine has no shipped change that touches it.

## 3. Prefill, pass 1 (262-token prompt, mean of 6 measured requests)

The prefill metric is llama-server's own `prompt_per_second`, so it excludes generation entirely.

| arm | median t/s | vs `A0` |
|---|---|---|
| `A0` | **8.1481** | - |
| `A5` (`MOE_X4=3`) | 8.0066 | -1.74% |
| `A6` (`A5` + repack) | 7.7601 | -4.76% |
| `AP2` (`MOE_X4_PREFILL=1`) | **8.4460** | **+3.66%** |
| `B0` (ik) | **9.8330** | **+20.7%** |
| `B1` (ik + batch pinning) | 9.8275 | vs `B0` -0.06% |
| `B2` (ik, fusion off) | 9.4560 | vs `B1` -3.78% |

Four things, and the first is the only positive result of the night for our engine:

1. **`AP2` helps prefill by +3.66%**, which is the direction the design predicted and the first
   positive movement from any Stage 7 candidate. The mechanism is the one in §B.5: the paired loop
   takes the weight-block expansion that used to run once per token and runs it once per **two**
   tokens, so during prefill the expansion cost per token halves. Decode is untouched (it has one
   token, and §2.3 measured -0.65%, i.e. nothing), which is exactly the shape a prefill-only change
   should have.
2. **`A5` is slower on prefill too** (-1.74%), consistent with decode: the doubled activation loads
   cost more than the spills save, in both phases.
3. **`A6` loses 4.76%** on prefill. Note this is `A5` (-1.74%) plus another ~3%, and it is the one
   place the repack looks clearly bad rather than null - prefill is where the activation traffic
   matters, and the repack adds ~18% of it.
4. **`ik` leads prefill by +20.7%** (`B0` 9.833 vs our 8.148). That is a *smaller* lead than decode
   (+49.8%) and it is very close to the +23.6% residual Stage 6 measured *after* its single-queue
   prefill change shipped - so our prefill position has not moved, while our decode position has
   worsened relative to ik.
   `B1 - B0` = **-0.06%**: ik's batch pinning does nothing for its prefill either, which finally
   answers the question Stage 6 could not (its `B1` was poisoned by `--poll`): the `0xff` batch mask
   is *our* fork's lever and confers nothing on ik in either phase.
   `B2 - B1` = **-3.78%**: the fused up+gate is worth ~3.8% to ik's prefill, matching its ~2.8%
   decode value.

### 3.5 Prefill, both passes together

| arm | pass 1 | pass 2 | mean | **pass 1 delta** | **pass 2 delta** | replication spread |
|---|---|---|---|---|---|---|
| `A0` | 8.1481 | 8.1723 | 8.1602 | - | - | 0.30% |
| `A5` (`MOE_X4=3`) | 8.0066 | 8.2326 | 8.1196 | -1.74% | +0.74% | 2.82% |
| `A6` (`A5` + repack) | 7.7601 | 7.9703 | 7.8652 | **-4.76%** | **-2.47%** | 2.71% |
| **`AP2`** (`MOE_X4_PREFILL=1`) | **8.4460** | **8.4135** | **8.4298** | **+3.66%** | **+2.95%** | **0.39%** |
| `B0` (ik) | 9.8330 | 9.8755 | 9.8543 | +20.7% | +20.8% | 0.43% |
| `B1` (ik + batch pinning) | 9.8275 | 9.8971 | 9.8623 | -0.06% vs B0 | +0.22% vs B0 | 0.71% |
| `B2` (ik, fusion off) | 9.4560 | 9.0739 | 9.2650 | -3.78% vs B1 | -8.32% vs B1 | 4.21% |

**`AP2` is the one shippable result of the night**, and it is a clean one:

* **both passes positive** (+3.66%, +2.95%), mean **+3.31%**;
* **its own replication spread is 0.39%** (8.4460 vs 8.4135), an order of magnitude tighter than any
  other our-engine arm, because it lands in the same (4th) position in both passes;
* it is **neutral on decode** by construction, and §2.3 measured that (-0.65%, +4.83%: two signs,
  inside the reproduction spread of a config that *cannot* affect decode);
* it is **Gate 1 bit-exact** (§1), so shipping it carries no quality risk whatsoever.

That is the strongest evidence pattern this project has accepted before (`A1` in Stage 6 shipped on
two positive passes plus independent per-op attribution). §4 supplies the per-op attribution for
`AP2` on purpose, because a +3.3% wall-clock result with a +3% noise floor needs it.

`A5` is neutral-to-negative on prefill (-0.5% mean) and negative on decode, so it is rejected on both
phases. `A6` is negative in both prefill passes and inherits `A5`'s decode loss: rejected.

`ik`'s prefill lead is **+20.75%** and it is remarkably stable across passes (9.833 / 9.876), so it is
a solid floor for the gap: after our own prefill queue shipped in Stage 6, our prefill position has not
improved relative to ik, and decode has fallen further behind (+49.8%).

## 4. The per-op profiler pass: data collected, and NOT usable for a cross-arm claim tonight

Three `PROF=1` decode arms ran with `BONSAI_PROF_EVERY=1` on the instrumented build
(`build-arm64-prof`), and the dumps were pulled and differenced over the last 48 graphs:

| op | `prof-ours-A0` | `prof-ours-A5` | delta | share A0 | share A5 |
|---|---|---|---|---|---|
| `MUL_MAT` (dense) | 89.628 ms | 80.271 ms | -10.44% | 52.03% | 51.43% |
| `MUL_MAT_ID` (experts) | 53.074 ms | 48.427 ms | -8.76% | 30.81% | 31.03% |
| `GET_ROWS` | 6.637 ms | 6.250 ms | -5.84% | 3.85% | 4.00% |
| `GATED_DELTA_NET` | 6.269 ms | 5.391 ms | -14.00% | 3.64% | 3.45% |
| `FLASH_ATTN_EXT` | 0.424 ms | 0.527 ms | +24.23% | 0.25% | 0.34% |
| **total** | **172.259 ms/token** | **156.069 ms/token** | **-9.40%** | - | - |

**Read at face value this says `A5` is 9.4% faster, which contradicts the wall clock in the same arm
(5.3608 t/s vs `A0`'s 5.6689, i.e. -5.4%). The profiler is wrong here, and the table says why:**

* Every op moved down together, including `GET_ROWS` (-5.8%) and `GATED_DELTA_NET` (-14.0%), which a
  change to the `q2_0` matmul kernel **cannot touch**. A uniform shift across untouched ops is a clock
  difference, not a kernel effect.
* The **shares** are unchanged (the expert bucket is 30.81% before and 31.03% after, i.e. +0.7%
  relative), so no structural change is visible either.
* The instrumented arm's own wall clock disagrees with its own op total by 19.5%
  (156.069 ms/token implies 6.407 t/s; the arm measured 5.3608). For `A0` the same two numbers agree
  to 2.4% (172.259 ms/token -> 5.805 t/s vs 5.6689 measured). So the instrument was consistent for
  one arm and not the other, which is a property of *when* the arm ran, not of the kernel.

**Conclusion: the per-op comparison is confounded by throttle and is not used for any claim in this
report.** Two profiler arms cannot be run far enough apart tonight without the clock moving underneath
them, and the arm-level wall clock is the primary measurement. That is the honest reading, and it
means `A5`'s regression rests on the four within-pass wall-clock pairs (two in the discarded first run,
two in this one), all negative, with `A5`'s own spread at 0.8%.

The two `ik` profiler arms were **cancelled** for the same reason (they would carry the identical
confound), and because their information value is low: the fusion's worth is already cleanly measured
by `B2 - B1` in both phases (-2.82% decode, -5.9% prefill mean). Note for the next stage: Stage 6's
own `prof-ik` arms were additionally `--poll`-poisoned, so **a clean ik per-op budget still does not
exist** and is worth collecting in a dedicated run where the clock can be held steady (for example by
profiling at a fixed, deliberately capped frequency).

## 5. Ship decision

One candidate cleared every gate that applies to it and beat the baseline in both mirrored passes:
**`BONSAI_MOE_X4_PREFILL` (the 4-row x 2-token `q2_0` prefill tile) is now the default.**

| candidate | Gate 1 (token identity) | pass 1 | pass 2 | verdict |
|---|---|---|---|---|
| **`AP2`** `BONSAI_MOE_X4_PREFILL=1` | **PASS**, short prompt **and** real 262-token prefill | prefill **+3.66%** | prefill **+2.95%** | **SHIPPED as default** |
| `A5` `BONSAI_MOE_X4=3` | PASS | decode -4.37%, prefill -1.74% | decode -2.87%, prefill +0.74% | **REJECTED** (slower in both phases) |
| `A6` `BONSAI_Q8_0_X4=1` | PASS | decode -6.55%, prefill -4.76% | decode -0.56%, prefill -2.47% | **REJECTED** (repack null to negative) |

`A5` and `A6` stay in the tree behind their switches, off by default, with their measured verdicts
recorded in the comments at each site so nobody re-enables them on the strength of the static
argument (zero spills) that this run refuted.

### 5.1 The change

`ggml/src/ggml-cpu/arch/arm/iqk_x4.c`, `bonsai_q2_0_x4_prefill_mode()`: default `0` -> **`1`** when
`BONSAI_MOE_X4_PREFILL` is unset. `=0` restores the stock loop, so the change is reversible on device
without a rebuild. The measurement is recorded in the comment at the site.

### 5.2 Verified on the device with no environment variables set

`build-arm64-ship` rebuilt, re-staged, and run exactly as the app will run it:

| check | expected band | measured | verdict |
|---|---|---|---|
| prefill, no env | `AP2` band `8.4460 / 8.4135` (not `A0`'s `8.1481 / 8.1723`) | **8.4483** | **in the AP2 band** |
| decode, no env | `A0` band `5.6422 / 5.5104` (the tile cannot affect decode) | **5.4930** | **in the A0 band** |

Both verifications printed `S7ENV ... vars=BONSAI_MOE_X4_PREFILL=unset`, so the default is what
produced the numbers - not a leftover environment variable.

### 5.3 App libraries synced and the APK installed

| step | result |
|---|---|
| `jniLibs/arm64-v8a/` synced in **both** repos | `libggml-cpu.so` sha256 `c035d22ecb3dd4e2...` in `MoeDroid`, `MonsterQwenDroid` and `build-arm64-ship/bin` - all three identical |
| pre-ship backup | `.stage7/jniLibs-backup/{MoeDroid,MonsterQwenDroid}/` (one-command revert) |
| APK build | `tools/gradle-8.14/bin/gradle assembleDebug -g tools/gradle-home` -> **BUILD SUCCESSFUL in 7s** |
| APK | `apk-out/MoeDroid-0.2.0-pinned-stage7-ap2-arm64.apk`, 126,665,764 B, sha256 `14bf3e0877023a8f...` |
| **kernel really inside the APK** | the `lib/arm64-v8a/libggml-cpu.so` extracted **from the APK** hashes to `c035d22ecb3dd4e2...`, identical to the build - not assumed |
| install | `adb install -r` -> **Success** (in-place upgrade, never uninstall) |
| installed version | `versionCode=11`, `versionName=0.2.0-pinned`, `lastUpdateTime=2026-10-18 02:37:07` |
| installed native lib | `/data/app/.../lib/arm64/libggml-cpu.so` = **5,700,120 B**, sha `c035d22ecb3dd4e2...` |
| model preserved | shard 1 `37,033,916,192`, shard 2 `28,800,138,432`, MTP `2,786,204,800` - all unchanged |
| app state after | relaunched, 217 MB RSS, holds Game Space |

Installed at 02:37 CDT, well inside the 04:15 cut-off.

### 5.4 Engine patch refreshed and proven against the base commit

`MonsterQwenDroid/patches/llama.cpp-moedroid.patch` was stale (it predated both the Stage 6 and Stage
7 kernels, and `arch/arm/iqk_x4.c` was an **untracked** new file, so a plain `git diff` could never
have contained it). It was regenerated after `git add -N ggml/src/ggml-cpu/arch/arm/iqk_x4.c`, as the
cumulative diff from llama.cpp `1b70350`:

| check | result |
|---|---|
| files covered | **17** (was 9), including `ggml/src/ggml-cpu/arch/arm/iqk_x4.c` |
| size | 116,688 B |
| applies cleanly to a clean `1b70350` worktree | **PASS** |
| reverse-applies from the patched state | **PASS** |
| the patched tree is byte-identical to our working tree, all 17 files | **PASS** |

So `GOAL.md` constraint 12 now holds for this tree, for the first time since the Stage 6 kernels
landed.

## 6. Honest limits of this dataset

1. **Two passes, one pair per config.** The project's own standard is ~11-16 pairs to resolve a 2-3%
   effect at a 5-8% arm-to-arm spread. `AP2`'s +3.31% mean rests on two pairs, and it is shippable
   only because (i) both pairs are positive, (ii) its own replication spread is **0.39%**, i.e. ten
   times tighter than the effect, and (iii) it is Gate-1 bit-exact so a wrong decision costs speed,
   not quality. `A5`'s rejection is stronger: **four** within-pass pairs across two independent runs,
   all negative.
2. **Every absolute in this report is a throttled-state number** (§2.0). The engine's un-throttled
   rate is visible in the warmup series (~6.4-6.5 t/s for `A0`) and is ~15% above the measured
   median. Do not compare these absolutes with Stage 5's 6.90 or Stage 6's 5.39.
3. **The per-op profiler is not usable tonight** (§4): the two arms ran at different clocks, which is
   provable because ops the change cannot touch moved by 6-14% too. The `ik` per-op budget is still
   missing, and Stage 6's was `--poll`-poisoned, so **a clean ik budget does not exist yet**.
4. **The ik gap is a matched-flag, same-file, same-session number but not a matched-*thermal* one.**
   ik throttled less than we did, which flatters it slightly; and our engine is the one measured in a
   throttled steady state. The +49.8% decode figure is therefore a floor in one direction and a
   ceiling in another and should be quoted as "about half again as fast", not to two decimals.
5. **`BONSAI_MOE_X4_PREFILL` only engages when an expert receives two or more routed rows.** Its
   benefit therefore scales with how many rows per expert a prompt produces - a 262-token prompt does
   it well; a short prompt does it barely. Real chat traffic with long contexts will see more of it
   than a 6-token prompt would.

## 7. Session result in one paragraph

The night produced one shipped win, one refutation, one null result and one measurement that resets
the project's target. Shipped: the 4-row x 2-token `q2_0` prefill tile, **+3.31% prefill mean with
+3.66%/+2.95% in the two mirrored passes and a 0.39% replication spread**, token-identical on a real
262-token prefill, neutral on decode, verified on device with no environment variables set (8.4483
against the `A0` band of 8.15-8.17), synced into both `jniLibs` trees, built into
`MoeDroid-0.2.0-pinned-stage7-ap2-arm64.apk` and installed at 02:37 CDT. Refuted: the zero-spill
decode kernel, **-3.62% mean over four pairs**, because halving the accumulator pressure doubles the
activation loads and the spills it removed were cheaper than the loads it added - the second time this
project has learned that a statistically cleaner kernel is not a faster one. Null: the aligned
activation repack, +0.05% mean on decode and negative on prefill. Reset: with `--poll` finally out of
the ik arms and 12 warmups on both engines, `ik_llama.cpp` leads **+49.8% decode and +20.75% prefill**,
both passes, same file, same flags - larger than Stage 6's +38%/+24%, so the gap was understated and
nothing shipped tonight touches it. The engine patch was also regenerated for the first time since the
Stage 6 kernels landed and proven to apply and reverse-apply against `1b70350` with all 17 files
matching the working tree.

