# Stage 11 / Window 3 report

Date: 2026-10-20. Device: RedMagic 10S Pro (NX789J), serial `91251291018D`, CPU only.
Written incrementally, one section per completed step. Steps run in the mandated order:
**Step 1 (build) -> Step 2 (instrument) -> Step 3 (cold, fanless) -> Step 4 (fan on, ~4 h)**.

## 0. Ground truth used (not re-derived)

| item | value |
|---|---|
| installed | `com.prismml.moedroid` versionCode **13** (`0.3.1-hybridview-ui`) |
| packaged `libllama.so` | sha256 `a8c15f40…`, byte-identical to `build-arm64-ship` |
| engine default | `BONSAI_HYBRID_VIEW=1` (V1) ON |
| UI | flattened header + live `tok/s` chip, readout always rendered |
| measured warm through the app | 5.43 / 7.47 / 7.48 / 7.48 t/s; `prompt_ms` 1571 -> 516 -> 379 -> 364 |
| bench V1 | 7.105 t/s |
| second cause of "~5 t/s" | RedMagicOS holds cores at 1152000/1017600 when the app is **not** in Game Space; `scaling_max_freq` is blind to that, `scaling_cur_freq` is the witness |

---

## 1. Step 1 — Build 14 (warmup chip)

### 1.1 The warmup counter, with the reset semantics the brief fixed

Implemented in `ui/AppViewModel.kt` as a `StateFlow<Int>` fed by `noteCompletedGeneration()`:

```kotlin
private val _warmupCount = MutableStateFlow(0)      // a fresh process starts at 0
private var lastCompletionAtMs = 0L

private fun noteCompletedGeneration() {             // called ONCE per completed generation
    val now = System.currentTimeMillis()
    if (lastCompletionAtMs != 0L && now - lastCompletionAtMs > 15 * 60 * 1000L) {
        _warmupCount.value = 0                      // >=15 min idle: page cache has gone cold
    }
    lastCompletionAtMs = now
    _warmupCount.value = _warmupCount.value + 1
}
```

* **Reset points: process start (implicit 0) and a >=15 minute idle gap.** Nothing else.
* **Not reset on engine start or model reload** - explicitly, because the page cache is a system
  property: killing the engine does not evict 66 GB of model pages, so an engine restart still yields
  a *warm* request. Labelling that cold would be a lie, and this is the exact semantics the brief
  insists on.
* **Incremented once per COMPLETED generation**: on the success path only, after the assistant turn is
  persisted. A cancelled or failed generation does not advance it.

### 1.2 The chip

`ui/ChatScreen.kt`, extending the existing chip only (header untouched):

| state | label |
|---|---|
| rate available, `warmupCount <= 2` | `warming (N/3) · X.XX tok/s` |
| rate available, `warmupCount >= 3` | `X.XX tok/s` |
| no usable rate | `tok/s —` (Build 13 behaviour, never hidden) |

Two decimals retained so the +5% stays visible, and the `>5000 t/s` guard from Build 13 is preserved
(instant-EOS garbage is not a measurement).

### 1.3 Clock witness — decision deferred to the device probe

The brief requires verifying readability **from inside the app** before shipping anything. The phone
was unplugged immediately after the build work began, so the probe has not run yet and **no witness
code has been written**. The plan is fixed and will not be improvised:

1. probe `/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq` from the app UID;
2. if readable: a subtle low-power hint when the value is `< 3000000`, nothing otherwise;
3. if denied/empty: **drop the witness entirely** and record "in-app clock witness infeasible
   (SELinux)" here - no chip that renders "ok" when it means "unknown".

### 1.4 Version, build, diff gate, install — pending

`versionCode 13 -> 14`, `versionName "0.3.2-warmup-chip"`, build, the mandatory native-byte diff gate
against Build 13, `apk-out/MoeDroid-0.3.2-warmup-chip-14-arm64.apk`, install and `versionCode=14`
confirmation all happen in one pass once the phone is back, so the witness decision is made once on
evidence rather than twice on a guess.

---

## 2. Step 2 — instrument upgrade (complete, host-only)

`MoeDroid/tools/s10_endurance.sh` **exists** (the Stage 10 soak was not inline), so it was extended in
place rather than adding a Stage 11 duplicate.

### 2.1 In-window `scaling_cur_freq`, sampled DURING the request

`MoeDroid/tools/s8_arm_ll.sh` gained `cur0()`/`cur6()` and a 1 Hz sampler that runs **for the whole
duration of each measured request**, started immediately before the `curl` and stopped after it:

```sh
: > "$OUT.cur"
( while :; do echo "$(cur0) $(cur6)" >> "$OUT.cur"; sleep 1; done ) &
CURPID=$!
curl -s ... > "$OUT.json"
kill $CURPID 2>/dev/null; wait $CURPID 2>/dev/null
```

per-request minimum and median for both clusters are then reported on the `S7REQ` line
(`cur0min= cur0med= cur6min= cur6med=`). **This is the whole point of the change**: an idle sample
sits at the floor even in the high-performance profile, so a between-requests reading would
false-positive on every healthy arm. `s10_endurance.sh`'s `last_temp()` now surfaces the same fields.

### 2.2 The `LOW-POWER-PROFILE` diagnosis in `s8_ll_table.py`

An arm is flagged when its **in-window** `cur_freq <= 1200000` while `scaling_max_freq` reads
at-profile (`>= 3000000`). It is a **configuration diagnosis, not a validity verdict** - it cannot
mark an arm INVALID, and its section says so. Logs from before this change carry no `cur` samples and
render:

```
*No in-window `scaling_cur_freq` samples in these logs* (pre-Stage-11 format):
the clock witness was not measured, which is not the same as healthy.
```

### 2.3 Acceptance evidence, both directions

Synthetic demo (`.stage11/demo-lowpower/`), two arms with identical caps, different in-window clocks:

| phase | pass | cfg | requests at <=1.2 GHz with max at-profile | reading |
|---|---|---|---|---|
| decode | 1 | LOW | 6/6 | **LOW-POWER-PROFILE** - OS-held low clock that the max does not show |

```
*2 arm(s) carried in-window clock samples; 1 flagged.*
```

The `OK` arm (cur 2841600/2918400 at the same max) is **not** listed. `s8_ll_table.py --selftest` now
carries four assertions for this and passes **30/30**:

```
PASS  low-power-profile flag fires on a low-cur/high-max arm
PASS  low-power-profile flag does NOT fire on a normal-clock arm
PASS  the low-power flag does not mark the arm INVALID
PASS  a log with no cur samples says so rather than claiming health
SELFTEST PASSED
```


### 1.4 Build and the mandatory native-byte diff gate — PASS

| step | result |
|---|---|
| version | `versionCode 13 -> 14`, `versionName "0.3.2-warmup-chip"` |
| `compileDebugKotlin --offline` | **BUILD SUCCESSFUL** (18 s) |
| `assembleDebug --offline` | **BUILD SUCCESSFUL** (7 s), 121,059,030 B (+1,248 vs Build 13) |
| artifact | `apk-out/MoeDroid-0.3.2-warmup-chip-14-arm64.apk` |
| **native diff gate** | **PASS - not one native byte changed** (below) |
| UI delta in dex | `warming (%d/3) · %s` present in Build 14, **absent in Build 13** (0 hits) |

The gate the brief makes mandatory, comparing every `lib/arm64-v8a/*` entry inside the two APKs:

| library | sha256 (first 16) | verdict |
|---|---|---|
| `libllama.so` | `a8c15f409cc25e88` | identical (still the shipped V1 build) |
| `libggml-cpu.so` | `119a61d7b80219fd` | identical |
| `libggml.so` | `95fb1e4b888b5258` | identical |
| `libggml-base.so` | `0c5928a09a94bff4` | identical |
| `libllama-common.so` | `f381ef81d6b86dd4` | identical |
| `libllama-server-impl.so` | `f1a9d154c364af00` | identical |
| `libllama-cli-impl.so` | `9d7fef378e65087c` | identical |
| `libllamaserver.so` | `8c62716220903936` | identical |
| `libmtmd.so` | `06351bc2e946fc5e` | identical |
| `libbonsai_native.so` | `aa33549857a5281d` | identical |
| `libc++_shared.so` | `ab4e6c71b96b851d` | identical |
| `libdatastore_shared_counter.so` | `d3e48717c9aa147e` | identical |
| `libandroidx.graphics.path.so` | `41e9a793c43a0f4f` | identical |

No library exists in 14 that was not in 13. The build picked up the right tree.

### 1.5 Install and the clock-witness probe — blocked on the phone

The phone dropped off USB at 15:0x and has not re-enumerated, so **`adb install -r` and the
`versionCode=14` confirmation are not done**, and the witness probe (§1.3) has not run. Deliberately
**no witness code ships in the APK that is waiting**, because the brief's rule is verify-first: a chip
that cannot read the clock must not exist, and one that paints "ok" while meaning "unknown" is worse
than nothing. When the phone returns the order is: probe -> (add witness and rebuild only if
readable) -> install -> confirm.

---

## 2.4 The Step 3 soak driver (`s11_durability_soak.sh`) — written, self-tested

`s10_endurance.sh` exists and was extended for the telemetry, but it runs 20 decode requests and has
no battery-temperature abort, so Step 3's protocol (40 continuous requests + R1) needed its own
driver. `MoeDroid/tools/s11_durability_soak.sh` carries **R1-R5 verbatim in its header**, enforces R1
with a 3 s watchdog on battery temperature that kills the arm and the engine on breach, runs the two
arms paired with a full cool-down between them, and records per-request t/s + `cur`/`max` clocks for
both clusters + `soc0/soc1/cpuss` + battery temperature.

Two defects were caught **before** it could run on the phone, both by exercising it on synthetic arms:

1. a positional field offset that would have mislabelled the battery maximum as the request count;
2. a branch-logic gap: when **ik suppresses and ours never does**, the first version fell through to
   R4. An arm that never suppresses cannot be the shorter one, so that case is R3.

Branch table, proven on synthetic arms (8/8, including the 5-vs-4 boundary that R2 defines):

| case | expected | got |
|---|---|---|
| ours never, ik at 20 | R3 | R3 |
| ours never, ik at 40 | R3 | R3 |
| both never within 40 | R5 | R5 |
| ours at 20, ik never | R4 | R4 |
| ours at 30, ik at 25 (5 earlier) | R3 | R3 |
| ours at 30, ik at 26 (4 earlier) | R4 | R4 |
| ours at 20, ik at 30 | R4 | R4 |
| ours at 39, ik at 35 (4 earlier) | R4 | R4 |


---

## 3. Step 3 — cold-start fanless durability soak: **R3 fires, the fork outlasts ik**

Run 23:14 -> 00:15 on a 100 W charger, **external fan OFF** (owner-asserted: software cannot see a
fan), Game Space active, app foregrounded, caps `3532800/3840000`.

### 3.1 Preconditions, and the two attempts that were thrown away

| gate | result |
|---|---|
| caps at profile | held 3 polls before the run |
| battery | **96%** at start (brief asks 100%; warned and recorded, not silently accepted) |
| thermal settle | **SETTLED at 31.0-32.0 C after 25 min** of a rolling 10-sample plateau gate |
| arm A baseline | **31.0 C battery / 39.1 C hottest** |
| arm B baseline | **32.0 C battery / 38.7 C hottest** after an 18-minute fanless cool-down |

Two attempts produced no result and were **aborted rather than published**:

1. the script was not executable (died in 2 s, no data);
2. **the cool-down gate was too lenient.** It required only "battery <= 40 C", so arm B (ik) started
   at **40.0 C / SoC 48 C** while arm A started at 29.0 C / 36 C - an **11 C head start** on the exact
   quantity being compared (time-to-first-suppression). The run was killed, the gate rewritten to
   require **baseline + 1.0 C, stable across two samples**, and the settle gate rewritten to wait for
   a *rolling plateau* rather than a low-but-falling reading.

### 3.2 The two arms (40 continuous decode requests, `WARM_N=0`, `WARM_COOL_S=0`)

| metric | ours (V1, no overrides) | ik (`build-arm64-ik`) |
|---|---|---|
| first suppression | **request 29** | **request 21** |
| requests at profile before it | **28** | 20 |
| caps actually fell at | request 30 | request 22 |
| suffix at profile (from the end) | 28 | 20 |
| t/s, median while at profile | **6.97** | 9.36 |
| t/s, median while suppressed | 5.96 | 8.54 |
| first five t/s | 5.38, 7.25, 6.98, 7.07, 7.20 | 4.26, 9.81, 9.72, 9.71, 9.34 |
| last five t/s | 5.97, 6.00, 5.97, 5.98, 6.01 | 8.07, 8.13, 8.07, 8.12, 8.15 |
| hottest `cpuss` | **82.2 C** | **90.3 C** |
| battery max | **41.0 C** | **43.0 C** |
| `cur0` median / min | **3532800 / 2400000** | 3427200 / 2227200 |
| R1 aborts | none | none |

Both arms paid the same cold-start shape (request 1 at 5.38 / 4.26 t/s, then steady), so the
suppression numbers are not a cache artefact. Neither arm aborted on R1: the battery peaked at
41-43 C, inside the 45.0 C safety bound.

### 3.3 The reading, per the pre-registered rules

```
S11ARM tag=s11-ours40 first_suppression=29 hottest_cpuss=82200 suffix_at_profile=28 battery_max=410 n_req=40 cur0_min=2400000
S11ARM tag=s11-ik40   first_suppression=21 hottest_cpuss=90300 suffix_at_profile=20 battery_max=430 n_req=40 cur0_min=2227200
S11R2 ik_materially_shorter=YES (ik suppressed at req 21, ours req 29)
S11READING=R3 durability case for the fork strengthened; ik archived as fan-required / fanless-limited
```

**R2 holds: ik suppressed 8 requests earlier than ours (>= the 5-request, 12.5% bar), from a matched
start.** **R3 therefore fires: the durability case for keeping our fork is strengthened, and ik is
archived as fan-required / fanless-limited at request 21.**

Three things the series say beyond the verdict:

* **ik is still much faster, in both regimes** (+34% at profile, +43% once both are throttled). This
  reading does not give the fork a speed advantage; it gives it a *thermal* one.
* **ik pays for its speed in heat:** 8.1 C hotter on `cpuss` and 2.0 C hotter on the battery under the
  same 40-request load from the same starting temperature.
* **the clock witness was informative at profile:** ours held a `cur0` median of exactly 3532800 while
  ik's was 3427200 - i.e. ik's cores were not pinned at the cap even before the OS dropped the caps.
  That is the `scaling_max_freq`-blind effect §2.1 exists to expose, and it is the first time the new
  instrument produced a number that the old one could not.

Per the brief this has **no adoption authority**: it cannot change a delta that already exists against
shipping V1, and it did not. It records our fork's time-to-first-suppression (29), ik's (21), and both
thermal margins.


---

## 4. Step 4 — `BONSAI_MOE_X4=2` on the shipped V1 baseline: **REJECTED**

Fan **ON** (owner-asserted), Game Space active, caps `3532800/3840000`, AC 100%, app foregrounded.
`ARMS="A0 M2 A0T"`, `WARM_N=12`, `WARM_COOL_S=90`, POSTCOOL cap gate, mirrored decode x2 and
prefill x2. `A0` = shipped defaults (V1 + `MOE_X4=1`), `M2` = V1 + `BONSAI_MOE_X4=2`, `A0T` = the
shipped defaults repeated at the tail of each pass.

### 4.1 Decode — the decision is made here, and it is a rejection

| arm | pass 1 | pass 2 | Δ vs that pass's `A0` | |
|---|---|---|---|---|
| `A0` | 6.9454 | 6.9455 | — | reference |
| `M2` | 6.9021 | 6.9811 | **−0.62% / +0.51%** | **not positive in both** |
| `A0T` (tail bracket) | 6.9408 | 6.8288 | −0.07% / −1.68% | bracket |

**Bar (i) FAILS**: `M2` is negative in pass 1 and positive in pass 2, so the mirrored requirement
("positive vs `A0` in BOTH passes") is not met. The bracket is what makes that trustworthy rather than
noise: `A0T` reads −0.07% and −1.68%, i.e. the pass-to-pass position effect is of the same order as
`M2`'s own movement, and `M2` crosses zero between the two passes instead of sitting above it.

**Bar (iii) also FAILS**: `M2` must beat `A0T` in both passes. It loses pass 1 (6.9021 vs 6.9408,
−0.56%) and wins pass 2 (6.9811 vs 6.8288, +2.23%) - again a sign flip.

### 4.2 Bar (ii) — the profiler pair could not be made valid tonight

Two attempts, both refused by the pre-registered same-engine canary (`RMS_NORM` + `GATED_DELTA_NET`,
2.5%; `MUL_MAT_ID` and `GET_ROWS` excluded because `MOE_X4` and V1 respectively move them):

| attempt | `RMS_NORM` | `GATED_DELTA_NET` | worst | reading |
|---|---|---|---|---|
| 1 (cool, `WARM_COOL_S=90`) | **−26.11%** | −2.26% | 26.11% | canaries disagree in direction - the small op was noisy |
| 2 (`WARM_COOL_S=120`) | +3.84% | **+5.43%** | 5.43% | canaries agree: a real ~5% clock skew between the arms |

So **bar (ii) is NOT established** - not because `MUL_MAT_ID` moved the wrong way, but because no
matched-clock pair was produced. Per the pre-declared rule that is a failure of the bar, not a waiver:
the pair's `MUL_MAT_ID` delta is therefore **not quoted anywhere**.

The instrument lesson is worth recording: a 1 ms op (`RMS_NORM`) is a poor 2.5% canary across 6-minute
arms, and the two canaries disagreeing in *sign* is the tell that separates "noise" from "real skew"
(attempt 1 vs attempt 2). A future window should gate on a large invariant op (dense `MUL_MAT` is
73-81 ms/token and `MOE_X4` does not touch it) or run the pair from a longer cold soak.

### 4.3 Decision

**`BONSAI_MOE_X4` stays at 1.** Two of the three bars fail outright and the third was never
established, so nothing is flipped, `build-arm64-ship` is not rebuilt, no Build 15 exists, `jniLibs`
are untouched, and the patch is not regenerated. The old `+5.79%` figure was **not** used at any point:
it was measured against the pre-x4 baseline, and this comparison is against V1.


### 4.4 Prefill, and the consolidated three-bar verdict

| arm | pass 1 | pass 2 | Δ vs that pass's `A0` | vs `A0T` (same pass) |
|---|---|---|---|---|
| `A0` | 8.4184 | 8.3563 | — | — |
| `M2` | 8.5088 | 8.5131 | **+1.07% / +1.88%** | **−0.84% / −0.52%** |
| `A0T` | 8.5812 | 8.5578 | +1.93% / +2.41% | — |

`M2` is positive in both prefill passes, so the "neutral-or-better on prefill" half is satisfied - but
it **loses to the tail bracket in both prefill passes as well** (−0.84%, −0.52%), which is the same
sign pattern as decode: the `A0T` repeats of the shipped defaults were *faster* than `M2` in three of
four cells.

**All 12 arms are valid**: `ok (6/6)` everywhere, `throttled=0/6` on all six decode arms, and the
prefill arms read `3-4/6` (the documented prefill regime, quoted with that column).

| bar | requirement | measured | verdict |
|---|---|---|---|
| (i) | positive vs `A0` in **both** mirrored decode passes; neutral-or-better on prefill | **−0.62% / +0.51%** decode; +1.07% / +1.88% prefill | **FAIL** (decode sign flip) |
| (ii) | `MUL_MAT_ID` ms/token negative under a **passing** canary | canary failed twice (−26.1% noise, then +5.43% real skew) | **NOT ESTABLISHED** |
| (iii) | beats the `A0T` bracket in **both** passes | decode −0.56% / +2.23%; prefill −0.84% / −0.52% | **FAIL** (loses 3 of 4 cells) |

```
S11M2 VERDICT = REJECT
```

**`BONSAI_MOE_X4` remains 1.** No default flip, no rebuild, no Build 15, no `jniLibs` change, no patch
regeneration. `BONSAI_MOE_X4=2` stays available behind its env var and its quality clearance from Stage
10 stands; what tonight establishes is that it does not pay for itself on top of V1.

**One instrument correction came out of this matrix, and it matters.** The new `LOW-POWER-PROFILE`
flag initially fired on 6 of 12 arms, which would have suggested the whole run was clock-suppressed.
Reading the raw samples showed the cause: **cpu0 sat at 3532800 in every decode request while cpu6
sat at ~1.02 GHz** - and cpu6 is *outside* the decode mask (`0x3f` = cpu0-5), so its idle floor is
correct behaviour. The rule judged "either cluster" and therefore flagged healthy arms. It now judges
**cpu0 only**, the one cluster inside every mask used here (0x3f and 0xff), and the matrix then reads
**12 arms with samples, 0 flagged** - i.e. the arms really did run at profile. A selftest case pins the
false-positive shape (cpu0 healthy + cpu6 at its floor) so the rule cannot regress into it.


---

## 5. Phone release

| item | state |
|---|---|
| harness engines | **none** (`ps` shows no `libllamaserver`; the soak and the matrix each stop their own) |
| port 8299 | **no listener** |
| `stay_on_while_plugged_in` | **0** (reverted, as the brief requires) |
| screen timeout | 60 s (restored) |
| adaptive battery management | 1 (restored) |
| battery | **86%, charging** |
| battery temp | 34.0 C |
| caps at hand-back | 2227200 / 1958400 - the OS drops to these when the app leaves Game Space, which it does as soon as the screen sleeps |
| keyguard | **showing** (`isKeyguardShowing=true`, PIN set). Reverting `stay_on` is what lets the screen sleep and the keyguard re-engage, so this is the expected consequence of the instructed release, not a fault. `com.prismml.moedroid` is the last foregrounded activity and returns on unlock. |

Recording the caps drop explicitly because it is the same mechanism the window's ground truth calls
out: `scaling_max_freq` reads *not* at profile the moment the app is not in Game Space, and no run in
this report was measured in that state (every arm's own cap gate and per-request samples prove the
profile held during measurement; the released state is simply idle).

## 6. Summary

| step | outcome |
|---|---|
| **1. Build 14** | warmup counter with the specified reset semantics + `warming (N/3)` chip, verified live (`(1/3)` -> `(2/3)` -> plain); clock witness **feasible** (in-app probe `readable=1`, `run-as` was a false lead) so the low-clock hint shipped, rendering nothing at a healthy clock; **native diff gate PASS, 13/13 libs identical to Build 13**; installed `versionCode=14` |
| **2. instrument** | in-window `scaling_cur_freq` (cpu0+cpu6, 1 Hz *during* each request) + `LOW-POWER-PROFILE` diagnosis + 4 selftest assertions; **30/30** |
| **3. fanless soak** | **R3 fired**: ours suppressed at request **29**, ik at **21** (8 earlier, >= the 5-request bar) from matched baselines; ik stays faster (+34% profile, +43% throttled) and runs 8.1 C hotter on `cpuss`; ik archived as fan-required / fanless-limited |
| **4. `MOE_X4=2` on V1** | **REJECTED**: bar (i) fails (decode −0.62% / +0.51%), bar (iii) fails (loses to the `A0T` bracket in 3 of 4 cells), bar (ii) **not established** (canary failed twice). `MOE_X4` stays 1; nothing shipped |

**Two instrument defects were found and fixed by using the instruments on real data** - the false
`LOW-POWER-PROFILE` flags caused by judging the idle cpu6 cluster, and the settle/cool-down gates that
would have compared two arms from different starting temperatures. Both are recorded with the evidence
that exposed them rather than quietly corrected.

**The fork's position after this window:** V1 is shipped (Build 14) and worth +4.30%/+6.71% on decode
from Stage 10; the fanless durability case now favours the fork over ik at equal starting temperature;
and `MOE_X4=2` is a measured dead end on top of V1 rather than an unmeasured candidate.


---

## 7. Final close-out and archive (host-only, zero ADB)

Campaign ratified complete at **Build 14**. This section seals the archive and records the two places
where the close-out instructions did not match the tree.

### 7.1 Build 14 artifacts verified against disk (not asserted)

| artifact | recorded | on disk | match |
|---|---|---|---|
| `apk-out/MoeDroid-0.3.2-warmup-chip-14-arm64.apk` size | 121,438,632 | **121,438,632** | yes |
| its `sha256` | `cfbb6fc2a7f93aa204a8ceeebf7271e00e890fb031f4ac1cbd7ba3eb782bfdba` | **identical** | yes |
| packaged `libllama.so` | `a8c15f409cc25e88…` | **`a8c15f409cc25e88e1c32854ef8a5203cb158ae9…`** | yes |
| packaged `libggml-cpu.so` | `119a61d7b80219fd…` | **`119a61d7b80219fdddb7e91f6c91e80e63c1616a…`** | yes |
| `llama.cpp-moedroid.patch` (both repos) | 155,923 B | **155,923 B both, byte-identical** | yes |
| `ik_llama.cpp-moedroid.patch` (both repos) | 34,271 B | **34,271 B both, byte-identical** | yes |

**Nothing was rebuilt and no ADB command was run** in this pass.

### 7.2 `GOAL.md` — the archive section is in

`## Final Campaign Archive — Ratified Complete at Build 14 (2026-10-20)` appended (GOAL.md 362 -> 499
lines), carrying all seven blocks: artifact identity + the withdrawn `"Day-1 4.4 -> 7.48 = +70%"` claim
and why it is forbidden, the four canonical numbers with the composite self-validation
(`5.61 x 1.234 = 6.93` vs **6.945 measured**, 0.2% agreement), the `~3%` noise floor with its bracket
receipts, the 13-row negatives table with first-click evidence, the four-currency ledger, the four
gates, the bracket-over-mirror and pre-registered-bar rule, the six protocol lessons, the three named
re-open triggers, and both honesty items (no valid per-op budget for the shipped V1; the two launch-flag
difference and its bound).

### 7.3 Three handoffs stamped `HISTORICAL` — with one path correction and one clarification

The banner was prepended to all three files, and each grew by exactly the banner's 515 bytes with the
banner on line 1:

| file | before | after |
|---|---|---|
| `HANDOFF.md` (workspace root) | 7,513 B (10-05) | **8,028 B** |
| `MonsterQwenDroid/docs/HANDOFF.md` | 43,419 B (10-15) | **43,934 B** |
| **`MonsterQwenHandoff.md` (workspace root)** | 43,419 B (10-15) | **43,934 B** |

**Correction 1 (path):** the instructions placed the third file at
`MonsterQwenDroid/MonsterQwenHandoff.md`. **No such file exists.** The real one is
`/home/kingfish600/workspace/MonsterQwenHandoff.md` — the path `GOAL.md` itself references as
`../MonsterQwenHandoff.md` — and that is the file stamped.

**Clarification 2 (the duplicate claim):** "MonsterQwenHandoff.md and docs/HANDOFF.md are byte-identical
duplicates" is **true** — verified with `cmp`, and still true after both were stamped with the same
banner. But the *root* `HANDOFF.md` (7,513 B, 10-05) is a **different, older file**, not a third copy.
So the tree holds **two distinct handoff artifacts, one of them duplicated**: the 7.5 KB root file, and
the 43 KB pair. The banner's wording covers this correctly.

### 7.4 README refreshed, and the MTP closure line

* `README.md` decode row replaced: `4.4 t/s at full quality` -> **`6.95 t/s` benched, same protocol as
  the 5.61 baseline (+23.4%)**, plus a separate warm row **`7.48 t/s` with its basis in parentheses**
  ("n=3 consecutive warm requests, single session; not a mirrored-pass measurement; cold request 1 read
  5.43"). Prefill row corrected to the measured `8.4-8.6 t/s` on a 262-token prompt.
* **Also updated (beyond the literal ask, for consistency):** the README status line still read
  `0.1.0-alpha`; it now reads `0.3.2 (Build 14, versionCode 14)` with a pointer to the archive. A cold
  reader would otherwise see a 0.1.0 status beside a 0.3.2 APK.
* **MTP closure line:** no pending marker existed in the docs (`grep` for a TODO/closure line found
  none). Its closure is recorded in the archive §2.2 with both kill receipts rather than appended to
  the handoffs, which are now frozen `HISTORICAL` - editing frozen documents would contradict the stamp.

### 7.5 Seal

| item | state |
|---|---|
| `GOAL.md` archive | **sealed** (499 lines) |
| `STAGE11-report.md` | **§1-§7 complete** |
| `iterations.jsonl` | **through 97** |
| three handoff banners | **sealed**, line 1, +515 B each, duplicate pair still identical |
| `README.md` | **refreshed** to Build 14, basis stated |
| ADB | **not used** this pass |

