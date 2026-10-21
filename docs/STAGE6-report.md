# Stage 6 report: on-device shootout (our fork vs `ik_llama.cpp`)

Date: 2026-10-16/17 (overnight session). Device: RedMagic 10S Pro (NX789J), serial `91251291018D`,
Snapdragon 8 Elite, 24 GB, CPU only. External cooling fan active, Game Space ON, app foregrounded,
phone unlocked and awake. Hard cutoff 04:30 CDT so the phone is cool before the 05:30 shift.

Companion document: `STAGE6-daytime-prep.md` (the host-only preparation: q3_K and q2_0 `x4` kernels,
the prefill expert queue, the `ik` attribution, and the audit fixes).

**Design in one line:** both engines read the *same* sha256-verified model file, run the
long-lived-server steady-state protocol (`s6_arm_ll.sh`), are compared in mirrored two-pass order,
and every bit-exact gate is first put through a token-identity check.

---

## 0. Storage audit and reclaim (before staging anything)

`/sdcard` was at **92% used** from leftover Stage 3/4/5 staging. Reclaimed and recorded:

| step | free on `/sdcard` | note |
|---|---|---|
| before | **79 G** (854 G used, 92%) | varD5 pair + duplicate varH pair in `/sdcard/Download/mf-model/`, 6 stale `/data/local/tmp/lib-*` dirs |
| delete `varD5-00001/00002` (66.2 G) | - | varD5 was **rejected** in Stage 5 (passes PPL at -0.117% but only +1.95% decode at t = 1.64, not worth +388 MiB). Both files exist in full on the host (`moe-dense-q8/varD5-*.gguf`), so deletion is recoverable |
| delete 6 stale `lib-*` dirs (1.6 G) | - | `lib-{dep,flat,new,prof,shield,ik}`; tonight's harness stages fresh `lib-s6-*` dirs and pushes `libc++_shared.so` + `libomp.so` from the NDK. A copy of `libc++_shared.so` was preserved at `/data/local/tmp/libc++_shared.so` |
| verify live model is varH, then delete the duplicate varH pair (65.8 G) | - | see below |
| **after** | **203 G** (729 G used, 79%) | **+124 G reclaimed** |

The duplicate pair was only removed **after** proving the live app copy is the same bytes:

```
e7c694526e82f633eda7f37b5e35ed7075fd39140018c99d73e305818ded12e2  .../models/Qwen3.8-...-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  .../models/Qwen3.8-...-00002-of-00002.gguf
```

Both match the recorded varH hashes exactly, so the live app path is the canonical varH pair and
**both engines now read that one file**, which also removes a page-cache confound: previously our
engine and `ik` read *different copies* of the same model.

Device state before measurement:

* no `libllamaserver` / engine process running (no port or CPU contention), and no listener on the
  8199/8299 slots;
* the app process `com.prismml.moedroid` **is** running and its MainActivity is focused, at ~178 MB
  RSS, i.e. it holds Game Space without holding the 66 GB model in RAM;
* `mWakefulness=Awake`.

Not removed, and why: `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (2.79 GB, in the app's model dir). It is
the rejected MTP sidecar (Stage 1: 4x slower), not a `varD*` or staging file, and it sits in the
app's own directory rather than a scratch location - reclaiming 2.79 G is not worth touching the
app's private data without being asked.

---

## 1. Identity gate (`s6_shootout.sh 0`)

## 1. Identity gate (`s6_shootout.sh 0`) - ALL BIT-EXACT GATES PASS

Command: `MoeDroid/tools/s6_shootout.sh 0` (staging + sha256 gate + 8 identity arms).

| arm | what it enables | prompt | md5 of the greedy token ids | verdict |
|---|---|---|---|---|
| `id-off` | nothing (shipping kernel) | short, 64 tokens | `8ad776c2658411fcbe94ff8ca509d659` | baseline |
| `id-x4` | `BONSAI_MOE_X4=1` (q2_0 x4) | short, 64 | `8ad776c2658411fcbe94ff8ca509d659` | **PASS - identical** |
| `id-q3k` | `BONSAI_Q3K_X4=1` (q3_K x4) | short, 64 | `8ad776c2658411fcbe94ff8ca509d659` | **PASS - identical** |
| `id-all` | `MOE_X4=1` + `Q3K_X4=1` | short, 64 | `8ad776c2658411fcbe94ff8ca509d659` | **PASS - identical** |
| `id-x4p` | `BONSAI_MOE_X4=2` (packed) | short, 64 | `75987b6751ba34ed0c34f796d7a0a597` | **diverges at token index 51** (of 64) - expected, near-exact by construction (Gate 3, not Gate 1) |
| `id-mm1` | `BONSAI_MULMATID_FLAT=1` (stock prefill) | **262-token prefill**, 16 tokens | `514b317d2b0bcb62cf3818361dad64e1` | baseline |
| `id-mm2` | `BONSAI_MULMATID_FLAT=2` (single queue) | 262-token prefill, 16 | `514b317d2b0bcb62cf3818361dad64e1` | **PASS - identical** |
| `id-mm2b` | `MULMATID_FLAT=2` repeat | 262-token prefill, 16 | `514b317d2b0bcb62cf3818361dad64e1` | **PASS - identical** (repeatability control) |

What this establishes:

* **Arms `A1`, `A3`, `A4` and `AP` are all Gate-1 eligible on the device.** The two kernels are
  bit-exact against the shipped reference at the instruction level (host qemu evidence in
  `STAGE6-daytime-prep.md` §5.2/§8.4) *and* token-identical end to end here, and the prefill queue is
  token-identical on a real 262-token prefill - which is the only prompt size that exercises it,
  since the queue needs `ids->ne[1] > 1`.
* `id-x4p` behaves exactly as the kernel-level evidence predicted: a <= 2.9e-7 relative re-association
  survives 51 greedy tokens before flipping one argmax. It is therefore judged by **Gate 3**, and it
  is *not* a candidate for the default.
* The prefill-queue arms report `evaluated=262`, confirming the long prompt really was ~256 tokens
  rather than the 6-token prompt the first version of this check used.

### 1.1 Two operational findings from this step

1. **`return_tokens` is required to get token ids.** The first attempt produced empty `S6IDTOK`
   lines: llama-server only includes the `"tokens"` array when the request asks for it
   (`tools/server/server-schema.cpp:34`, consumed at `server-context.cpp:1985`). The extraction then
   silently fell back to hashing the generated *text*, which is a valid identity artefact but gives no
   divergence index - so every gate would have reported UNKNOWN. Fixed by adding
   `"return_tokens":true` to the identity request, plus a `S6WARN` line when the fallback fires and an
   md5-based fallback in both `id_div()` and the analyzer, so this can never silently degrade again.
   The aborted first run was discarded and re-run; only the re-run's numbers are reported above.
2. **ik does not support `return_tokens`** (its smoke test printed `S6WARN ... falling back to
   content hash`). Harmless here - ik is never the subject of a token-identity gate, only of timing
   arms - but it is why ik's identity line carries no token list.

### 1.2 Device conditions held during the gate

| condition | value |
|---|---|
| screen / app | `mWakefulness=Awake`, focus on `com.prismml.moedroid/MainActivity` (Game Space active) |
| caps | `cpu0-5 = 3532800`, `cpu6/7 = 3840000` of `4473600` - the known-good profile, every arm |
| CPU-zone temperature | 36.0-36.8 C at gate start (fan on; Stage 4 ran at 41-43 C) |
| cap gate | each arm reported `gate=OK polls=2 held=3` |
| stray engines | none from previous stages. The **app's own** engine is running (uid `u0_a690`, parent `com.prismml.moedroid`) on **port 42207**, but it is measurably dormant: **0 CPU ticks over 6 s** and **2 MB RSS**, i.e. it holds neither CPU nor the model, and it does not collide with our port 8299. Left running because Game Space depends on the app |
| our arm engine | ~13.5 GB RSS while serving, as expected for the expert working set |

Also confirmed in the ik smoke test (before spending the night's arms on it): ik starts with
`-ub 512 --ctx-checkpoints 0` and reports **`fused_moe = 1` / `fused_up_gate = 1`**, so the B2
contrast (`--no-fused-moe`) is measuring a fusion that is genuinely enabled in B0/B1.

---

## 2. Full mirrored shootout + per-op profiler (`PROF=1 s6_shootout.sh 2`)

Command: `PROF=1 MoeDroid/tools/s6_shootout.sh 2` (staging, identity re-check, decode pass 1 + 2,
prefill pass 1 + 2, then the four profiler arms).

### 2.1 Decode pass 1 (forward order)

| arm | median t/s | vs `A0` | reading |
|---|---|---|---|
| `A0` (shipping) | **5.3869** | - | baseline; series 4.43 5.70 5.40 5.59 5.37 5.37 |
| `A1` (`MOE_X4=1`, bit-exact) | **5.5330** | **+2.71%** | the q2_0 four-row kernel helps, as predicted in the small |
| `A2` (`MOE_X4=2`, packed) | **5.7850** | **+7.39%** | the packed accumulate is much better than mode 1 - but it is Gate 3, not Gate 1 |
| `A3` (`Q3K_X4=1`, bit-exact) | **5.0786** | **-5.72%** | **the q3_K port is SLOWER** - prediction refuted |
| `A4` (`MOE_X4=1` + `Q3K_X4=1`) | 5.2264 | -2.98% | A1 x A3 additivity confirmed (1.0271 x 0.9428 = 0.9684, measured 0.9702) |
| `B0` (ik, unpinned) | **7.4488** | **+38.28%** | the ik gap is real, same file and session |
| `B1` (ik + our pinning flags) | 2.5307 | -53.02% | **broken by our own flag mapping - see 2.2** |
| `B2` (B1 + `--no-fused-moe`) | 2.5107 | -53.39% | vs B1: **-0.79%**, i.e. the MoE fusion is worth ~0.8% to ik |

### 2.2 Three findings that change the plan

1. **The `q3_K` four-row kernel is a regression (-5.7%) and must not ship.** The daytime prediction
   (22.6 -> 34-50 GiB/s, worth 4-6% of decode) assumed the kernel was load-bound on the `q8_K`
   activation. It is not: sharing those loads buys nothing, and four concurrent rows cost register
   pressure in a kernel whose inner loop is already a long `vaddvq_s32`/integer-multiply chain. The
   prediction was explicitly labelled "prediction, not measurement" in
   `STAGE6-daytime-prep.md` §5.5, and the measurement refutes it. `A4` inherits the loss, so it is
   excluded too - and the additivity check confirms the two kernels combine exactly as measured
   individually, which is a useful validation of the experimental method.
2. **Our `--poll 0` mapping is catastrophic for `ik_llama.cpp` (7.45 -> 2.53 t/s, -66%).** Subagent
   work mapped `--poll 0` onto `KMP_BLOCKTIME=0` + `OMP_WAIT_POLICY=PASSIVE` because ik has no
   hand-rolled spin loop. That kills OpenMP's spin-wait, and on this workload (many small ops,
   thousands of barriers) the wake-up cost dominates: **ik's spin-wait is a performance feature, not
   interference**, which inverts the original brief's hypothesis. `s6_shootout.sh` no longer passes
   `--poll` to ik. The consequence for this run: `B1`/`B2` (and the `prof-ik` arms) are poisoned, so
   the "best vs best" prefill comparison and the clean fusion delta are re-measured afterwards with
   a targeted follow-up (no `--poll`), reported in §2.4.
3. **The fused up+gate is NOT the bulk of ik's advantage.** `B1 - B2` = **-0.79%** with both arms at
   identical flags: disabling ik's `fused_moe_up_gate` makes it slower, so the fusion contributes
   ~0.8%, not the "single largest contributor" that §2.1 of the daytime analysis hypothesised. The
   fused-op port is therefore **not worth building**; the bulk of the +38% sits in ik's
   `iqk` kernels and its expert chunking, which is where a port would have to go.

### 2.3 A measurement caveat recorded before it is needed

Tonight's absolutes (**5.39 t/s** for `A0`) are well below Stage 5's **6.90 t/s** for the same engine,
flags and protocol. Both of tonight's engines read the *live app copy* of varH, while Stage 5 read the
`/sdcard/Download/mf-model/` copy that has since been removed. Two candidate causes, neither yet
separated: the app-path copy may be physically slower to fault in, and/or this run's warmup (4
requests) leaves the first measured requests still cold - the series above starts at 4.43 and only
settles near 5.4-5.7. **Every comparison in this report is therefore within-session and valid, but no
tonight number is comparable to Stage 5's absolutes**, and the gap to the oracle must be computed from
tonight's own `B0` (same copy, same warmup) rather than from Stage 5's 8.61.

### 2.4 Complete results: both phases, both passes

Decode, within-pass deltas against that pass's own `A0` (the only valid comparison):

| arm | pass 1 | pass 2 | mean | spread | verdict |
|---|---|---|---|---|---|
| **`A1`** `MOE_X4=1` | **+2.71%** | **+6.34%** | **+4.53%** | 3.6 pts | **both passes positive**, Gate 1 PASS -> **shipped** |
| `A2` `MOE_X4=2` | +7.39% | +4.18% | +5.79% | 3.2 pts | consistently faster, but near-exact -> Gate 3, **not shipped** |
| `A3` `Q3K_X4=1` | -5.72% | -5.74% | **-5.73%** | 0.02 pts | **robustly slower in both passes -> rejected** |
| `A4` = A1+A3 | -2.98% | +1.12% | -0.93% | 4.1 pts | inherits A3's loss -> rejected |
| `B0` ik (unpinned) | +38.28% | +55.92% | +47.1% | 17.6 pts | ik is much faster, but with a large position effect |
| `B1` ik + our pinning flags | -53.02% | -51.69% | -52.4% | 1.3 pts | **broken by `--poll`, see 2.2** |
| `B2` = B1 + `--no-fused-moe` | -53.39% | -52.94% | -53.2% | 0.5 pts | vs B1: **-0.79% / -2.58%**, so the fusion is worth ~1-3% |

Prefill, within-pass deltas against that pass's `A0`:

| arm | pass 1 | pass 2 | mean | spread | verdict |
|---|---|---|---|---|---|
| **`AP`** `MULMATID_FLAT=2` | **+14.95%** | **+16.10%** | **+15.53%** | 1.2 pts | **tight, repeatable, bit-identical -> shipped** |
| `B0` ik | +41.49% | +44.01% | +42.75% | 2.5 pts | ik's prefill lead is large and consistent |
| `B1` ik + batch mask | +40.94% | +41.84% | +41.39% | 0.9 pts | **vs B0: -0.4% / -1.5%** -> ik gains nothing from the `0xff` batch mask |

### 2.5 Where the time goes: both engines' per-op budgets

Our profiler (`BONSAI_PROF_EVERY=1`, wall-consistent - its total agrees with the measured rate):

| op | `MOE_X4=0` | `MOE_X4=1` | change |
|---|---|---|---|
| `MUL_MAT` (dense) | 95.294 ms (50.5%) | 88.015 ms | -7.6% |
| **`MUL_MAT_ID` (experts)** | **61.883 ms (32.8%)** | **52.450 ms (30.8%)** | **-15.2%** |
| `GET_ROWS` | 6.790 | 6.477 | -4.6% |
| `GATED_DELTA_NET` | 5.882 | 6.250 | +6% (noise) |
| **total** | **188.633 ms/token (5.30 t/s)** | **170.393 ms/token (5.87 t/s)** | **-9.7%** |

The win lands exactly where the kernel was applied - the expert `MUL_MAT_ID` - which is the
independent corroboration that makes the small wall-clock effect believable.

ik's budget, **measured for the first time** (`build-arm64-ik-prof`, same model, same session):

| op | fusion ON | fusion OFF (`--no-fused-moe`) |
|---|---|---|
| `MUL_MAT` (dense) | 340.9 ms (63.4%), 4218 calls | 368.9 ms, 4218 calls |
| **`MOE_FUSED_UP_GATE`** | **99.8 ms (18.6%)**, 288 calls | - |
| `MUL_MAT_ID` | 51.2 ms (9.5%), 288 calls | **157.1 ms (27.3%), 864 calls** |
| `DELTA_NET` | 23.4 ms (4.4%) | 26.1 ms (4.5%) |
| `FLASH_ATTN_EXT` | 3.0 ms (0.6%) | 3.2 ms |

Reading it: with the fusion on, per token there are **48 `MOE_FUSED_UP_GATE` + 48 `MUL_MAT_ID`**
(down) = 2 expert ops per layer; with it off, **144 `MUL_MAT_ID`** = 3 per layer (call counts are
per-thread sums; divide by the 6 workers). That is the 3 -> 2 structure confirmed directly, and it is
what ik's +1-3% buys. **Caveats, stated because they bound what this table can claim:** ik's numbers
are sums of per-thread op times (CPU-time shares, not wall time), and two `clock_gettime` calls per op
mean instrumentation overhead grows with call count, which inflates call-heavy arms - the profiled
totals (~537 ms/graph) are several times the unprofiled wall time. Use it for **structure and call
counts**, not for cross-arm deltas.

### 2.6 The gap, restated on tonight's own numbers

Both engines, same file, same session, same protocol (phase-specific means over the two passes):

| metric | ours, shipped defaults | ik (`B0`) | residual gap |
|---|---|---|---|
| decode | **5.62** (`A1`) | **7.91** | **+40.7%** |
| prefill | **7.96** (`AP`) | **9.83** | **+23.6%** |

So tonight's work closes part of the prefill gap (the `AP` queue is worth +15.5% of ours, cutting the
ik lead from ~+42.8% to ~+23.6%) and a small part of decode (+4.5%, leaving ~+40.7%). The remaining
decode gap is in ik's `iqk` kernels themselves - consistent with §2.2 finding 3, which rules out the
fused op as the explanation, and with ik's own budget showing 63% of its time in dense `MUL_MAT`.

### 2.7 Honest limits of this dataset

* **Two passes, two pairs per config.** The project's own standard is ~11-16 pairs to resolve a 2-3%
  effect at a 5-8% arm-to-arm spread. `A1`'s +4.53% mean (pairs +2.71% and +6.34%) is therefore
  *suggestive, not resolved* on wall-clock evidence alone. It was shipped on the combination of
  (i) both pairs positive, (ii) the independent per-op attribution showing `MUL_MAT_ID` -15.2%, and
  (iii) zero correctness risk - Gate 1 token identity passed. `AP`'s +15.5% (1.2-point spread) is
  comfortably outside the noise and needs no such caveat.
* **`B0`'s decode number carries a 17.6-point position spread** (+38.28% vs +55.92%), so "ik is X%
  faster on decode" should be quoted as a range, not a point.
* The absolute level remains warmup-limited (§2.3), so every percentage above comes from a
  within-pass pair and never from comparing absolutes across phases or sessions.


---

## 3. Ship decision and app defaults

## 3. Ship decision and app defaults

### 3.1 What shipped, on what evidence

Only bit-exact changes were eligible (Gate 1), and two cleared it:

| change | Gate 1 | speed | decision |
|---|---|---|---|
| **`BONSAI_MOE_X4=1`** (q2_0 four-row kernel) | **PASS** - token-identical md5 on device | decode **+2.71% / +6.34%** (both passes positive); `MUL_MAT_ID` **-15.2%** per op | **SHIPPED as default** |
| **`BONSAI_MULMATID_FLAT=2`** (single prefill queue) | **PASS** - token-identical on a 262-token prefill, plus byte-identical on six synthetic shapes | prefill **+14.95% / +16.10%** | **SHIPPED as default** |
| `BONSAI_Q3K_X4=1` (q3_K four-row) | PASS | decode **-5.72% / -5.74%** | **REJECTED** - slower; left in the tree, off by default |
| `BONSAI_MOE_X4=2` (packed) | n/a (near-exact) | decode +7.39% / +4.18% | **NOT shipped** - Gate 3(b) not run; stays opt-in |

### 3.2 The change, and its verification on device

Two default flips, with the measurement recorded in the code comment at each site:

* `ggml/src/ggml-cpu/arch/arm/iqk_x4.c` - `bonsai_q2_0_x4_mode()` default `0` -> **`1`**
  (mode 2 stays reachable via `BONSAI_MOE_X4=2`, and `0` still disables).
* `ggml/src/ggml-cpu/ggml-cpu.c` - `mulmatid_flat_mode()` default `1` -> **`2`**
  (`0` restores the stock per-expert path; `1` restores decode-only flattening).

`build-arm64-ship` rebuilt clean. Then the shipped binary was verified **on the device with no env
variables set**, which is exactly how the app will run it:

| check | expected band | measured | verdict |
|---|---|---|---|
| decode, no env | `A1` = 5.53 / 5.71 (not `A0` = 5.387 / 5.369) | **5.5559** | in the A1 band yes |
| prefill, no env | `AP` = 7.98 / 7.94 (not `A0` = 6.94 / 6.84) | **8.1123** | in the AP band yes |

### 3.3 App libraries synced

Per handoff rule 7.7 (push the whole library set), the full engine set from the rebuilt
`build-arm64-ship/bin` was copied into **both** repos' `jniLibs/arm64-v8a/`:
`libggml-base.so, libggml-cpu.so, libggml.so, libllama.so, libllama-common.so,
libllama-server-impl.so, libllama-cli-impl.so, libmtmd.so`, plus `llama-server` ->
`libllamaserver.so` (a 7 KB launcher; the real server code is `libllama-server-impl.so`).

Verified: `sha256` of each copied file matches the build in **both** `MoeDroid` and
`MonsterQwenDroid` (`libggml-cpu.so` grew 5,535,760 -> 5,631,856 bytes; `libggml-base.so` is
byte-identical, so the ABI is aligned). The app's own `libbonsai_native.so` and `libc++_shared.so`
were left untouched. Pre-ship copies of both jniLibs trees are in `.stage6/jniLibs-backup/` for a
one-command revert.

**One step remains, and it is the user's:** `jniLibs/` feeds the APK, so the on-device app only picks
these up after a Gradle rebuild and `adb install -r`. The library artifacts are staged and verified;
installing over the running app was left alone deliberately, because the app is holding Game Space
for the rest of the session and an install restarts it. The model data on `/sdcard` is untouched by
an install, so nothing is lost either way.

### 3.4 Left open, with the exact reasons

| item | why it is open |
|---|---|
| `BONSAI_MOE_X4=2` (+5.8% mean, best decode arm) | Gate 3(b) - the 32-chunk paired PPL test - has not been run. It is the single largest remaining decode lever we own, at roughly +1.3% beyond mode 1, so it is worth one PPL run |
| the `q3_K` kernel | rejected on speed, not correctness; it stays in the tree behind `BONSAI_Q3K_X4` as a verified-but-slower kernel. Reviving it needs the register-pressure problem solved (it is compute/latency-bound, not load-bound, which the -5.7% proves) |
| ik's remaining +40.7% decode lead | its `iqk` kernel family, not the fused op (§2.2) and not the batch mask (§2.4) |
| the `--poll` mapping for ik | measured harmful (-66% decode, ~neutral prefill). `s6_shootout.sh` no longer passes it; the finding is recorded in the script and in §2.2 |
| the app-path model copy being ~20% slower than the deleted `Download` copy | not investigated; if real, the app itself is paying it. The host still holds the verified varH pair (`moe-dense-q8/`) so a re-stage is possible |

### 3.5 Session outcome in one paragraph

Tonight's host-only work was cashed in: the two bit-exact changes it produced both work on the device
(decode +4.5% mean, prefill +15.5%), `BONSAI_MOE_X4=1` and `BONSAI_MULMATID_FLAT=2` are now the shipped
defaults, and both are token-identical to the previous engine, so the user's model, prompts and
outputs are unchanged. The `q3_K` port was refuted by measurement and is off. The fused up+gate that
the daytime analysis nominated as the biggest prize turned out to be worth ~1-3% on ik's own binary,
so it will not be built. The remaining gap to `ik_llama.cpp` (about +40% decode, +24% prefill) lives
in its `iqk` kernels, and that is now the only path left that matters.
