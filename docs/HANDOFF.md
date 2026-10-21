**[HISTORICAL — FROZEN AT STAGE 5 (5.61 t/s) OR EARLIER (10-05). DO NOT USE AS CURRENT STATE. See MonsterQwenDroid/GOAL.md ("Final Campaign Archive — Ratified Complete at Build 14") and MonsterQwenDroid/docs/STAGE11-report.md for the final Build 14 state (5.61 -> 6.95 t/s benched, +23.4% decode / +45.0% prefill; 7.48 t/s warm in-app, n=3 single-session; 100% bit-exact; 28 fanless requests to thermal suppression). Note: MonsterQwenHandoff.md and docs/HANDOFF.md are byte-identical historical duplicates.]**

# MonsterQwenDroid — handoff

Written to survive a context reset. Everything here is measured unless marked otherwise.
Owner: github.com/kingfish600. Repo: github.com/kingfish600/MonsterQwenDroid (MIT).

## 1. The result

Qwen3.8-Flash-Next (512 experts, top-10 routed, ~6B active/token, 176B total) running fully
offline on a RedMagic 10S Pro — Snapdragon 8 Elite, 24 GB, 1 TB UFS, **CPU only, no NPU/GPU**.

| metric | value |
|---|---|
| decode | **4.6 t/s** at full quality (k=10) |
| prefill | **7.3-8.6 t/s** on a 966-token prompt (the older "2.9 t/s" was an 11-word prompt = fixed cost) |
| model | 66.4 GB, 2 shards, mmap + demand-paged |
| working set | ~12-15 GB of experts lives in a ~15.5 GB page cache (99.8% hit rate) |

## 2. What is shipped

- **Repo** `MonsterQwenDroid/` — commits `20e2b83` (app + patch + docs) → `01a63c3` (base revision
  named) → `162c712` (prefill correction + THP/flat results). Pushed.
- **Release** `v0.1.0-alpha` with `MonsterQwenDroid-0.1.0-alpha.apk` (121 MB,
  sha256 `9e04902e70ce3c28…`). NOTE: the GitHub release notes still say "~2.9 t/s prefill" — needs
  editing to 7.3-8.6 t/s. Only the owner's account can do it.
- **App identity:** `io.github.kingfish600.monsterqwendroid`, versionName `0.1.0-alpha`.
  The internal Java namespace is still `com.prismml.bonsaidroid` — JNI symbols embed the package,
  so renaming it would break the native binding for no user-visible gain.
- **On the phone:** the *old* package `com.prismml.moedroid` v0.2.0-pinned is what actually runs and
  holds the 66 GB model in `/sdcard/Android/data/com.prismml.moedroid/files/models/`.
  `MoeDroid/` is the source that matches it; `MonsterQwenDroid/` is the renamed public release.
  **Build/test from `MoeDroid/`; installing the new applicationId creates an app that cannot see
  the model.**

## 3. The engine patch

`MonsterQwenDroid/patches/llama.cpp-moedroid.patch` — 8 files, base **llama.cpp `1b70350`**
(`1b70350c92563449576bcbbf552994b1d1ed847d`), verified to reverse-apply cleanly (so it applies to a
clean checkout of that revision).

| change | effect |
|---|---|
| BF16 dot product had **no ARM SIMD path** upstream (AVX512/AVX2/SSE only) → added NEON `vshll` + hardware `vbfdotq_f32` | **+40% decode, +55% prefill** |
| cluster pinning `--cpu-mask 0x3f --cpu-strict 1` (stock flag, no patch) | **+26% decode** |
| `GGML_OP_MOE_SUM` — single-pass expert reduction vs 10 views + 9 adds per layer | −30% graph nodes, bit-identical |
| flat whole-expert work distribution in `mul_mat_id` (env `BONSAI_MULMATID_FLAT`, default ON since commit `9058ea6`) | **+4.8%, bit-identical**; major faults 719 → 318/token |
| flat static row slices in the dense single-token `mul_mat` (env `BONSAI_MULMAT_FLAT`, **default OFF**) | **neutral** (see §4.3); bit-identical, ships opt-in |

Env switches: `BONSAI_MOE_TOPK` (0=off; 8 = +8% for +3.3% PPL, 6 = +27% for +11.3%),
`BONSAI_Q20_UNROLL` (1=default), `BONSAI_MOE_PREFETCH` (0=default), `BONSAI_MULMATID_FLAT`
(1=default since `9058ea6`; `=0` restores the stock expert walk), `BONSAI_MULMAT_FLAT`
(0=default; `=1` enables the flat dense single-token `mul_mat`), `BONSAI_MULMAT_FLAT_REPORT`
(1 prints `flat path engaged` on the first flat call and a call count at exit),
`BONSAI_PROF`/`BONSAI_PROF_EVERY` (set 1 to dump every graph and difference consecutive dumps
into a per-token budget).

Fork git state: `d18f1fa` (accumulated engine work) → `9058ea6` (MULMATID_FLAT default on) →
`10288ca` (opt-in MULMAT_FLAT). Only `llama-mmap.cpp` `BONSAI_THP` is still uncommitted (dead
end, see §5). The exported patch `MonsterQwenDroid/patches/llama.cpp-moedroid.patch` is
regenerated from those 8 files and reverse-applies cleanly against `1b70350`.

## 4. Open work, in order

0. **SHIP THE BATCH-POOL AFFINITY FLAG — DONE, MEASURED, FLAG-ONLY. See §10.1.**
   `-tb 8` makes the prefill threadpool **unpinned across all 8 cores**, not pinned to `0x3f`:
   `postprocess_cpu_params` only copies `cpuparams` into `cpuparams_batch` when
   `cpuparams_batch.n_threads < 0`, so passing `-tb` silently skips the copy and leaves
   `mask_valid = false`. Adding **`--cpu-mask-batch 0xff --cpu-strict-batch 1`** pins the 8
   prefill threads 1:1 across cpu0-7 and is worth **+21.5% prefill** (4/4 mirrored pairs over
   two blocks, t = 9.08) and **+17.9% prefill / +3.5% decode** in a realistic prefill+decode
   request. Decode does not regress. No engine patch needed — add it to the shipping flags.
1. **Dense requantisation — DONE, `varH` IS THE SHIPPABLE VARIANT. See §10.2.**
   `llama-quantize ... COPY` with `--tensor-type` needed only a ~30-line patch (commit `6488af5`).
   **Result: 193 tensors converted, -638.3 MiB on shard 1, shard 2 byte-identical, but the
   perplexity gate FAILS at +0.852% (32 chunks, t=7.72)** because `output.weight Q5_K -> Q4_0`
   alone costs +0.814%. Drop that half. The 192 `hc_*` BF16 -> Q8_0 conversions (`varH`) are
   +0.092% PPL for 562.5 MiB and pass, and on device they are worth **+8.19% decode**
   (8 arms / 4 mirrored pairs, t = 7.09, all pairs positive) — better than the full rewrite's
   +2.82%, so the `output.weight` conversion was negative on speed as well as quality.
   `varH` shard 1 is `moe-dense-q8/varH-00001-of-00002.gguf`, sha256 `e7c69452...ded12e2`.
   Residual: `output_hc_up.weight` and `output_hc_down.weight` (bf16, 6.25 MiB each) are in
   shard 1 but the regex `hc_.*_(up|down)\.weight$` cannot match them, so 12.5 MiB of
   convertible BF16 is still unconverted.
2. **Prefill eviction shielding — MEASURED, BOTH OPTIONS REJECTED. See §10.3.**
   The post-prefill dip is real: the first 16 decode tokens after a 256-token prefill run at
   ~77% of the steady-state rate, and the decode phase carries 11-14.7k major faults/s against
   3.3-3.8k/s during prefill. But it is **I/O-volume-bound, not fault-bound**:
   `POSIX_MADV_WILLNEED` on the dense tensors at the transition cuts decode-phase major faults
   by 70-79% for **+0.33% decode (n.s.)** — the same bytes still have to come off storage, only
   the accounting changes. `POSIX_MADV_RANDOM` on the experts during prefill is a **dead end**:
   **-32.7% prefill, -9.0% decode, +649% major faults**, because it disables the fault-around
   and readahead that the expert access pattern actually benefits from. Both live behind
   `BONSAI_SHIELD` (0=off, 1=Option A, 2=Option B, 3=both), default off, token-identity clean.
   Any future attempt must **reduce bytes re-read after prefill**, not fault events:
   prefill-only expert top-k, or `MADV_DONTNEED` on prefill-only experts using the
   `uint16_t[48][512]` decode-hit table.
3. **Flat static partitioning for the dense `mul_mat` — DONE, MEASURED NEUTRAL (commit `10288ca`).**
   Implemented as `BONSAI_MULMAT_FLAT` (default OFF) in `ggml/src/ggml-cpu/ggml-cpu.c`:
   for `ne11==ne12==ne13==1 && ne01>=64` each thread gets one contiguous `ne01` slice aligned
   to the vec_dot row group, plus a streaming prefetch every 8 rows. Bit-identical to stock
   (md5 `20e5b496` across all 10 runs). Two mirrored A/B/B/A blocks, all arms on one cap profile
   (`cpu0-5=3532800`, `cpu6/7=3840000` of `4473600`), 64-token greedy decode:

   | block | A (OFF) | B (ON) | B (ON) | A (OFF) | pairs |
   |---|---|---|---|---|---|
   | 1 | 4.7122 | 4.7805 | 4.6027 | 4.7977 | +1.45%, -4.06% |
   | 2 | 4.7917 | 4.8492 | 4.8280 | 4.4450 | +1.20%, +8.62% |

   Point estimate **+1.7%** (pairs: mean +1.80%, sd 5.21%, SE 2.60%, t=0.69) — **not resolvable**.
   Coverage is real, not a dead path: ~420 dense `mul_mat` calls/token take it and 319 dense
   tensors have `ne01 >= 1536`, the threshold above which the stock loop actually steals 64-row
   chunks (`4 * nth * 64`). Below that the stock loop already hands each thread one contiguous
   chunk, so there the change adds only the prefetch.
   **The blocker is the noise floor, not the code** — see §7.9. Revisit only with a same-process
   alternation harness; the `BONSAI_MULMAT_FLAT` switch and `BONSAI_MULMAT_FLAT_REPORT` are in place
   for it.
4. **Revisit MTP only if the sidecar can be shrunk below the page-cache cliff** (~2.79 GB → ~1 GB by
   requantising `blk.48` experts to Q2_0). Same tooling as (1).

## 5. Do not re-try these (measured)

| idea | result |
|---|---|
| speculative decoding (MTP head) | **4x slower** (3.94 → 0.95 t/s): 77.5% acceptance, but the 2.8 GB sidecar evicts the expert working set — major faults 1,130 → 6,100/token |
| Q4_0 experts | would double the 34 GB pool and blow the cache fit that gives the 99.8% hit rate |
| static hot-expert pinning | modern fine-grained MoEs are load-balanced (near-uniform global frequency), so a static pin captures ~45% and starves the dynamic tail |
| file-backed THP (`MADV_HUGEPAGE`) | **inert and harmful**: f2fs on 6.6 never calls `mapping_set_large_folios()` so `FilePmdMapped` stays 0, but the hint changed VMA reclaim behaviour → 4.60 → 2.54 t/s. Judge mapping hints on `FilePmdMapped`/`AnonHugePages`, never on t/s alone |
| "1,440 barriers per token in `mul_mat_id`" | **false** — that function (`ggml-cpu.c:1640-1841`) has exactly ONE `ggml_barrier`, at line 1771, *before* the per-expert loop; upstream already uses a per-expert atomic counter |
| kernel micro-optimisation (1.52x unrolled Q2_0) | zero end-to-end change at two thread counts; the expert path is not ALU-bound |
| MOE_SUM as a *speed* lever | neutral (it is a size/clarity win, bit-identical) |
| flat dense `mul_mat` row partitioning (`BONSAI_MULMAT_FLAT`) | **neutral**: bit-identical and the path is live (~420 calls/token), but 4 mirrored pairs gave +1.45/-4.06/+1.20/+8.62% and the point estimate +1.7% is inside the noise. Ships opt-in, off |
| `output.weight` Q5_K -> Q4_0 via `llama-quantize COPY` | **fails the quality gate**: -75.8 MiB for **+0.814% PPL** (t=9.5). It is the entire cost of the dense rewrite. Use Q5_0/Q8_0 if that tensor must shrink |
| `-t 8 --cpu-mask 0xff` (all eight cores) | **14-18% SLOWER** than `0x3f` (4.11/3.96 vs 4.60-4.80 t/s) even though the kernel places the 8 workers 1:1 across cpu0-7 with no bunching. Memory-bound workload; the two prime cores add core clock and cross-cluster barriers, not bandwidth |
| putting a model under `/data/local/tmp` via `adb push` | **0.09 t/s vs 4.76 t/s** for the same bytes on `/sdcard` - `read_bytes` hit 1.9 TB and never converged. Same f2fs filesystem (same fsid); it is the physical layout of the pushed copy. Always stage models on `/sdcard` |
| `POSIX_MADV_RANDOM` on the expert tensors during prefill (`BONSAI_SHIELD=2`) | **-32.7% prefill, -9.0% decode, +649% major faults** (2.87e5 -> 2.15e6). Disabling fault-around/readahead turns amortised bulk reads into individual faults; the readahead on this access pattern is useful, not waste |
| `POSIX_MADV_WILLNEED` on the dense tensors after prefill (`BONSAI_SHIELD=1`) | mechanism works (**decode-phase major faults -70-79%**) but **+0.33% decode, n.s.** The post-prefill dip is I/O-volume-bound, not fault-bound - the same bytes still come off storage |
| `-t 6 -tb 6 --cpu-mask-batch 0x3f --cpu-strict-batch 1` (pin prefill to the 6-core cluster) | **neutral** for prefill: +1.38% group mean, pairs -1.32%/+4.30%, t = 0.53 (df 1). Prefill is compute-bound, so the two prime cores earn their keep there - unlike decode |
| `-tb 8 --cpu-mask-batch 0x3f` as an *oversubscription* concern | **not a real failure mode**: the batch pool never inherits `0x3f`, so the 8 prefill threads were never crammed onto 6 cores - they were unpinned across all 8 |
| an anonymous THP buffer for the dense weights (`MADV_HUGEPAGE` + `MADV_COLLAPSE`) | **dead**: the kernel grants a hard ceiling of ~207-210 huge pages (~420 MiB) no matter how much you map - 14% of the 2.9 GB dense set, 5% of 8 GB. `defrag=[never]` and not writable without root, so `MADV_COLLAPSE` returns ENOMEM past ~420 MiB. 2 MB alignment is not the limit (210 vs 204 pages) |
| dense `q3_K`+`q2_0` -> `q4_K` (`varD4`) | **fails the quality gate**: paired **+0.613%** PPL (t = 2.01, 19/32 chunks worse) for a predicted +5.8% decode. The dequant->requant round-trip adds a *second* quantization on top of the first, and these particular tensors are sensitive |
| dense `q3_K`+`q2_0` -> `q8_0` (`varD8`) | **passes quality (+0.107%, t = 0.50) but is slower**: measured **-3.87%** over 4 mirrored pairs, 4/4 negative. `q8_0` is 2.5x the bytes of `q3_K`, so the byte cost beats the 22.6 -> 50.0 GiB/s rate gain. Rate is only half the trade; per *token* it is the slowest of the three |
| assuming a higher bit width is always safer for speed | **false**: every dense type from `q4_K` to `q8_0` already sits at the same 43-50 GiB/s DRAM wall, so moving up one width costs bytes linearly and buys rate sub-linearly. `q5_K` is the last point where it still nets out positive |
| `-ub 256` or `-ub 128` (shrinking the physical microbatch) | **traps**: ~9.0 t/s prefill against ~10.3-10.65 for `-ub 512` and `-ub 64`, a **-14.9%** clean mirrored pair at matched temperature. Non-monotonic in `-ub` with no established mechanism. Keep the default 512 |

## 6. Python GGUF writer status (`tools/gguf_retarget.py`)

Dry-run verified (290 tensors, 1,328 → 706 MB, Gate 0 = per-tensor packed-length + row cosine
≥0.999). It runs to completion but **produces a malformed file** — gguf-py cannot parse one tensor
of the output (`cannot reshape array of size 129445600 into shape (512,2560,180)`), i.e. the
`raw_shape` convention for raw-copied tensors is still wrong. Four failures were all convention
guessing (logical vs byte shapes; array fields need `sub_type=field.types[-1]`, which IS fixed).
**Prefer the C++ route in §4.1**; if the Python route is ever revived, take `raw_shape` from
gguf-py's own writer path rather than deriving it, and re-run the inventory check:
tensor count must stay 1,223 and the type table must change only where intended.
Two more measured facts about the C++ route:

- **`--keep-split` produces BOTH output shards** from a single input shard path, named
  `<model>-<NNNNN>-of-00002.q8dense-<NNNNN>-of-00002.gguf`. So one invocation can rewrite the pair.
- **`q8dense-...-00002-of-00002.gguf` must be byte-identical to the source shard 2** (it holds only
  the `iq4_nl` ngram/PLE table, which is excluded from every rule). Compare `sha256sum` of the two
  `-00002-` files as a free integrity check on any future rewrite — if they differ, the copy path
  is not byte-exact and nothing downstream can be trusted.

Scratch output to delete: the 36 GB malformed file from the Python attempt, plus the ~62 GB verbatim
pair from the `COPY`/`llama-quantize` run, both under `moe-dense-q8/`. **The source shard was
never modified** (sha256 starts `69820c02…`).

## 7. Process rules (each one cost real time)

1. **Run arms mirrored (A/B/B/A), never sequentially.** Sequential runs hand the second arm a
   warm page cache — that manufactured a fake +70% once.
2. **Regime gate: record `scaling_max_freq` AND `cpuinfo_max_freq` per core per arm, and only
   compare arms with matching cap profiles.** This device settles at cpu6/7 = `3840000` of a
   `4473600` hardware max and does not return; the 6-core cluster reads `3532800`. The same config
   has measured 0.1 t/s and 4.6 t/s purely on this.
3. **The phone throttles within ~4 arms.** Keep arms short (decode-only), cooldown between blocks,
   and report same-profile *pairs* rather than means.
4. **Gate correctness separately from speed.** Kernel changes → require **token-identical** output
   (arithmetic unchanged). Requantisation → **token identity is the wrong gate** (a 0.38%
   perturbation legitimately diverges greedy output after a few tokens); use first-token top-5
   agreement + 8-chunk perplexity ≤0.5%.
5. **`pkill -f <pattern>` kills the invoking shell when the pattern appears in its own command
   line.** Hit it three times. Use `pkill -x <exact-name>` or bracket the pattern (`[g]guf_…`).
6. **A locked phone unmounts credential-encrypted `/sdcard`** — the app's model dir "disappears"
   and nothing can be measured (cores park in doze; `svc power stayon true` does not prevent it).
   `/data/local/tmp` stays reachable. Meaningful measurement requires: unlocked, app foregrounded,
   Game Space on.
7. **Push the whole library set when a change lives in `src/`.** Graph changes are in
   `libllama.so`, not `libggml-cpu.so`; two device results were invalid before this was caught.
8. **The device's engine can be run headlessly from `/data/local/tmp` with `LD_LIBRARY_PATH`**, which
   makes build-swapping instant — `libllamaserver.so -m <shard1> --host 127.0.0.1 --port <p> -c 8192
   -ctxcp 0 -np 1 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -ngl 0 --jinja -t 6 -tb 8
   --cpu-mask 0x3f --cpu-strict 1 --poll 0`. `llama-server` answers `/health` **before** the model
   is loaded — wait for `"status":"ok"`.
9. **Know the noise floor before you believe a small effect (new, measured 2026-10-14).** Repeating
   the *same* arm in a later block moved decode by up to **7.9%** (`a2`: 4.7977 then 4.4450) while
   `scaling_max_freq` stayed identical before and after the arm. The cap profile is necessary but not
   sufficient: a matching profile does not mean a matching thermal state. With a ~5-8% arm-to-arm
   spread, a 4-arm mirrored block **cannot** resolve the "few percent" this project keeps hunting —
   Stage 1 needed ~16 pairs to resolve 2% and had 4. So: (a) quote the pair spread next to any point
   estimate, (b) prefer harnesses that remove the variance instead of averaging it — for env-gated
   kernel switches, read the flag per call and alternate A/B *inside one process* on back-to-back
   requests, which deletes the model reload and most of the thermal term.
10. **Stage every model on `/sdcard`, never on `/data/local/tmp` (new, measured 2026-10-14).** The
   same 66 GB model, sha256-verified, ran at **4.76 t/s from `/sdcard` and 0.09 t/s from
   `/data/local/tmp`** after an `adb push`. `stat -f` shows one filesystem (same fsid), `dd` read the
   pushed copy at 2.8 GB/s, and the process showed no read syscalls - only mmap faults pulling
   2.9 GB/s, 1.9 TB total, never converging. Free memory and a forced page-cache eviction did not
   help. The expert tensors are ~460 KB random slices, and a copy pushed into an 80%-full f2fs volume
   is fragmented enough to make them ~50x slower while large sequential reads stay fast. **Validate a
   new model copy with one decode arm before spending a block on it.**
11. **`pkill -f <pattern>` self-matches through `adb shell` too (new).** `pgrep -f libllamaserver`
   inside an `adb shell` command matches that shell and kills it (exit 137). Use
   `ps -A -o PID,NAME | grep '[l]ibllamaserver'` and `kill -9` the PIDs. Also: `pkill -x` did not
   reliably kill the engine on this device, and a surviving engine holds port 8199, which makes the
   next arm report `ready after ~0s` and silently measure the *old* process's model.
12. **`adb` here is a bridge, not a local daemon.** The sandbox has no `/dev/bus/usb`, so it reaches
   the phone through a host-side adb daemon on the shared `127.0.0.1:5037`. **`adb kill-server` kills
   that host daemon and nothing in the sandbox can restart it** (no USB nodes, no systemd user bus).
   Recovery is manual, on the host: `adb kill-server && adb start-server && adb devices`.

## 8. Boundaries

- The Halogen / PC-inference-engine workspace is **read-only** as far as the phone work is
  concerned. Do not modify it.
- No upstream commits or PRs to llama.cpp (per its AGENTS.md); this is a local fork experiment.
- The model weights are not bundled anywhere and carry their own license. The app pins them by
  SHA-256 (`69820c02…` shard 1, `316b46f3…` shard 2).

## 9. Stage 2 results (2026-10-14)

Full reports: `MonsterQwenDroid/docs/STAGE2A-8core-placement.md` and
`MonsterQwenDroid/docs/STAGE2B-dense-rewrite.md`.

**Fork commits:** `d18f1fa` (accumulated engine work) → `9058ea6` (MULMATID_FLAT default on) →
`10288ca` (opt-in `BONSAI_MULMAT_FLAT`) → `6488af5` (`llama-quantize COPY` honours `--tensor-type`).
Only `llama-mmap.cpp` `BONSAI_THP` is still uncommitted.

**A. Eight cores are closed.** `-t 8 --cpu-mask 0xff` = 4.107 / 3.958 t/s against 4.60-4.80 on
`0x3f`. `/proc/<pid>/task/*/stat` field 39 shows the eight workers placed **1:1 across cpu0-7, 100%
sticky, no bunching** — the kernel does the ideal thing and it is still 14-18% slower. The workload
is memory-bound; the prime cluster adds core clock and cross-cluster barriers, not bandwidth. Keep
`-t 6 --cpu-mask 0x3f`.

**B. Dense rewrite: the mechanism works, the quality gate does not.** `llama-quantize ... COPY` with
`--tensor-type` now quantises exactly the matched 2-D tensors and raw-copies everything else
(proven: shard 2 came out byte-identical, metadata and the 1223-name tensor set unchanged).

| variant | MiB saved | paired PPL delta (32 chunks) | verdict |
|---|---|---|---|
| 192 x `hc_*_(up\|down)` BF16 -> Q8_0 | 562.5 | **+0.092%** (t=1.0) | passes |
| `output.weight` Q5_K -> Q4_0 | 75.8 | **+0.814%** (t=9.5) | fails |
| both (the full rewrite) | 638.3 | **+0.852%** (t=7.7) | fails the <=0.5% gate |

An 8-chunk run of the full rewrite read +0.408% and looked like a pass; 32 chunks show +0.852% with
31 of 32 chunks worse. **Eight chunks cannot resolve this gate** - pair the per-chunk values.

On device (both models on `/sdcard`): decode **+6.30%** on the one fully cap-stable mirrored pair,
**+2.82%** on group means over 8 arms across 2 blocks, 3 of 4 pairs positive. Prefill **+1.24%** on
group means once the arm was redesigned to fit the thermal window - a 966-token prompt makes an arm
~3.5 min and all four arms throttled mid-arm (pairs +0.31% / +9.63%, unusable); a 256-token prompt
makes it ~1.7 min and all four arms held one cap profile (pairs -0.95% / +3.57%). **Shorten the
prefill arm before trusting any prefill pair on this device.**

**Next:** measure the `hc_*`-only variant (varH) on device - it is the shippable one, its host PPL
already passes (+0.092%), and its own device speed is the one number still missing. It is already
built on the host: `moe-dense-q8/varH-00001-of-00002.gguf` (37,033,916,192 B, sha256
`e7c69452...ded12e2`) plus `varH-00002-of-00002.gguf` (sha256 `316b46f3...c161e113`, byte-identical
to source shard 2). Push it to `/sdcard`, never `/data/local/tmp` (rule 7.10).

Cleanup already done (2026-10-14): the stale full-rewrite copies are gone from both the phone
(`/data/local/tmp/mf-model`, `/sdcard/Download/mf-model`) and the host (`moe-dense-q8/Qwen3.8-...q8dense-*`).
Phone 249 GB free, host 177 GB free. The source model in `files-drop-bonasidroidv2/flashnext-q2_0/`
is untouched.

## 10. Stage 3 results (2026-10-14/15)

Full report: `MonsterQwenDroid/docs/STAGE3-report.md`. Engine: tree at `6488af5` plus the
`BONSAI_SHIELD` addition in `src/llama-context.cpp` / `src/llama-context.h` (uncommitted).
Device library set `/data/local/tmp/lib-shield`; models on `/sdcard/Download/mf-model/`
(varH shard 1 sha256 `e7c69452...ded12e2`, shard 2 `316b46f3...c161e113`, both verified on
device). `/sdcard` supports **no symlinks and no hard links** (EPERM), so shard 2 had to be
copied on-device rather than linked.

### 10.1 The prefill threadpool was never pinned (new, flag-only win)

`-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1` does **not** run prefill on `0x3f`. `-tb 8` sets
`cpuparams_batch.n_threads = 8`, which makes `postprocess_cpu_params` skip the
`cpuparams_batch = cpuparams` copy (`common/common.cpp:294-301`); `mask_valid` stays false,
`ggml_threadpool_params_from_cpu_params` therefore leaves `tpp_batch.cpumask` all-zero
(`common/common.cpp:1760-1762`), and affinity is applied only for a valid mask
(`ggml-cpu.c:3589-3591`). All-zero means "inherit", so the 8 prefill threads float across
cpu0-7. The CLI help's "default: same as --cpu-mask" only holds when `-tb` is **not** passed.
Confirmed on device: 19 of 24 threads carried `Cpus_allowed_list: 0-7`, and the 7 prefill
workers had identical CPU histograms weighted to cpu6/7 (`6:92 7:66` and siblings); adding
`--cpu-mask-batch 0x3f --cpu-strict-batch 1` pinned 14 threads across cpu0-5 and the ggml
workers stopped touching cpu6/7 entirely.

| config | prefill t/s (group mean) | vs shipping |
|---|---|---|
| `-t 6 -tb 8` (shipping, unpinned) | 7.539 | baseline |
| `-t 6 -tb 6 --cpu-mask-batch 0x3f --cpu-strict-batch 1` | 7.792 | +1.38% (n.s.) |
| **`-t 6 -tb 8 --cpu-mask-batch 0xff --cpu-strict-batch 1`** | **9.072** | **+21.5%** |

TB2 + TB3, 8 arms, 256-token prefill, all cap-stable at `3532800/3840000`: pairs
+23.69% / +21.78% / +15.03% / +26.16%, group means 7.4650 -> 9.0716 = **+21.5%**, pairs
t = 9.08 (df 3). A position check rules out ordering: in TB1, where the flag is neutral, the
2nd/3rd slots ran +1.38% above the 1st/4th; here they run +21.5% above. TB4 (one request,
256-token prefill + 64 decode) gives prefill **+17.9%** and decode **+3.5%**, i.e. no decode
regression. **Recommendation: add `--cpu-mask-batch 0xff --cpu-strict-batch 1` to the
shipping flags.**

### 10.2 `varH` banked: +8.19% decode

Staged on `/sdcard`, sha256 verified against the host, one throwaway warmup arm first
(**5.6199 t/s**, cap-stable) to prove the copy was not fragmented. Two mirrored A/B/B/A
decode blocks (64 tokens), every arm cap-stable with no sample falling to `2841600`:

| block | pair | shipping | varH | delta |
|---|---|---|---|---|
| D1 | a1/b1 | 5.2162 | 5.6004 | +7.37% |
| D1 | b2/a2 | 5.0757 | 5.6271 | +10.86% |
| D2 | a1/b1 | 5.3384 | 5.6314 | +5.49% |
| D2 | b2/a2 | 5.1229 | 5.5950 | +9.22% |

Group means shipping **5.1883** vs varH **5.6135** = **+8.19%**; pairs mean +8.23%, sd 2.32,
SE 1.16, **t = 7.09 (df 3)**, 4/4 pairs positive. This is the first dense result that clears
the Stage 1 noise floor on its own terms. `majflt` is also lower on varH (13,857-16,340 vs
15,547-17,099 per 64 tokens). **Ship `varH`.**

### 10.3 Prefill eviction shielding: the dip is I/O-volume-bound

The dip is real. In a single 256-token-prefill + 64-token-decode request, the first 16 decode
tokens run at **~77% of the steady-state rate** (ratios 0.759 / 0.784 / 0.811 / 0.744 across
four baseline arms), and splitting the fault counters at the request's own `prompt_ms`
boundary shows the decode phase carrying **11,000-14,700 major faults/s** against
**3,300-3,800/s** during prefill.

| option | mechanism | prefill | decode | majflt | verdict |
|---|---|---|---|---|---|
| `BONSAI_SHIELD=1`: `WILLNEED` dense at the transition | decode-phase faults -70-79% | +3.4% (n.s.) | +0.33% (n.s.) | -36.9% | mechanism confirmed, **no speedup** |
| `BONSAI_SHIELD=2`: `MADV_RANDOM` experts during prefill | readahead disabled | **-32.7%** | **-9.0%** | **+649%** | **dead end** |

Option A's log line, with the server at `-lv 4` (libllama INFO maps to `LOG_LEVEL_TRACE` (4)
while the default threshold is 3, so it is invisible otherwise):

```
I bonsai_shield_apply: BONSAI_SHIELD applied to 144 expert tensors (32400.0 MiB)
I bonsai_shield_apply: BONSAI_SHIELD applied to 1079 dense tensors (2907.8 MiB)
```

Greedy token identity (`--temp 0`) is byte-identical across shield 0/1/2:
md5 `f1ac9c8a77541e447f1f4810a8cadc73`, 170 bytes. Note the dense set must exclude
`per_layer_token_embd.weight` - it is a single 26.8 GiB tensor in the same list, and a
`WILLNEED` on it would readahead 26.8 GiB per transition.

### 10.4 Thermal methodology (this device)

The phone sits near 41-43 C in the CPU cluster zones while charging at 100% on AC, so a
temperature-threshold gate is unreachable on this bench; `-lv`-style temperature gating was
tried and abandoned. What works: require the known-good cap profile (`cpu0 = 3532800`,
`cpu6 = 3840000`) to hold for **three consecutive 10 s polls** before loading the model, use
a **120 s cooldown between arms**, and judge stability from the caps read **immediately either
side of the measured window** plus a 5 s sampler across the whole arm. An earlier TB1 attempt
with only a 30 s cooldown produced two arms that dropped to `2400000/2841600` mid-arm; those
were discarded and the block re-run. Decode arms are short enough that the cap never moved;
256-token prefill arms need the full 120 s cooldown.

### 10.5 Still open after Stage 3

* `output_hc_up.weight` / `output_hc_down.weight` (bf16, 6.25 MiB each) are in shard 1 and
  convertible, but `hc_.*_(up|down)\.weight$` cannot match `output_hc_down.weight`. 12.5 MiB
  left on the table.
* `MADV_DONTNEED` on prefill-only experts, using the `uint16_t[48][512]` decode-hit table, is
  the one eviction variant not yet tried - and it is the one that would actually reduce bytes
  re-read, which §10.3 says is the binding constraint.
* A prefill-only `BONSAI_MOE_TOPK` reduction is the other way to shrink the eviction footprint.
* The `BONSAI_SHIELD` change is uncommitted; `--cpu-mask-batch 0xff` needs no code change but
  the shipping flag list in the app needs updating.

## 11. Stage 4 results (2026-10-15)

Full report: `MonsterQwenDroid/docs/STAGE4-report.md`. Engine: same tree, plus a new
`BONSAI_PROF` per-weight/per-op profiler and per-graph fault counters in
`ggml/src/ggml-cpu/ggml-cpu.c` (uncommitted, default off). A second device library set
`/data/local/tmp/lib-prof` is the `-DBONSAI_PROF=1` build; its CMake cache diffs clean against
`build-arm64-ship` apart from the define.

### 11.1 The per-op budget at 5.77 t/s decode

Measured by differencing `BONSAI_PROF_EVERY=1` dumps over the last 48 graphs of a 98-graph run
(measured decode tokens 17..64), three runs. Total **173.229 ms/token = 5.773 t/s**, which
reconciles with the 5.61-5.79 t/s measured end to end - there is no large un-attributed gap.

| op | ms/token | share |
|---|---|---|
| `MUL_MAT_ID` (experts) | **79.468** | 45.9% |
| `MUL_MAT` (dense) | **70.875** | 40.9% |
| `GET_ROWS` | 5.617 | 3.2% |
| `GATED_DELTA_NET` | 4.940 | 2.9% |
| `CPY` + `CONCAT` + `MUL` + `UNARY` + `ARGSORT` | 7.539 | 4.4% |
| everything else | 4.789 | 2.8% |

The two matmuls are **86.8% of decode**. Nothing else reaches 3.3%, so the non-matmul ops
(Gated DeltaNet recurrence 7.4 ms, attention 1.5 ms, `hc_*` elementwise 1.6 ms, RMSNorm
0.7 ms) are all dead ends.

Dense runs at **36.4 GiB/s** aggregate, but two types drag it down: dense `q2_0` at
**16.0 GiB/s** and `q3_K` at **22.6 GiB/s**, against 43-50 GiB/s for every other type. Those
two are 24.9 ms/token (14.4% of decode) for 546 MiB of 2644 MiB of dense bytes - **the one
place in the dense path where a requant would pay**. `output.weight` is 8.664 ms/token
(5.00% of decode) and already runs at 47.0 GiB/s, the best rate in the model, which is why
the Stage 2 `Q5_K -> Q4_0` trade was bad: no headroom left. The 291 remaining BF16 tensors are
4.381 ms/token (2.53%) at 46.2 GiB/s - also no headroom.

Experts: **632.8 MiB/token in 79.468 ms = 7.8 GiB/s (8.4 GB/s)**, 144 calls, **4.7x slower per
byte than dense** for 2.5x the dense backbone's time in 24% of the bytes. That is the
bottleneck.

**A major fault costs ~170 us** (regression of per-graph op time against per-graph majflt over
47 graphs: 166.5 / 168.2 us, r = +0.41/+0.45), and it lands **entirely in the expert path** -
dense `MUL_MAT` shows r = -0.04/-0.15, which is what a cache-resident backbone looks like.
Of the 62.49 ms/token by which the expert path exceeds the dense rate for the same bytes,
**~19.6 ms (31%) is page-fault stalls** and the residual 42.9 ms is the Q2_0 compute path
itself. So the expert penalty is neither purely ALU-bound nor purely fault-bound.

Prefill (one 262-token graph): **113.976 ms/token = 8.77 t/s of compute** against 8.574
measured. `MUL_MAT_ID` **70.521 (61.9%)**, `MUL_MAT` 31.544 (27.7%), `RMS_NORM` 2.594,
`GATED_DELTA_NET` 2.177. Prefill is far more expert-dominated than decode. Major faults
1,032.8/token, 9x the decode rate.

**This also explains the Stage 3 batch-pinning win**: same model and prompt, pinned vs
unpinned, total 131.61 -> 113.98 ms/token, and the change is *entirely* in the expert path
(87.70 -> 70.52 ms/token, -19.6%) while dense is untouched (31.27 -> 31.54). The expert rate
rises 1.38 -> 1.71 GiB/s even though majflt rises 49%, so the expert path is not purely
storage-bound: 8 pinned workers extract more from the same flash.

**The 26.8 GiB shard-2 PLE table is not a cost.** `GET_ROWS` totals 5.62-5.90 ms/token over
174 calls, but `per_layer_token_embd.weight` never reaches the dense top-16 (whose 16th entry
is 0.318 ms/token), so it is **< 0.32 ms/token**. The `GET_ROWS` time is spread over ~174
small per-layer lookups, not concentrated in the big table. That question is closed.

### 11.2 Anonymous THP is capped at ~420 MiB on this device

`/data/local/tmp/thp_probe` (source `MoeDroid/tools/thp_probe.c`). `enabled=[always]`,
`defrag=[never]` and **not writable without root**, `shmem_enabled=[never]`, `FileHugePages=0`,
`AnonHugePages` 178 MiB idle, 12,287 MB swap with 10,573 MB free (**zram is active**, no
physical swap partition).

| mapping | coverage after touch | `MADV_COLLAPSE` |
|---|---|---|
| 64 MiB aligned | 100% | rc=0 |
| 512 MiB aligned | 80.9% | ENOMEM |
| **2908 MiB** (the dense set) | **14.0%** (204 x 2 MB) | ENOMEM |
| 2908 MiB aligned | 14.4% (210) | ENOMEM |
| **8192 MiB** | **5.1%** (209) | ENOMEM |
| 8192 MiB aligned | 5.1% (209) | ENOMEM |

**Going from 2.9 GB to 8 GB bought exactly zero extra huge pages (210 -> 209).** With
`defrag=never` nothing compacts, so only the order-9 blocks already free in the buddy
allocator are usable and there are ~420 MiB of them. Alignment is not the limit. The
anonymous dense-cache plan (pread the 1079 dense tensors into an anon buffer, `MADV_DONTNEED`
the file ranges) would get 14% PMD coverage and pay a copy for nothing; an 8 GB anonymous
expert cache would get 5% and take 8 GB away from the page cache that gives the 99.8% expert
hit rate. `SwapFree` did not move during either allocation - the kernel reclaimed page cache,
it did not swap.

### 11.3 `-ub` is non-monotonic and `256`/`128` are traps

518-token prompt, `n_predict 16`, shipping flags. Two ordered passes (descending
512/256/128/64 and ascending 64/128/256/512) plus a mirrored A/B/B/A (64/256/256/64), 90 s
cooldown and the cap gate before every arm.

| -ub | desc | asc | mean |
|---|---|---|---|
| 512 | 10.7888 | 10.5162 | **10.6525** |
| 256 | 9.0385 | 9.1049 | 9.0717 |
| 128 | 8.9998 | 8.9314 | 8.9656 |
| 64 | 10.1762 | 10.3827 | **10.2795** |

The two passes agree on the ordering to ~2.5%, and the spread is ~19%. Both passes put
{512, 64} in slots 1 and 4 and {256, 128} in slots 2 and 3, so slot position is confounded
with the value; the mirror pass breaks it. The one **fully cap-stable pair at matched
temperature** (start 91.5 vs 93.0 C, both windows on `3532800/3840000`) is
**`-ub 64` 10.5919 vs `-ub 256` 9.0194 = -14.85%**. Group means over all four mirrored arms
are -11.8%. **Keep the default `-ub 512`**; the actionable finding is that 256 and 128 are
~9-15% slower, so shrinking the microbatch to save memory must not start there.

The prefill/decode **major-fault split is not a usable `-ub` signal**: the decode-phase count
for `-ub 512` was 7,210 in one pass and 157,406 in the other, a 22x swing driven by history.
Do not quote it.

### 11.4 Shipping defaults and the device model are updated

* `EngineProfile.pinBatchAllCores` (default true) plus a guard in `LlamaServerEngine.kt` emits
  `--cpu-mask-batch 0xff --cpu-strict-batch 1` whenever `pinPerfCluster` applies and `-tb` is
  actually passed. QUALITY/BALANCED/SPEED (`-t 6 -tb 8`) now produce exactly the measured flag
  set; **THERMAL_SAVER (`-tb 4`) opts out** because that configuration was never characterised.
  No native rebuild needed - both flags are stock options of this revision.
* **Shard 1 on the device is now varH.** Done as stage -> sha256 verify -> atomic rename, not
  an in-place overwrite: `/sdcard/Android/data/com.prismml.moedroid/files/models/...00001-of-00002.gguf`
  is 37,033,916,192 B, sha256 `e7c69452...ded12e2`. Shard 2 is untouched (byte-identical
  between the variants). The `.meta.json` sidecar was rewritten.
* The swap is transparent: `isOneDownloaded()` short-circuits to a plain `exists()` when a
  sidecar is present, and `ModelAsset.sha256` is **never read anywhere in the app** (grep
  finds no consumer outside `ModelCatalog.kt`). The catalog was still updated so the UI and
  free-space arithmetic agree with disk.
* Re-verified through the app's own path: a decode arm loaded
  `.../models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` and ran **5.4626 t/s**,
  cap-stable - the varH band, not the shipping band.
* **Open:** the hosted URL still serves the un-rewritten shard 1. Because the hash is not
  enforced, a fresh install would silently download the original and lose the varH win rather
  than failing loudly. The rewritten shard must be published (or the asset URL repointed)
  before re-provisioning from scratch. Recorded in a KDoc block above the asset list.

### 11.5 Still open after Stage 4

* Dense `q2_0` (16.0 GiB/s) and `q3_K` (22.6 GiB/s) are the only dense types below 35 GiB/s -
  14.4% of decode for 20.6% of dense bytes. A requant or a faster dequant for those two is the
  remaining dense lever, and unlike `output.weight` they have headroom.
* The expert path's 42.9 ms/token residual (69% of its excess over the dense rate) is the
  Q2_0 compute path, not faults. That is where a kernel change would have to land.
* `-ub`'s non-monotonic shape has no mechanism; it needs a randomised design, not an
  explanation.
* `BONSAI_PROF` and `BONSAI_SHIELD` are both uncommitted in `ggml-cpu.c` /
  `llama-context.{cpp,h}`.


## 12. Stage 5 results (2026-10-15/16)

Full report: `MonsterQwenDroid/docs/STAGE5-report.md`. Design doc:
`MonsterQwenDroid/docs/STAGE5-iqk-port-plan.md`. External cooling fan attached for the whole
session (CPU-cluster zones started at 29-35 C against 41-43 C in Stage 4; `cpu6` never unlocked
above `3840000`).

### 12.1 `ik_llama.cpp` is a real oracle and it loads `varH` unmodified

Cloned `ikawrakow/ik_llama.cpp` @ `8313388a`, cross-compiled for Android arm64 with the same
NDK flags as our build plus `-DLLAMA_BUILD_EXAMPLES=ON` (ik's server lives in `examples/`, not
`tools/`) and `-DGGML_IQK_MUL_MAT=ON` (its default). **No compatibility shim was needed**:

* same arch string `qwen4exp` / `LLM_ARCH_QWEN4EXP`, with `create_qwen4exp_tensors()` present;
* same `GGML_TYPE_Q2_0 = 42`, same `QK2_0 = 64`, and a **byte-identical** `block_q2_0`
  (`{ggml_half d; uint8_t qs[16];}`, 18 B) with the same `{00=-1,01=0,10=+1,11=+2}` mapping;
* every distinctive tensor present, including `hc_*`, `ple_*`, `ssm_*`, `ffn_*_shexp`,
  `indexer.*`, `per_layer_token_embd`, `output_hc_*`.

**Correction to the brief:** ik's `Q2_0` is **not** a 10-byte block. It is 18 bytes, same as
ours. The 34-byte `{fp16 d; uint8_t qs[32];}` type is ik's **`PQ2_0`** (`QK=128`), a separate
Prism-ternary format we do not use. The 10-byte shape does not exist for either.

ik needs `libomp.so` (OpenMP is on by default there, off in ours); push it from the NDK.
ik does **not** implement `--cpu-strict`, `--cpu-mask-batch` or `--poll`, so it cannot take the
batch-pool pinning that buys us +21.5% prefill - and it still wins.

Matched flags, same session, same model file, q8_0 KV both sides:

| engine | decode t/s (steady) | median | prefill t/s | mean |
|---|---|---|---|---|
| ours (`varH`, shipping flags) | 6.67, 6.89, 6.40, 6.96, 6.91, 6.93, 6.91 | **6.90** | 9.23, 8.32, 7.64, 8.25, 7.51, 6.89 | 8.40 |
| **`ik_llama.cpp`** | 9.55, 10.06, 8.63, 8.56, 8.59, 8.54 | **8.61** | 12.67, 12.06, 11.21 | **11.98** |

**ik: +25% decode, +43% prefill**, with no overlap between the steady-state sample sets.

### 12.2 A measurement-protocol effect worth ~30% (new, important)

Reconciling Stage 5's numbers against Stages 3/4 turned up a protocol effect, not a hardware one:

| protocol | our engine, `varH`, shipping flags |
|---|---|
| fresh load + one warmup + one measured request (the `s3_arm.sh` arm protocol) | **5.2264 t/s** |
| one load + many repeated requests (long-lived server) | **6.90 t/s** |

Same model, same flags, same session, same cap profile, both cap-stable. The cause is page-cache
state: the ~12-15 GB expert working set is not established by a single warmup.

* **Stage 3/4 absolute numbers are arm-protocol numbers** and understate steady-state decode by
  ~25-30%. Their *relative* results stand, because both arms of every comparison shared the
  protocol.
* **Stage 5's `varD` deltas are arm-protocol numbers too**, so they are internally valid but not
  directly comparable to the 6.90/8.61 figures above.
* **New rule: state the protocol next to every t/s number.** Same class of error as the
  `/data/local/tmp` storage trap.

### 12.3 `varD`: only `q5_K` passes, and the win is small

148 tensors / 545.89 MiB targeted (90 dense `q3_K` + 58 dense `q2_0`), `token_embd.weight` and
all 144 routed experts excluded, matched by an **explicit anchored name list** rather than a
regex - the families overlap across types (`attn_q` exists as `q3_K`, `q2_0` *and* `iq4_xs`), so
a family regex would have wrongly converted fast types. Built with the existing
`llama-quantize --allow-requantize --keep-split --tensor-type-file ... COPY 16` path; shard 2
byte-identical in every variant.

| variant | target | +MiB vs varH | PPL vs varH | gate <=0.5% | measured decode |
|---|---|---|---|---|---|
| `varD4` | `q4_K` | +220 | **+0.613%** (t=2.01) | **FAIL** | not run (rejected on quality) |
| **`varD5`** | **`q5_K`** | **+388** | **-0.117%** (t=-0.43) | **PASS** | **+1.95%, 7/8 pairs, t=1.64** |
| `varD8` | `q8_0` | +895 | +0.107% (t=0.50) | PASS | **-3.87%, 4/4 pairs negative** |

`varD5` is the only variant that wins on both axes, and its decode gain is **not formally
significant at 8 mirrored pairs** (95% CI -0.86% to +4.75%) against a +3.2% prediction. It is
safe on quality and worth ~2% decode if the +388 MiB is free; it is not the +5% that would
justify a rewrite.

**Methodology correction worth keeping:** `llama.cpp` prints `[i]<ppl>` as `exp(nll/count)` with
**cumulative** `nll`/`count`, i.e. a running estimate, not a per-chunk value. Differencing those
directly overstated the delta by ~5x. Recover per-chunk PPL as `c_i^i / c_{i-1}^{i-1}`.

### 12.4 The port plan, and its main correction

`STAGE5-iqk-port-plan.md`. The brief's premise that our `ggml_vec_dot_q2_0_q8_0` is a "14-op
scalar" loop is **wrong**: `arch/arm/quants.c:222` is hand-written NEON with a dual-accumulator
unroll whose comment already records the bottleneck (`vcvtq_f32_s32 -> vmlaq_n_f32` once per 32
weights; 6.2 -> 9.4 Mrow/s, 1.52x). The real gap is `nrc`: our `q2_0` is `.nrows = 1` with
`assert(nrc == 1)`, while ik runs `mul_mat_q2_0_q8_0_x4` / `_q8_2_x4` with 4 rows and packs two
block scales into one `float32x4` so a single `vmlaq` retires 128 weights.

The plan says **profile first**: with `n=1` there is only one activation row, so `nrc` cannot be
the whole story, and Stage 4 showed 31% of the expert path's excess is page-fault stalls with
~43 ms/token left as compute. The unattributed candidates are ik's `GGML_EXPERT_CHUNKING`, the
`q8_2` activation format, runtime kernel selection, and its own flash-attention implementation.
