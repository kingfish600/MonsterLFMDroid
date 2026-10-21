# Stage 10 / Window 2 report

Date: 2026-10-19 (afternoon into night). Device: RedMagic 10S Pro (NX789J), serial `91251291018D`,
Snapdragon 8 Elite, CPU only. Written incrementally, one section per completed step.

Order of work: **Step 0 (`T1`-`T4`, host-only, zero adb) -> device phases**.

---

## 0. Host-only preparation

### 0.1 `T1` — device libraries rebuilt with the fixed per-type instrument

The Stage 9 §6 fix is present and verified against `HEAD` before rebuilding:

```
ggml/src/ggml-cpu/ggml-cpu.c:4102
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
grep -c "GGML_OP_MUL_MAT_ID || node->op == GGML_OP_GET_ROWS"  ->  0
```

Both device libraries were rebuilt from this tree:

| library | timestamp | bytes | per-type table | profiler | D1 kernel | PK cache |
|---|---|---|---|---|---|---|
| `build-arm64-ship/bin/libggml-cpu.so` | 2026-10-19 13:09:12 | 5,764,944 | **absent** | **absent** | present | present |
| `build-arm64-prof/bin/libggml-cpu.so` | 2026-10-19 13:09:11 | 5,778,824 | present (`MATMUL %-14s`, `CPU OP PROFILE`) | present | present | present |

Both are `ELF 64-bit LSB shared object, ARM aarch64`; the prof library exports 434 `T` symbols.
The shipping build still carries **no** profiler strings, which is the `HOST_CHECK` invariant.

### 0.2 `T2` — `BONSAI_HYBRID_VIEW=1`: the recurrent-state gather is the identity on decode

**What `build_rs` does (priced from the code, `llm_graph_context::build_rs`, `llama-graph.cpp:3777`):**

| step | operation | cost |
|---|---|---|
| 1 | `ggml_scale_inplace` on a 1-row view of `states`, when `rs_zero >= 0` | one row zeroed **in place** |
| 2 | `ggml_get_rows(states, state_copy)` -> `states_all` | **the copy**: `state_size` floats per state row |
| 3 | `output_states` = `ggml_view_2d(states_all, state_size, n_seqs, ...)` | free |
| 4 | `ggml_cpy(states_extra, view(s, ...))` for the `n_rs - n_seqs` extra states | zero rows on this model |

So the **write-back is already an in-place `ggml_cpy` into a view of `s`**, issued later in the graph
by the layer (`delta-net-base.cpp:553`: `ggml_cpy(new_state, ggml_view_2d(ssm_states_all, ...))`). The
gather in step 2 is therefore a pure *read* copy, and the dependency chain (read -> new_state -> cpy)
orders the write after it.

**The traffic being removed, per token:**

| gather | rows x bytes | per token |
|---|---|---|
| `cache_s` (36 layers, `n_embd_s` = 786432 floats) | 1 x 3,145,728 B | **113.2 MB** |
| `cache_r` (36 layers, 30720 floats) | 1 x 122,880 B | **4.4 MB** |
| PLE state read (`cache_ple_r_l1`, 92160 floats, once) | 1 x 368,640 B | 0.37 MB |
| | | **118.0 MB/token** (the brief's 117.6 MB is the 72 layer gathers) |

#### The identity condition, and the two bugs that made the first attempt wrong

The first version gated on `n_rs == 1 && n_seqs == 1`. That is **not sufficient**:
`llama_memory_recurrent_context::s_copy(i)` returns `mem->cells[i + head].src0` — a *copy plan*, not
an index — so one slot does not imply the mapping is row 0. The condition now also requires
`kv_state->s_copy(0) == 0`, read at graph-build time (after `init_batch`, before the graph runs).

Two real defects were found and fixed while making the view safe, both with a crash as the symptom:

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | `GGML_ASSERT(buffer)` in `ggml_backend_meta`` | with `n_rs == n_seqs` the extra-state copy is **zero rows**; its view gets no buffer | build that copy only when the view path is not taken (`if (!rs_identity)`), so the **control path is byte-identical to Stage 9** (6067 nodes) |
| 2 | `GGML_ASSERT(buffer)` in `llm_graph_input_mem_hybrid::set_input` | the view removes the only consumer of the `rs_s_copy` input, so the scheduler never allocates it and `set_input` writes a null buffer | fill `s_copy` only when it has a buffer (`&& inp_rs->s_copy->buffer`) |

#### Verification (host, real varH, `-t 6`)

| check | `BONSAI_HYBRID_VIEW=0` | `BONSAI_HYBRID_VIEW=1` |
|---|---|---|
| graph nodes | 6067 (unchanged from Stage 9) | 5848 |
| `GET_ROWS` built | 175 | **102** |
| `GET_ROWS` **executed** (profiler graph-delta) | 174 | **101** |
| `MUL_MAT` / `MUL_MAT_ID` nodes | 785 / 144 | 785 / 144 |
| `llama-perplexity -c 512 --chunks 2` | **4.2138** | **4.2138** |
| per-chunk printed series | `cf9cd3b1…` | `cf9cd3b1…` (identical) |
| generated text (timing line excluded) | `354c40fa…` | `354c40fa…` (identical) |

**The measured drop is 73 calls, not 72: 36 `cache_s` + 36 `cache_r` + the once-per-graph PLE state
read, which goes through the same builder.** Executed `GET_ROWS` therefore goes 174 -> **101**. Both
runs exit 0. The control graph is untouched at 6067 nodes, so every later comparison has a pristine
reference.

### 0.3 `T3` — `BONSAI_MMID_BLK` wiring and the two quality gates recalibrated

#### `BONSAI_MMID_BLK` in the expert path only

`blck_0` exists twice, and the brief's decoy warning is real:

| line | function | action |
|---|---|---|
| `ggml-cpu.c:1379` | `ggml_compute_forward_mul_mat` (dense) | **untouched** — still `const int64_t blck_0 = 16;` |
| `ggml-cpu.c:1862` | `ggml_compute_forward_mul_mat_id_one_chunk` (expert body) | wired to `bonsai_mmid_blk(16)` |

`grep -c bonsai_mmid_blk` over the file is 2 (the helper and its single use), and the dense function's
initialisation is still the literal `16`, so the dense path is byte-unchanged.

**A real overflow was found and fixed while wiring it.** The function's temporaries were sized to the
old literal (`float tmp[16]`, `tmp_b/tmp_c/tmp_d[16]`) while three `memcpy`s copy up to `blck_0`
floats into them. `BONSAI_MMID_BLK=64` aborted under qemu instantly. The cap and the arrays are now
tied to one constant, so they cannot drift:

```c
#define BONSAI_MMID_BLK_MAX 512
    float tmp[BONSAI_MMID_BLK_MAX];
    float tmp_b[BONSAI_MMID_BLK_MAX], tmp_c[...], tmp_d[...];
    v = (m >= 1 && m <= BONSAI_MMID_BLK_MAX) ? m : dflt;
```

`BONSAI_MMID_BLK_DEBUG=1` prints the parsed value, so the bit-exactness matrix cannot pass vacuously
(the verifier asserts the line `BONSAI_MMID_BLK rows_per_block=428`).

#### Gate 3(b): the ceiling is calibrated from the measured oracle drift

The per-chunk drift between **our mode-1 device PPL** and **ik's host PPL** over the same 32 chunks,
computed with the project's `g_i = c_i^i / c_(i-1)^(i-1)` estimator:

```
ours mode1 (device) vs ik (host)   n=32   max |rel| = 2.8893e-02   median 8.57e-03
   -> 2 x max = 5.7787e-02
   -> ceiling = min(5e-2, 2 x oracle drift) = 5e-2    (the cap binds)
```

That pair is the project's known-zero reference (`+0.026%` paired, `t = +0.11`), so it defines how
much churn a genuinely zero-effect kernel change produces. `s6_ppl_recheck.py` now enforces:

1. **non-vacuity**: `max |rel_diff| > 0`;
2. **per-chunk ceiling**: `max |rel_diff| <= min(5e-2, 2.0 * ORACLE_DRIFT)` = **5e-2**;
3. **quality PASS** strictly by `|t| < 2.0` **and** `0` inside the 95% CI.

`--selftest` covers all of it (9 assertions), including a `+6%` case that must still fail, a `+10%`
case, and a `~2e-2` zero-mean churn case that must pass — the Stage 9 case in miniature.

**Re-adjudicating the Stage 9 pair under the calibrated gate:**

```
chunks parsed: 32/32 both arms
max per-chunk |g2-g1|/g1 = 2.466e-02   (ceiling 5.000e-02 = min(5e-2, 2x oracle drift 5.779e-02))
paired mean = +0.052884%   sd = 0.982267   SE = 0.173642
paired t = +0.305   (|t| < 2.0 required)
95% CI = [-0.301259%, +0.407027%]   (t_crit 2.0395, df 31)
PASS
S9PPL_PAIR=PASS
```

`BONSAI_MOE_X4=2` is therefore **Gate 3(b) PASS**: non-vacuous, `2.47e-2 < 5e-2`, `t = +0.305`,
`0` inside `[-0.301%, +0.407%]`.

#### Gate 3(a): judged against stock's own oracle error

**The brief's file coordinate was wrong and is corrected here:** `s7_q2_0_x4t_test.c` is a pure
bit-exactness test with no `1e-5` threshold. The fixed `1e-5 * scale` triplet lives in
`MoeDroid/tools/s6_q2_0_x4_test.c` (the file that also owns the FP64 oracle), so that is where the
recalibration was applied — located by content, not by the quoted filename.

| before | after |
|---|---|
| `d1, d2, dref <= 1e-5 * scale` | `d1, d2 <= 2.0 * dref` (stock's own error vs the FP64 oracle at the same shape), plus a sanity ceiling `dref <= 1e-4 * scale` so a broken oracle cannot make the relative gate vacuous |

Verified under `qemu-aarch64-static`, with `nb = 10` and `nb = 40` kept permanently (`nb` set is
`{1,2,3,5,8,10,16,32,40,64}`):

| nb | mode 2 max abs error | stock vs FP64 oracle | 2x stock | verdict |
|---|---|---|---|---|
| 10 | 3.815e-06 | 3.952e-06 | 7.90e-06 | PASS |
| 40 | 1.144e-05 | 9.797e-06 | 1.96e-05 | PASS |
| 64 | 6.104e-05 | 4.108e-05 | 8.22e-05 | PASS |

```
ALL CHECKS PASSED
```

### 0.4 `T4` — `ik_llama.cpp` fork hygiene and the Option A/B decision

**Base confirmed:** `HEAD = 8313388 "Fix iq4_ks_r16 quantization tail"` (2026-10-15), exactly the base
the brief names. The fork carries **9 modified files, 426 insertions, 14 deletions**:

| file | + lines | belongs to | profiler strings |
|---|---|---|---|
| `common/common.cpp` | 125 | `--cpu-mask-batch` / `--cpu-strict-batch` (parse + usage) | 0 |
| `common/common.h` | 7 | the two new `gpt_params` fields | 0 |
| `ggml/include/ggml.h` | 3 | affinity API declarations | 0 |
| `ggml/include/ggml-backend.h` | 1 | backend affinity setter | 0 |
| `ggml/src/ggml-backend.cpp` | 11 | CPU backend affinity plumbing | 0 |
| `ggml/src/ggml.c` | 197 | **both**: batch affinity (`set_cpu_thread_affinity`, cplan) **and** `BONSAI_PROF` (26 lines) | 26 |
| `include/llama.h` | 14 | context params for the batch pool | 0 |
| `src/llama-context.h` | 7 | context fields | 0 |
| `src/llama.cpp` | 75 | graph-compute plumbing for the batch pool | 0 |

**The profiler is compiled out of the shipping binary by build flag, not by source separation** — and
that was verified rather than assumed:

```
build-arm64-ik:      CMAKE_C_FLAGS = -march=armv8.2-a+dotprod+i8mm+fp16+bf16          (no BONSAI_PROF)
build-arm64-ik-prof: CMAKE_C_FLAGS = ... -DBONSAI_PROF=1
HOST_CHECK: "ik ship build has no profiler" -> OK   (grep -qa 'CPU OP PROFILE' returns nothing)
```

**Patch artifacts written and verified** (byte-identical copies):

```
MonsterQwenDroid/patches/ik_llama.cpp-moedroid.patch   723 lines  sha256 9d418b632958006a...
MoeDroid/patches/ik_llama.cpp-moedroid.patch           723 lines  sha256 9d418b632958006a...
```

Verification in a **clean worktree at `8313388`** (not in the working tree, so the test cannot be
confounded by the changes it is meant to describe):

```
git worktree add .stage10/ik-base 8313388
git apply --check <patch>        -> FORWARD CHECK OK
git apply <patch>                -> APPLIED (9 files)
git apply --reverse --check      -> REVERSE CHECK OK
```

#### Option A vs Option B

`MoeDroid/app/src/main/java/com/prismml/bonsaidroid/engine/LlamaServerEngine.kt:450` passes
`--cpu-mask-batch 0xff --cpu-strict-batch 1` whenever `pinBatchAllCores` is set, and `ink`'s
unpatched `8313388` does not parse those flags — its `gpt_params_parse` would exit with
`error: invalid argument`.

| | **Option A — keep the minimal CLI patch (chosen)** | **Option B — unpatched `8313388`,<br>drop `--cpu-mask-batch` in app config** |
|---|---|---|
| upstream ancestry | 8313388 + 3 hunks that only add two CLI flags and their plumbing | pristine 8313388 |
| app changes | none | `LlamaServerEngine.kt` must not emit the flags; a second flag-set path to maintain and test |
| measured cost of dropping the flags | none | **~+1%** (Stage 9, same protocol: `B1` vs `B0` decode 9.210/9.085 vs 9.135/9.133, prefill 11.698/11.942 vs 11.581/11.771) |
| failure mode if the flags are still passed | — | llama-server exits immediately at startup on every launch |
| profiler exposure | none in the shipping binary (compile flag, verified) | none |

**Option A.** The patch is three small, auditable hunks on top of a named commit; Option B saves
upstream lineage at the price of a ~1% regression and a silent-startup-failure mode in the shipped
app. The patch artifact contains only these two features; nothing else was ever added to the fork.

### 0.5 `T4.4` — the unreviewed `ChatScreen.kt` edit is backed up and reverted

The worktree's only unreviewed UI change was saved first, then reverted so that the Step 4
control-vs-adoption APK pair is built from a clean tree:

```
.stage10/ChatScreen-unreviewed.patch      (the full diff, 1 file)
MoeDroid/app/src/main/java/com/prismml/bonsaidroid/ui/ChatScreen.kt  -> restored to HEAD
```

Verification is recorded in §4.1 once Step 4 builds from that tree.

#### `BONSAI_MMID_BLK` bit-exactness (qemu-aarch64-static, shipping objects)

`s7_mmid_prefill_verify.sh` was extended with the five cells plus a repeat control each, and a
`BONSAI_MMID_BLK_DEBUG=1` parse proof so the matrix cannot pass vacuously. All 22 arms pass:

| shape (MUL_MAT_ID) | `K16` | `K64` | `K80` | `K108` | `K428` | overall |
|---|---|---|---|---|---|---|
| case0 16 tok / 8 thr | == stock | == stock | == stock | == stock | == stock | **all 5 cells bit-exact** |
| case1 16 tok / 8 thr | == stock | == stock | == stock | == stock | == stock | " |
| case2 32 tok / 8 thr | == stock | == stock | == stock | == stock | == stock | " |
| case3 16 tok / 1 thr | == stock | == stock | == stock | == stock | == stock | " |
| case4 4 tok / 8 thr | == stock | == stock | == stock | == stock | == stock | " |
| case5 **1 tok / 6 thr** | == stock | == stock | == stock | == stock | == stock | " |

```
case0..case5  ...  |all 5 cells bit-exact
PASS: BONSAI_MOE_X4_PREFILL=1 is bit-identical to the stock path, and the paired path
      demonstrably ran on 4 of 6 shapes
```

Coverage is asserted, not assumed: every cell repeats and must match itself, the stock arm must
report `x4t2_calls=0`, the debug line must read `BONSAI_MMID_BLK rows_per_block=428`, and the D1
cache shows `pk_hits=300 pk_misses=20` on the single-token shape.

**One pre-existing verifier defect was found and fixed on the way:** the Stage 9 node-alias case ran
unconditionally, but it requires the packed cache (`BONSAI_Q8_0_X4=2`); in the stock/prefill arms it
reported `S9NODE VACUOUS` and failed the arm. It is now skipped with a stated reason when the cache is
off, and still runs (and passes) in the `d1` arms. Its real gate, `s9_pk_node_alias.sh`, is unchanged.

### 0.6 Patch freshness debt (Stage 9 §3.3) — CLOSED

Stage 9 left `patches/llama.cpp-moedroid.patch` at 2026-10-18 02:37, i.e. **before** the `D1` kernel,
the PK cache and the Stage 10 changes: it contained no `bonsai_x4_rows4_mode1_packed`, no
`bonsai_pk_cached`, no `BONSAI_PK` and no `BONSAI_HYBRID_VIEW`. That is now repaired:

```
upstream-llama: git diff 1b70350  ->  18 files, 2550 insertions(+), 84 deletions(-)
MonsterQwenDroid/patches/llama.cpp-moedroid.patch   3083 lines  sha256 db7e5b0b452a23c7...
MoeDroid/patches/llama.cpp-moedroid.patch           byte-identical copy
```

Verified in a **clean worktree at `1b70350`** (created with `git worktree add`, not the working tree):

```
git worktree add .stage10/llama-base 1b70350
git apply --check <patch>     -> FORWARD CHECK OK
git apply <patch>             -> APPLIED (18 files)
git apply --reverse --check   -> REVERSE CHECK OK
```

The patch now contains every fork feature: the D1 packed four-row kernel, the per-node-invalidated PK
cache and its periodic reporter, `BONSAI_HYBRID_VIEW`, `BONSAI_MMID_BLK`, the per-type profiler with
the `GET_ROWS` exclusion, the `__aarch64__` guard, and the `BONSAI_GRAPH_NAMES` diagnostic.

---

## 1-4. Device phases — NOT RUN in this window

**The phone was not connected for the whole session.** `adb devices` reported no device and `lsusb`
showed no matching USB device (the GNOME session's `usb` scan is empty), so Step 1 (cool profiler
pair), Step 2 (endurance gate), Step 3 (insurance arms) and Step 4's device half were not executed.
Nothing is claimed about them, and no device number appears anywhere in this report.

This is a **host-only window by instruction**, decided with the operator rather than assumed: Steps 1-4
are gated on hardware that was absent, and reporting a partial device run as if it were the window
would be the same class of error the project keeps catching (a plausible result that was never
measured).

### 1.1 What IS ready for the device phase, verified

| item | state |
|---|---|
| `build-arm64-ship` / `build-arm64-prof` | fully rebuilt 13:41 from the current tree; `libllama.so` carries `BONSAI_HYBRID_VIEW`, `libggml-cpu.so` carries `BONSAI_MMID_BLK` (+ debug), ship has no profiler strings |
| Step 1 launcher | `s8_shootout.sh 1v` — `gated_pair(prof-ours-A0, prof-ours-V1, same)` on the 2.5% canary, then `prof-ik` recorded uncrossed (Stage 8 §2.3 established the cross-engine canary fails for a real kernel difference) |
| Step 2 launcher | `MoeDroid/tools/s10_endurance.sh` — preflight (3 x 10 s caps, 100% AC), `WARM_N=0 WARM_COOL_S=0`, 20 continuous decode + 6 continuous prefill requests per engine, per-request `soc0/soc1/cpuss/bt/bp` telemetry added to the arm's `S7REQ` line, cool-down between engines to baseline temperature, pre-declared `>= +12%` both arms |
| Step 3 arms | `V1`, `K64`, `K80`, `K108`, `K428` wired into both the identity phase and the timing matrix, plus `A0T` (a second tag for the shipped defaults at the tail of each pass) — the drift bracket is a tag, never an overwrite |
| Step 4 APK gate | `MoeDroid/tools/s10_apk_diff.sh`, self-tested in both directions: a lib-only change passes, a tampered `classes.dex` fails, and it refuses to pass when nothing was adopted |
| host gates after every change | `HOST_CHECK=1 s8_shootout.sh 0` = **0 failures**; `s8_ll_table.py --selftest` = PASSED; `s6_ppl_recheck.py --selftest` = 9/9 |

---

## 1. On-device cool-profiler pair (A0 vs V1) — counts obtained, ms withheld

Run at 23:09-00:09 (two attempts), `s8_shootout.sh 1v`, device awake/foregrounded, caps
`3532800/3840000`, `WARM_COOL_S=90`, 12 warmups + 6 measured, profiler `BONSAI_PROF_EVERY=1`.

### 1.1 The canary had to be re-scoped, and that is the finding

The same-engine canary treats `GET_ROWS`, `CPY` and `GATED_DELTA_NET` as invariant. **`V1` changes two
of them by construction**, so the first two attempts were refused for the wrong reason:

| attempt | canary set | `GET_ROWS` | `CPY` | `GATED_DELTA_NET` / `MUL_MAT` | verdict |
|---|---|---|---|---|---|
| 1 | default (GET_ROWS, GATED_DELTA_NET) | **−90.09%** | — | +14.13% | FAIL (clock) |
| 2 | default | −90.09% | — | +14.13% | FAIL (clock) |
| 3 | RMS_NORM, CPY, MUL_MAT | — | **−48.63%** | MUL_MAT +5.38% | FAIL (clock) |
| 4 | **RMS_NORM, MUL_MAT** | — | — | **MUL_MAT −10.39%**, RMS_NORM −29.69% | **FAIL (clock, real)** |

`GET_ROWS −90.09%` and `CPY −48.63%` are **the effect**, not a confound: `V1` deletes 73 of the 174
gathers and the 73 zero-row `ggml_cpy` nodes that went with them. The gate was therefore rewritten to
accept an explicit `--canary=` set (`s4_prof.py`), `phase1v` now passes `RMS_NORM,MUL_MAT`, and the
moved ops are read as results. This is the same failure class as Stage 9's drift bound: an assertion
that assumed the thing under test was invariant.

### 1.2 What IS quotable: the call counts are exact and clock-independent

| op | A0 calls/token | V1 calls/token | delta |
|---|---|---|---|
| `GET_ROWS` | **174** | **101** | **−73** |
| `CPY` | **186** | **113** | **−73** |
| `MUL_MAT` | 833 | 833 | 0 |
| `MUL_MAT_ID` | 144 | 144 | 0 |

This reproduces the host prediction exactly (174 → 101, §0.2) **on the device**, and adds one thing the
host census had hidden: **the zero-row state copies were not free — 73 `CPY` nodes per token were
being built and executed for nothing.** `V1` removes them along with the gathers.

### 1.3 What is NOT quotable, and why: the two arms ran ~10% apart in clock

| op | A0 ms/token | V1 ms/token | delta |
|---|---|---|---|
| `MUL_MAT` (untouched by V1) | 81.514 | 73.041 | **−10.39%** |
| `MUL_MAT_ID` (untouched) | 47.066 | 42.979 | −8.68% |
| `GET_ROWS` | 6.311 | 0.224 | (effect) |
| `CPY` | 3.193 | 1.247 | (effect) |

`MUL_MAT` and `MUL_MAT_ID` cannot both be ~10% faster because of a state-view change — they measure the
same code. The arms ran at different effective clocks, so **no ms/token delta from this pair is
published**, and the measured medians (A0 6.3456 vs V1 7.0265 t/s, +10.7%) are **not** attributed to
`V1` here: the clock shift is the same size as the apparent gain. The speed decision is Step 3's
mirrored passes, which exist precisely to cancel this.

Contributing causes, recorded: the arm sequence lost the screen once (device locked, user unlocked it),
and the V1 arm spent **280 s in its PRE cap gate** (28 polls) recovering the profile. `svc power
stayon true` was set for the remainder of the window.

### 1.4 The fixed per-type table, on device, both engines

Ours (A0), fixed instrument — **no `f32` row**, and `calls = mm + mmid` now closes:
`q2_0 202 = 58 + 144`:

| type | ms/token | calls | mm | mmid |
|---|---|---|---|---|
| q2_0 | 51.710 | 202 | 58 | 144 |
| q3_K | 24.140 | 90 | 90 | 0 |
| q8_0 | 14.600 | 194 | 194 | 0 |
| iq4_xs | 13.913 | 56 | 56 | 0 |
| q5_K | 10.970 | 16 | 16 | 0 |
| bf16 | 5.079 | 291 | 291 | 0 |
| q4_K | 5.038 | 38 | 38 | 0 |
| q6_K | 2.257 | 12 | 12 | 0 |
| q4_0 | 0.296 | 16 | 16 | 0 |

ik (`prof-ik`, same instrument): **106.710 ms/token total → 9.371 t/s**, `MUL_MAT` 68.237 ms / 703
calls, `MOE_FUSED_UP_GATE` 17.181 ms / 48, `MUL_MAT_ID` 8.774 ms / 48, `DELTA_NET` 4.381 ms / 36. Its
type table also carries **no f32 row**, which is the cross-check that the Stage 9 double-count is gone
rather than merely relabelled.


### 1.5 The pair's own retry could not run, and that is not a gate failure

The gated pair's attempt 2 (00:12:46) started while the phone was being unplugged for fast charging.
Both arms "completed" in 4 s with `S8PROF_GATE=FAIL: could not pull the engine log`, i.e. adb had no
device and no engine ever started. The gate refused the pair (correctly - a missing log is not a
matched clock), so nothing vacuous was recorded. The 23:54 attempt is the one analysed above.

**Status of Step 1: counts delivered, ms withheld.** The device side of §1 needs no re-run for its
counts; the ms attribution would need a *cool, undisturbed, matched-clock* pair, which the Step 3
mirrored arms supersede anyway for the speed question.

---

## 2. Unconditional cool-device endurance gate — **FAIL by 1.59 pp on prefill**

Protocol exactly as pre-registered: preflight caps held 3 polls, **battery 100%**, `WARM_COOL_S=0`,
`WARM_N=0` (every request is part of the endurance run), per-request caps + `cpu-0-0-0` /
`cpu-0-0-1` / `cpuss-0-0` + battery temperature + battery capacity, and a full cool-down between
engines (250 s and 230 s, back to 39.1 / 38.7 C, caps at profile).

### 2.1 Arm 2A — 20 continuous decode requests

| request | ours t/s | ours soc0 | ours cpuss | ik t/s | ik soc0 | ik cpuss |
|---|---|---|---|---|---|---|
| 15 | 6.81 | 92.3 | 73.6 | 9.267 | 99.3 | 98.5 |
| 16 | — | — | — | 9.336 | 99.3 | 96.9 |
| 17 | 6.79 | 92.7 | 74.1 | 8.951 | 100.4 | 98.9 |
| 18 | — | — | — | 9.147 | 100.4 | 98.5 |
| 19 | 6.81 | 92.3 | 73.6 | 9.246 | 100.8 | 98.1 |
| 20 | 6.47 | 93.8 | 78.3 | 9.297 | 97.7 | 100.4 |
| **median 15-20** | **6.7945** | 93.8 peak | 78.3 peak | **9.2564** | 100.8 peak | 100.4 peak |

**Decode: +36.23%** (bar +12.0%) - PASS, with the advantage surviving deep load and with ik running
**hotter** than ours (100.4 C vs 93.8 C on `cpuss`), so it is not a thermal head start.

The in-engine profiler independently agrees: ik's per-op budget sums to **106.710 ms/token = 9.371
t/s**, within 1.2% of the arm median.

### 2.2 Arm 2B — 6 continuous 262-token prefill requests

| arm | median prefill t/s | final soc0 / soc1 / cpuss | battery |
|---|---|---|---|
| ours | **10.7170** | 95.0 / 90.7 / 80.6 C | 36 C / 98% |
| ik | **11.8322** | 100.8 / 102.0 / 97.7 C | 36 C / 98% |
| **delta** | **+10.41%** | | |

**Prefill: +10.41%** against a +12.0% bar - **FAIL by 1.59 percentage points.**

### 2.3 Verdict

```
S10DEC ours=6.7945 ik=9.2564 delta=+36.23%
S10PRE ours=10.7170 ik=11.8322 delta=+10.41%
```

**Adoption gate: FAIL.** The rule was fixed before the run ("PASS if and only if >= +12.0% on BOTH"),
and prefill does not reach it. **No `adoption.apk` is built and nothing is installed from ik.** The
threshold is not moved after the fact: a +10.41% prefill is close to the bar, but "close" is exactly
what the pre-declared number exists to arbitrate, and the next window can re-run it rather than this
one redefining it.

Two observations worth carrying forward, recorded as observations rather than as a reinterpretation:

* the prefill gap (+10.4%) is far smaller than the decode gap (+36.2%), and both engines now run
  well above the Stage 9 arm-protocol prefill figures (8.60 / 11.70), which the cool start and
  `throttled=0` window explain;
* ik pays for its decode advantage in heat: it peaked 6.6 C higher on `cpuss` under the same load.


---

## 3. Insurance fork arms: identity PASS, matrix running

### 3.1 Identity gates — 10/10 PASS, nothing dropped

Arms: `V1` (`BONSAI_HYBRID_VIEW=1`) and `K64/K80/K108/K428` (`BONSAI_MMID_BLK`), each run on the
short prompt and on the real 262-token prefill, against the shipped engine in the same session.

```
IDENTITY PASS (V1 vs shipped) / (V1 vs shipped, 262-token prefill)
IDENTITY PASS (K64 vs shipped) / (K64 vs shipped, 262-token prefill)
IDENTITY PASS (K80 vs shipped) / (K80 vs shipped, 262-token prefill)
IDENTITY PASS (K108 vs shipped) / (K108 vs shipped, 262-token prefill)
IDENTITY PASS (K428 vs shipped) / (K428 vs shipped, 262-token prefill)
IDENTITY PASS (A0 repeatability (short)) / (A0 repeatability (262-token prefill))
```

Both features are therefore bit-exact on device as well as under qemu: `V1` because it replaces a
provably identity gather, and `BONSAI_MMID_BLK` because `blck_0` changes only the loop granularity
(every output row is still one `vec_dot`).

### 3.2 Mirrored matrix — in progress

`ARMS="A0 V1 K64 K80 K108 K428 A0T"`, 7 arms x 4 passes = 28 timed arms, `WARM_COOL_S=90`,
`throttled` enforced per arm, and `A0T` (a second tag for the shipped defaults, run last in each pass)
brackets the pass-to-pass drift so a head-vs-tail comparison can be attributed to thermal state rather
than to the candidate.

Started 01:28:42. A device watchdog (`.stage10/watchdog.sh`, managed job) samples wakefulness, screen
state, caps and battery every 10 minutes and re-wakes the phone if it locks, because one lock earlier
in the window stalled a cap gate for 280 s and skewed an arm's clock.

Results are recorded in §3.3 when the matrix completes.


### 3.3 Matrix result: `V1` clears the bar, no `K` cell does

28 timed arms, 01:28-05:20. **Every decode arm is valid**: `6/6` measured, `throttled=0/6`. Prefill
arms read `2-3/6` throttled, which is the documented prefill regime (the trip lands inside the window);
their medians are quoted with that column, exactly as Stage 8/9 did.

**Decode** (median t/s, delta vs that pass's own `A0`):

| arm | pass 1 | pass 2 | Δ pass 1 | Δ pass 2 | positive in both? |
|---|---|---|---|---|---|
| `A0` | 6.8233 | 6.6483 | — | — | reference |
| `A0T` (tail bracket) | 6.6673 | 6.6399 | −2.29% | −0.13% | control |
| **`V1`** | **7.1166** | **7.0942** | **+4.30%** | **+6.71%** | **YES** |
| `K64` | 6.5030 | 6.8929 | −4.69% | +3.68% | no |
| `K80` | 6.7202 | 6.7287 | −1.51% | +1.21% | no |
| `K108` | 6.5469 | 6.6306 | −4.05% | −0.27% | no |
| `K428` | 6.6643 | 6.7465 | −2.33% | +1.48% | no |

**Prefill** (median t/s, delta vs that pass's own `A0`):

| arm | pass 1 | pass 2 | Δ pass 1 | Δ pass 2 | positive in both? |
|---|---|---|---|---|---|
| `A0` | 8.5644 | 8.7543 | — | — | reference |
| `A0T` | 8.6195 | 8.7527 | +0.64% | −0.02% | control |
| **`V1`** | **8.6236** | **8.8041** | **+0.69%** | **+0.57%** | **YES** |
| `K64` | 8.6389 | 8.9798 | +0.87% | +2.58% | YES (but decode fails) |
| `K80` | 8.5056 | 8.7222 | −0.69% | −0.37% | no |
| `K108` | 8.8443 | 8.7542 | +3.27% | −0.00% | no |
| `K428` | 8.8703 | 8.7401 | +3.57% | −0.16% | no |

**The `A0T` drift bracket does its job.** The shipped defaults, run at the head and at the tail of the
same pass, differ by **−2.29% / −0.13% decode** and **+0.64% / −0.02% prefill**. That is the pass-to-pass
position effect, and `V1`'s decode gain (+4.30% / +6.71%) is larger than it in both passes and has the
same sign in both - so the win is not junction drift. The bracket lands inside the 2.5% noise floor,
which assigns the drift to branch **(iii) neither** of the three pre-registered explanations: no
junction-thermal or skin-heat signature was visible with the phone on a 100 W charger.

**`V1` is the only arm that wins both mirrored passes on its targeted phase and is neutral-or-better on
the other.** Its mechanism is measured, not inferred: 73 fewer `GET_ROWS` and 73 fewer `CPY` nodes per
token (§1.2), i.e. ~118 MB/token of state traffic and 73 useless graph nodes removed.


---

## 4. Execution of the decisions

### 4.1 `BONSAI_HYBRID_VIEW` ships as the default — done and verified on device

Flipped in `llm_graph_context`'s helper: **unset -> 1, `=0` -> 0** (the gather is kept as the control):

```cpp
// Default ON from Stage 10: it won both mirrored decode passes (+4.30% / +6.71%) and is
// neutral-or-better on prefill (+0.69% / +0.57%), and it is bit-exact on device.
mode = (!e || atoi(e) != 0) ? 1 : 0;
```

`build-arm64-ship` and `build-arm64-prof` were rebuilt in full. **Verified on the phone with no
`BONSAI_*` set at all** (the arm's own `S7ENV` line shows every variable `unset`):

```
S7ENV tag=checkdflt … vars=BONSAI_MOE_X4=unset,BONSAI_MOE_X4_PREFILL=unset,BONSAI_Q8_0_X4=unset,
                           BONSAI_MULMATID_FLAT=unset,BONSAI_Q3K_X4=unset,BONSAI_PROF_EVERY=1
GET_ROWS per graph (last 4): [101, 101, 101, 101]
VERDICT: default ON (101/graph)
```

101/graph is the signature of the view; the gather path would read 174. Token identity for the default
is already covered: the `id-A0` arm at this build's predecessor produced `8ad776c2…` and `V1` produced
the same md5 in §3.1.

### 4.2 `BONSAI_MMID_BLK` stays at its default 16 — four cells rejected

| cell | decode both passes | prefill both passes | verdict |
|---|---|---|---|
| `K64` | no (−4.69% / +3.68%) | yes | **rejected** — cannot pass the decisive phase |
| `K80` | no (−1.51% / +1.21%) | no | rejected |
| `K108` | no (−4.05% / −0.27%) | no (+3.27% / −0.00%) | rejected |
| `K428` | no (−2.33% / +1.48%) | no (+3.57% / −0.16%) | rejected |

No default change. The cells remain available behind the env var, all bit-exact (§0.3, §3.1). The
interesting negative is `K64`/`K428`: widening the expert inner block helps **prefill** slightly and
does nothing for **decode**, which is the opposite of what ik's static 6-chunks/expert shape suggested -
so the decode gap is not closed by loop granularity alone.

### 4.3 `ik_llama.cpp` adoption: NOT executed

Step 2's pre-declared gate (`>= +12.0%` on **both** the decode req-15-20 median and the 6-request
prefill median) came out **+36.23% / +10.41%** -> **FAIL on prefill by 1.59 pp**. Therefore, per the
brief, **no `control.apk`/`adoption.apk` pair was built, `s10_apk_diff.sh` was not run on device
artefacts, nothing was installed, and the phone's app is untouched.** The diff tool remains ready and
self-tested (§0 readiness table) for the window that clears the bar.

### 4.4 Patches — regenerated and verified against the base commit

| patch | files | verification |
|---|---|---|
| `patches/llama.cpp-moedroid.patch` (both trees, byte-identical) | 18 | `git apply --check` **FORWARD OK**, applies, `--reverse --check` **REVERSE OK** in a clean `1b70350` worktree |
| `patches/ik_llama.cpp-moedroid.patch` (both trees, byte-identical) | 9 | same procedure against a clean `8313388` worktree (§0.4) |

The llama patch now contains the `V1` default itself, so rebuilding the app from a clean
`1b70350 + patch` yields the engine that was measured tonight.

### 4.5 What this window did NOT do

* No APK was built, installed or replaced; `MoeDroid`'s `jniLibs` were not synced. **The phone's app
  still runs the Stage 7 library**, so `V1`'s +5% is in the engine build, the patch and the harness,
  not yet on the user's launcher icon. That is the honest state and the obvious next step.
* `T2`'s ms/token attribution (GET_ROWS/CPY deltas) is still withheld: the clocks did not match in
  either profiler attempt (§1.3). The arm-level timing in §3.3 does not depend on it.
* `iterations.jsonl` entries 87-89 record this window.


---

## 5. Phone release state

| item | state |
|---|---|
| harness engines | none (`ps -A | grep libllama` shows only the app's own PID 5073) |
| port 8299 | no socket at all (the TIME_WAIT entry expired) |
| `stay_on_while_plugged_in` | **0** (reverted) |
| adaptive battery management | **1** (restored - it was disabled only so the phone could reach 100%) |
| screen timeout | 60 s (restored) |
| battery | 74%, Charging |
| **keyguard** | **showing - the phone locked itself again, and a PIN is set, so I cannot dismiss it.** `com.prismml.moedroid` is the top activity behind the keyguard; one unlock by the owner brings it straight back. Recorded rather than worked around: guessing at unlock gestures on someone's locked phone is not a thing an agent should do. |

The device watchdog (which re-woke the phone every 10 min during the matrix) was stopped before the
release, otherwise it would have fought the stand-down.

## 6. Summary

| step | outcome |
|---|---|
| **0. host prep (T1-T4)** | per-type instrument verified and both arm64 trees rebuilt; `BONSAI_HYBRID_VIEW` wired (174 -> 101 `GET_ROWS`, bit-identical on host); `BONSAI_MMID_BLK` wired in the expert path only (dense untouched, 5 cells bit-exact on 6 shapes under qemu, one stack overflow found and fixed); Gate 3(b) recalibrated to a measured ceiling (2.8893e-2 oracle drift -> 5e-2) and Gate 3(a) made oracle-relative; both fork patches written and verified in clean worktrees |
| **1. cool profiler pair** | call counts delivered (`GET_ROWS` 174->101, `CPY` 186->113 on device) and the fixed per-type table produced with no `f32` row; ms attribution **withheld** - the canary caught a 10% clock split, and the gate itself had to be re-scoped because `GET_ROWS`/`CPY` are the effect, not invariants |
| **2. endurance gate** | decode **+36.23%** (ours 6.7945 -> ik 9.2564), prefill **+10.41%** (10.7170 -> 11.8322) -> **adoption FAIL by 1.59 pp**, no APK built or installed |
| **3. insurance arms** | identity **10/10 PASS**; matrix 28 valid arms: **`V1` wins both decode passes (+4.30% / +6.71%)** and is neutral-or-better on both prefill passes; no `K` cell wins both decode passes; `A0T` bracket +-2.3% -> drift assigned to "neither" |
| **4. decisions** | `BONSAI_HYBRID_VIEW` flipped to default ON, ship+prof rebuilt, **verified on device at 101 `GET_ROWS`/graph with every `BONSAI_*` unset**; `MMID_BLK` default unchanged; patches regenerated (18 files) and re-verified against `1b70350` |

**Net for the fork, in one line: `V1` ships as the default for a measured +5.5% decode (both mirrored
passes) at identical output, and ik still does not clear the adoption gate because its prefill gain is
1.59 pp short.**

**The one thing that is not done:** the app on the phone still carries the Stage 7 library, so the +5%
is in the engine build, the patch and the harness rather than on the launcher icon. Syncing
`jniLibs/arm64-v8a/` and building an APK would put it there, and that is the obvious first act of the
next window.

---

## 7. Build 12 — `V1` delivered to the launcher

Executed exactly as specified, with every step verified rather than assumed.

| step | result |
|---|---|
| 1. sync native libs -> `jniLibs/arm64-v8a/` | `libllama.so`, `libggml-cpu.so`, `libggml.so` copied; **`libllama.so` = 44,356,104 B dated 2026-10-20 09:20**, matching the build's own figure |
| 2. version bump | `versionCode 11 -> 12`, `versionName "0.2.0-pinned" -> "0.3.0-hybridview"` |
| 3a. `:app:compileDebugKotlin --offline` | **BUILD SUCCESSFUL** (19 s, 7 executed / 8 up-to-date) |
| 3b. `:app:assembleDebug --offline` | **BUILD SUCCESSFUL** (8 s); `app-debug.apk` 121,051,958 B |
| 3c. copy to `apk-out/` | `MoeDroid-0.3.0-hybridview-12-arm64.apk`, 121,051,958 B, sha256 `926eb46e27dd12e8…` |
| 3d. `adb install -r` | **Success** (streamed install) |
| 3e. `dumpsys package` | **`versionCode=12`**, **`versionName=0.3.0-hybridview`**, `lastUpdateTime=2026-10-20 09:21:33` |

**Only three libraries actually differed**, so the ABI mix is provably clean. Comparing every entry in
`jniLibs` against `build-arm64-ship/bin`:

```
libggml-base.so   identical     libllama-common.so       identical
libggml.so        identical     libllama-server-impl.so   identical
libllama.so       DIFFERS  44352504 -> 44356104   (BONSAI_HYBRID_VIEW default)
libggml-cpu.so    DIFFERS   5700120 ->  5766648   (Stage 9/10 CPU changes)
libllama-cli-impl.so / libmtmd.so identical; libllamaserver.so == ship llama-server (byte-identical)
```

### 7.1 Verification that the shipped bytes are the measured bytes

1. **Inside the APK**: `unzip`-ed `lib/arm64-v8a/libllama.so` and `libggml-cpu.so` are byte-identical to
   the ship build (44,356,104 / 5,766,648) - so the build packed the new libraries rather than a stale
   copy from Gradle's cache.
2. **Installed on the device**: `/data/app/~~W1qNf…==/lib/arm64/libllama.so` hashes
   `a8c15f409cc25e88e1c32854ef8a5203cb158ae94798c89ecbddf42a16128d0d`, **identical to the ship build's
   hash**, and the sizes match to the byte.
3. **Runs, headlessly, from the installed library directory** (so this is the shipped set, not the
   harness set): engine healthy after 52 s, model loaded, and it generated correctly at temp 0:

```
engine pid=598 (waited 52s for health)
{"content":" Paris.", ... "stop":true, "tokens_predicted":3, "prompt_n":5}
```

(The 1.2-1.6 t/s in that response is a cold first request predicting three tokens; it is a functional
check, not a measurement, and no number from it is quoted as a result.)

### 7.2 What is and is not visually confirmed

`com.prismml.moedroid` is installed at version 12 and its process starts **without a crash and without
any `UnsatisfiedLinkError`** in logcat. The activity could not be brought to the foreground for a
visual check because the phone's keyguard is up and a PIN is set: focus stays on `NotificationShade`,
so the app never resumes and never starts its own engine. **One unlock by the owner shows Build 12.**

**Device after the install:** no harness engine running (only the app's own process, idle), battery
**100%**, screen off / dozing as normal.


---

## 8. Build 13 — the UI fix, shipped

The recipe only bumped the version, so before building I checked what the worktree actually contained:
the UI fix was **already present** (`ChatScreen.kt` +62 insertions, no conflict markers), including the
readout that cannot silently vanish:

```kotlin
// Speed is always rendered. The old form hid the chip whenever ...
val pps = timings?.predictedPerSecond ?: 0f
if (pps > 0.01f && pps < 5000f) "%.2f tok/s".format(pps)
else "tok/s —"
```

| step | result |
|---|---|
| version bump | `versionCode 12 -> 13`, `versionName "0.3.0-hybridview" -> "0.3.1-hybridview-ui"` |
| `compileDebugKotlin --offline` | **BUILD SUCCESSFUL** (18 s) |
| `assembleDebug --offline` | **BUILD SUCCESSFUL** (8 s) |
| artifact | `apk-out/MoeDroid-0.3.1-hybridview-ui-13-arm64.apk`, **121,057,782 B** (+5,824 vs Build 12 - consistent with a UI-only dex delta) |
| `adb install -r` | **Success** |
| `dumpsys package` | **`versionCode=13`**, **`versionName=0.3.1-hybridview-ui`**, `lastUpdateTime=2026-10-20 09:48:59` |

**Three checks that the shipped bytes are what was intended:**

1. **The new UI code is in the packaged dex**: `strings classes*.dex` contains the new format strings
   (`%.2f tok/s`, `%.0f tok/s`), so the readout path really shipped rather than a cached dex.
2. **The engine was not disturbed**: `lib/arm64-v8a/libllama.so` and `libggml-cpu.so` inside the APK are
   still byte-identical to the ship build, and the installed
   `/data/app/~~LCEGpn5fQdkfuo6NeI6UeA==/…/lib/arm64/libllama.so` hashes **`a8c15f40…`** - the same hash
   Build 12 installed. Build 13 adds the UI on top of the measured engine.
3. **It starts cleanly**: launched after the install, PID alive at 20 s, and logcat shows **no
   `FATAL`, `AndroidRuntime`, `UnsatisfiedLinkError` or `ClassNotFoundException`**.

**What is still not visually confirmed, and it is the same limitation as Build 12:** the phone's
keyguard is up with a PIN set (`mWakefulness=Dozing`, focus on `NotificationShade`), so the activity
cannot resume and the readout cannot be exercised. The app process starts without error; one unlock
shows Build 13 and the new speed chip.

**Housekeeping note:** `.stage10/ChatScreen-unreviewed.patch` (the Stage 10 backup) is now **stale** - it
holds the older 52-insertion edit, while the tree carries the newer 62-insertion one with the readout
guard. Do not restore the backup over the current file.

