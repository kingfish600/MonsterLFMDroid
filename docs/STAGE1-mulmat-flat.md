# Stage 1 report: flat static row partitioning in the dense single-token `mul_mat`

Date: 2026-10-14. Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite, CPU only.
Engine: fork of llama.cpp `1b70350` at commit `10288ca`, arm64 build
`armv8.2-a+dotprod+i8mm+fp16+bf16`, no profiler.
Shipping run flags held constant in every arm:
`-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1 --poll 0 -c 8192 -fa on
--cache-type-k q8_0 --cache-type-v q8_0 -ngl 0`, 64-token greedy decode.

## 1. What was changed

`ggml/src/ggml-cpu/ggml-cpu.c` (note: the dense `mul_mat` forward lives in `ggml-cpu.c` in this
revision, not in `ops.cpp`; `ops.cpp` only has the dispatch macro).

Stock, for one token (`nr1 == 1`), splits `ne01` into 64-row chunks that all threads steal from
`params->threadpool->current_chunk`, so each thread jumps between slices. The new path gives thread
`ith` one contiguous slice `[ith*per, min(ith*per+per, ne01))` with `per` rounded up to a multiple
of the vec_dot row group, and issues `__builtin_prefetch(src0_row + ir0*nb01 + 4096)` every 8 rows.

Gated by `BONSAI_MULMAT_FLAT` (default **0**). `BONSAI_MULMAT_FLAT_REPORT=1` prints
`flat path engaged` on the first flat call and a call count at exit, so a null A/B can be told
apart from a path that never ran. Gate condition: `ne11==ne12==ne13==1 && ne01 >= 64 && !use_ref`.

## 2. Correctness gate: token identity — PASS

The change only re-assigns which thread computes which output row, so with one token the arithmetic
must be untouched. Greedy 64-token output, same prompt, `temperature 0`, `seed 1`:

| run | md5 of generated text |
|---|---|
| block 1 a1 (OFF), b1 (ON), b2 (ON), a2 (OFF) | `20e5b49647ecfe3cfcf354b9e15ce587` |
| block 2 a1 (OFF), b1 (ON), b2 (ON), a2 (OFF) | `20e5b49647ecfe3cfcf354b9e15ce587` |
| final opt-in build, default (OFF) and `BONSAI_MULMAT_FLAT=1` (ON) | `20e5b49647ecfe3cfcf354b9e15ce587` |

All ten runs are bit-identical, 321 characters each.

## 3. Mirrored A/B/B/A on device

Two blocks, every arm recorded before and after. All eight arms ran on **one cap profile**:
`cpu0-5 = 3532800` (uncapped), `cpu6/7 = 3840000` of a `4473600` hardware max (the known persistent
cap). No arm changed profile while running.

| block | arm | MULMAT_FLAT | flat path engaged | decode t/s | minflt | majflt |
|---|---|---|---|---|---|---|
| 1 | a1 | OFF | 0 | 4.7122 | 180,276 | 32,154 |
| 1 | b1 | ON | 1 | 4.7805 | 177,902 | 31,762 |
| 1 | b2 | ON | 1 | 4.6027 | 175,847 | 32,507 |
| 1 | a2 | OFF | 0 | 4.7977 | 177,545 | 31,727 |
| 2 | a1 | OFF | 0 | 4.7917 | 180,375 | 31,934 |
| 2 | b1 | ON | 1 | 4.8492 | 176,415 | 31,355 |
| 2 | b2 | ON | 1 | 4.8280 | 173,625 | 30,837 |
| 2 | a2 | OFF | 0 | 4.4450 | 178,291 | 31,540 |

Same-profile mirrored pairs (the only comparable unit):

| pair | OFF | ON | delta |
|---|---|---|---|
| block 1 a1/b1 | 4.7122 | 4.7805 | **+1.45%** |
| block 1 b2/a2 | 4.7977 | 4.6027 | **-4.06%** |
| block 2 a1/b1 | 4.7917 | 4.8492 | **+1.20%** |
| block 2 b2/a2 | 4.4450 | 4.8280 | **+8.62%** |

Pairs: mean **+1.80%**, sd 5.21%, SE 2.60%, t = 0.69 (df 3). Group means: OFF 4.687, ON 4.765,
**+1.67%**. The pairs disagree in sign and the point estimate sits well inside the spread.

`majflt` (≈31k per 64 tokens, ~490/token) and `minflt` (≈178k per 64 tokens) are unchanged between
arms — unlike the `mul_mat_id` flat change, which collapsed major faults 719 → 318/token and gained
+4.8%. No mechanism signature here.

## 4. Why it is neutral, not a dead path

The path is genuinely live: ~420 dense `mul_mat` calls per decode token take it (20,172 calls over
8 decode steps at `nth = 6`).

The stock loop only *steals* chunks when `nchunk0 * nchunk1 >= nth * 4`, i.e. when
`ne01 >= 4 * nth * 64 = 1536` at `nth = 6`. Below that it re-chunks to `nchunk0 = nth` and every
thread already gets exactly one contiguous chunk, so there the new code adds only the prefetch.
Tensor inventory of the model (non-expert, 2-D):

| bucket | tensors |
|---|---|
| take the flat path (`ne01 >= 64`) | 608 |
| of those, the row partition really changes (`ne01 >= 1536`) | 319 |
| of those, only the prefetch differs | 289 |

The partition-changing set includes `output.weight` (`ne01 = 248,320`), the 48 x `hc_attn_up` /
`hc_ffn_up` (`ne01 = 10,240`), 36 x `attn_qkv` (`ne01 = 10,240`) and 36 x `ssm_out` / 48 x
`ffn_down_shexp` (`ne01 = 2,560`), so the change covers real dense work — it just does not move
the needle.

## 5. The real finding: the noise floor

Repeating the **same** arm in a later block moved decode by up to **7.9%** (`a2`: 4.7977 then
4.4450) with an identical cap profile before and after the arm. A matching `scaling_max_freq` is
necessary but not sufficient — it does not imply a matching thermal state.

Consequences:

* A 4-arm mirrored block cannot resolve the "few percent" this project keeps hunting. Resolving 2%
  at this spread needs ~16 pairs; Stage 1 had 4.
* The arm-to-arm variance (5-8%) is larger than the effect class being measured (~2%), so the
  mirrored design removes *bias* but not *variance*.
* A better harness for env-gated kernel switches: read the flag per call and alternate A/B
  **inside one process** on back-to-back requests. That deletes the model reload (~30 s, the main
  heat source) and most of the thermal term.

## 6. Decision and state

Per the Stage 1 rule ("commit if positive, or gate behind env if neutral"):

* `BONSAI_MULMAT_FLAT` **defaults to 0**. The code is committed at `10288ca` and available
  opt-in; the shipping app does not get an unproven change.
* Token identity is bit-exact, so there is no correctness risk either way.
* `MonsterQwenDroid/patches/llama.cpp-moedroid.patch` regenerated (8 files, reverse-applies
  cleanly against `1b70350`).

## 7. Recommendation for Stage 2

Do **not** spend more device time on this change as specified. If a dense-path win is still
wanted, the cheapest decisive experiment is a single-process A/B harness (~30 lines): one server,
two env-gated modes read per call, alternating requests. That measures the same code with the
thermal term removed and would settle a 2% question in one session.

Otherwise proceed to the Stage 2 candidates already queued: the 8-core `0xff` test, or the C++
`llama-quantize` dense `BF16 -> Q8_0` rewrite (which needs the `--tensor-type` regex list wired
into `src/llama-quant.cpp` and `only_copy` bypassed for matched tensors).

## 8. Reproducing

* Harness: `MoeDroid/tools/mulmatflat_arm.sh` (one arm), `mulmatflat_run.sh` (A/B/B/A driver),
  `mulmatflat_table.py` (table + same-cap pairs), `mulmatflat_identity.py` (greedy-text md5),
  `mulmat_flat_coverage.py` (GGUF tensor inventory).
* Device library set: `/data/local/tmp/lib-flat` on the phone, built from
  `upstream-llama/build-arm64-ship`.
* Raw logs: `mf-ab-block1.log`, `mf-ab-block2-full.log`; responses in `.mf-arms/`, `.mf-arms2/`.
