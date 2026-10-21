# Stage 2 Part B report: surgical dense rewrite via `llama-quantize ... COPY`

Date: 2026-10-14. Host: 32-core x86_64. Engine: fork `10288ca` + the quantizer patch below.

## 1. Correction to the handoff first

Handoff §4.1 concluded that `--tensor-type` is dead code because
"`src/llama-quant.cpp` never consults it - grepping that file for `tensor_types` returns nothing".
That grep used the wrong symbol. The regex list is plumbed and consulted, under the name
`tt_overrides`:

* `tools/quantize/quantize.cpp:546` builds `t_params.tt_overrides` from `--tensor-type` / `--tensor-type-file`.
* `src/llama-quant.cpp:194` compiles them into `qs.tensor_type_patterns` once.
* `src/llama-quant.cpp:753-766` applies them ("applying manual override").

The only real blocker was `src/llama-quant.cpp:289`: `tensor_allows_quantization()` returns `false`
immediately under `only_copy`, so `llama_tensor_get_type()` is never called and the override block is
unreachable in `COPY` mode. That is a ~30-line fix, not a rewrite of the quantiser.

## 2. The patch

`src/llama-quant.cpp`, two places:

1. `tensor_allows_quantization()`: under `only_copy`, return `true` only for a 2-D tensor whose name
   matches a `tt_overrides` pattern and whose requested type differs from its current type. Everything
   else stays `false` and is raw byte-copied.
2. `llama_tensor_get_type()`: an explicit `only_copy` branch that returns the matched override type
   directly, via `tensor_type_fallback()` for shape safety. This is required because `COPY` sets
   `ftype = ml.ftype` and `default_type` can be `F32`; letting the normal path run would have
   *widened* matched tensors to F32 instead of quantising them. The mixture logic and `default_type`
   are deliberately bypassed.

Every converted tensor logs `COPY override: <old> -> <new>`.

## 3. What was converted

```
llama-quantize --allow-requantize --keep-split \
  --tensor-type '^output\.weight$=q4_0' \
  --tensor-type 'hc_.*_(up|down)\.weight$=q8_0' \
  <shard1> out.gguf COPY 16
```

| | |
|---|---|
| tensors converted | **193** = 192 x `blk.N.hc_{attn,ffn}_{up,down}.weight` (BF16 -> Q8_0) + 1 x `output.weight` (Q5_K -> Q4_0) |
| tensors raw-copied | 1031 |
| fallbacks | 0 |
| model size | 63336.23 MiB (3.00 BPW) -> **62697.95 MiB (2.97 BPW)** |
| saved on shard 1 | **638.3 MiB** (37,623,740,192 -> 36,954,453,792 bytes) |
| wall time | 88.7 s |

Untouched as required: all 48 `ffn_gate_inp.weight` routers, all 1-D norms/injects, all `Q2_0`
experts, both embedding tables. `hc_*_inject.weight` (`ne00 = 4`) was not matched by the regex and
could not have been quantised anyway.

Note: the handoff's "290 BF16 tensors / 1,328 MB" figure does not match the given regex. Measured
inventory of shard 1: 483 BF16 tensors / 1407.1 MiB total, of which the regex matches 192 / 1200.0
MiB. The remainder is 48 routers (125.8 MiB, must not move), 96 `hc_*_inject` (tiny, `ne00=4`),
`ssm_alpha`/`ssm_beta`, `indexer.*_proj`, `ple_value`, and `output_hc_up/down`.

## 4. Integrity checks - all pass

| check | result |
|---|---|
| output shard 1 tensor count | 1223, identical name set to source |
| tensors whose type/bytes changed | exactly 193, transitions `{Q5_K->Q4_0: 1, BF16->Q8_0: 192}` |
| unexpected changes | none |
| GGUF metadata | identical (no differing keys) |
| **output shard 2 vs source shard 2** | **byte-identical, sha256 `316b46f3...c161e113`** (the free integrity check from handoff §6) |
| source shard 1 | unchanged, sha256 `69820c02...b199b720` |
| new shard 1 | sha256 `1d719728...c5258a008` |

## 5. Host validation gate: **FAILS the <= 0.5% perplexity gate**

Greedy generation (`--temp 0`, "The capital of France is", 32 tokens) is byte-identical between the
two models on the host: 37 chars, md5 `26f04bc00c68dabd6e5a82d398f3b502`. Coherent on both.

Perplexity on wikitext-2 raw, `-c 512`. The absolute estimate carries +/- 4.3% at 8 chunks, so the
comparison is done **paired per chunk** (same text, so the text variance cancels):

| variant | MiB saved | PPL (32 chunks) | paired delta | SE | t | chunks worse |
|---|---|---|---|---|---|---|
| original (baseline) | 0 | 2.7680 | - | - | - | - |
| **varH**: `hc_*` BF16 -> Q8_0 only | **562.5** | 2.9036 | **+0.092%** | 0.091% | 1.02 | 26/32 |
| **varO**: `output.weight` Q5_K -> Q4_0 only | 75.8 | 2.9174 | **+0.814%** | 0.085% | 9.52 | 31/32 |
| **full rewrite** (both) | **638.3** | 2.9253 | **+0.852%** | 0.110% | 7.72 | 31/32 |

An 8-chunk run of the full rewrite gave only +0.408%, which is why the gate looked like a pass at
first: the first chunk improves (-1.74%) and 8 chunks are too few. 32 chunks give +0.852% with
t = 7.72 and 31 of 32 chunks worse - unambiguous.

**The attribution is clean and additive:** `hc_*` costs +0.09% for 88% of the bytes saved;
`output.weight` costs +0.81% for 12% of the bytes. `varH` passes the gate comfortably;
`output.weight Q5_K -> Q4_0` is a bad trade and should be dropped.

## 6. On-device A/B

Both models live on `/sdcard`. Harness: `MoeDroid/tools/mqdense_arm.sh` (one arm: load, a
mode-shaped throwaway warmup, then the measured request) and `mqdense_run.sh` (mirrored A/B/B/A).
Shipping flags throughout: `-t 6 -tb 8 --cpu-mask 0x3f --cpu-strict 1 --poll 0 -c 8192 -fa on
--cache-type-k q8_0 --cache-type-v q8_0 -ngl 0`.

### 6.1 A storage trap that cost most of the session - read this before the next device run

The first attempt pushed the rewritten model to `/data/local/tmp/mf-model/` and it ran at
**0.09 t/s** (11 s/token) with `read_bytes` climbing at 2.9 GB/s and reaching **1.9 TB** - total
page-cache thrash, while the untouched shipping model on `/sdcard` ran at 4.49 t/s under the
identical harness and flags.

The bytes are not the problem. `sha256sum` on the device matched the host for both shards, and
`dd` read the `/data/local/tmp` copy sequentially at 2.8 GB/s. **Copying the same file to `/sdcard`
and running it there gave 4.76 t/s immediately**, with `majflt` 14,418 per 64 tokens - normal.

`stat -f` shows both paths on the same f2fs filesystem (same fsid), so this is not a different
device; it is the **physical layout of the `adb push` copy**. The expert tensors are read as
~460 KB random slices, and a copy written by `adb push` into a nearly-full f2fs volume (80% used)
is evidently fragmented enough that those reads are ~50x slower, even though a large sequential
read is unaffected. Re-eviction of the page cache and 9 GB of free memory did not help.

Consequence: **`/data/local/tmp` is not a safe place for a model on this device.** Always stage on
`/sdcard`, and validate a new model copy with a single decode arm before spending a block on it.

### 6.2 Decode, mirrored A/B/B/A (two blocks)

Block D1:

| arm | model | cap profile | decode t/s | minflt | majflt | profile stable |
|---|---|---|---|---|---|---|
| D1a1 | shipping | cpu0=3532800 cpu6=3840000 | 4.9698 | 117,012 | 16,114 | yes |
| D1b1 | q8dense | cpu0=3532800 cpu6=3840000 | **5.2829** | 117,931 | 16,759 | yes |
| D1b2 | q8dense | cpu0=2400000 cpu6=2841600 | 5.0789 | 108,352 | 14,717 | NO |
| D1a2 | shipping | cpu0=3532800 cpu6=3840000 | 5.0840 | 115,824 | 15,893 | yes |

Block D2 (the device was already hot, so every arm throttled to 2841600 during load/warmup):

| arm | model | cap start | cap end | decode t/s |
|---|---|---|---|---|
| D2a1 | shipping | 3840000 | 2841600 | 4.6440 |
| D2b1 | q8dense | 3840000 | 2841600 | 4.5940 |
| D2b2 | q8dense | 3840000 | 2841600 | 4.7757 |
| D2a2 | shipping | 2841600 | 2841600 | 4.4926 |

Same-profile pairs:

| block | pair | shipping | q8dense | delta |
|---|---|---|---|---|
| D1 | a1/b1 | 4.9698 | 5.2829 | **+6.30%** |
| D1 | b2/a2 | 5.0840 | 5.0789 | SKIPPED (D1b2 throttled mid-arm) |
| D2 | a1/b1 | 4.6440 | 4.5940 | **-1.08%** |
| D2 | b2/a2 | 4.4926 | 4.7757 | **+6.30%** |

Note on D2: the `profile stable` column flags a1/b1/b2 because their cap fell between the start-of-arm
and end-of-arm readings, but that drop happened during model load and the throwaway warmup, i.e.
*before* the measured window. All four D2 measured windows therefore ran at the same throttled
2841600 profile, and are comparable with each other. D1b2 is the one genuinely contaminated arm -
its cap fell during the measured window.

**Aggregate:** group means over the 8 arms give shipping 4.798 vs q8dense 4.933 = **+2.82%**;
per-block means +3.06% (D1) and +2.55% (D2). Three of the four pairs are positive, the best
fully-stable pair is +6.30%, and the single negative pair is -1.08%.

This is the same measurement regime that Stage 1 showed has a 5-8% arm-to-arm noise floor, so
**+2.8% is a direction, not a proven effect**. It is consistent with the mechanism: the rewrite
removes 638.3 MiB of the ~3.35 GB the dense path streams per token (19%), and the dense path is
roughly half of token time.

### 6.3 Prefill, mirrored A/B/B/A

First attempt (block P1, 966-token prompt): **unusable.** Every arm throttled mid-arm and the block
drifted monotonically downward (7.79, 7.81, 7.36, 6.71 t/s), so the two mirrored pairs read +0.31%
and +9.63%. A 966-token prefill takes ~130 s, making an arm ~3.5 min and the block ~16 min - far
beyond this device's stable thermal window.

Redesigned (block P2, 256-token prompt, arm ~1.7 min, block ~8 min) - **all four arms held one
stable cap profile**:

| arm | model | cap profile | prefill prompt t/s | minflt | majflt | profile stable |
|---|---|---|---|---|---|---|
| P2a1 | shipping | cpu0=3532800 cpu6=3840000 | 6.6399 | 571,447 | 262,951 | yes |
| P2b1 | q8dense | cpu0=3532800 cpu6=3840000 | 6.5768 | 586,232 | 408,044 | yes |
| P2b2 | q8dense | cpu0=3532800 cpu6=3840000 | 6.4451 | 603,013 | 348,695 | yes |
| P2a2 | shipping | cpu0=3532800 cpu6=3840000 | 6.2227 | 607,893 | 261,772 | yes |

| pair | shipping | q8dense | delta |
|---|---|---|---|
| a1/b1 | 6.6399 | 6.5768 | -0.95% |
| b2/a2 | 6.2227 | 6.4451 | +3.57% |

Group means: shipping 6.4313 vs q8dense 6.5110 = **+1.24%**. Residual downward drift across the
block is ~6% (6.64 -> 6.22), and the mirrored A/B/B/A placement balances it (A at positions 1 and 4,
B at 2 and 3), so the group mean is the better statistic here.

**Prefill conclusion: +1.2%, inside the noise floor.** The rewrite is a decode-side change: it removes
dense bytes, and prefill is less sensitive to that than decode.

## 7. Recommendation

* **Ship `varH`** (the 192 `hc_*_(up|down)` BF16 -> Q8_0 conversions): -562.5 MiB of dense bytes per
  token for +0.09% PPL (not statistically distinguishable from zero), and the only device speed
  number obtained on the *full* rewrite was **+6.30% decode** on the single cap-stable mirrored
  pair. The dense byte saving is 88% of the full rewrite, so varH should carry most of that speed -
  but **varH's own device number is not measured**, and it needs to be before shipping, since the
  `output.weight` Q5_K -> Q4_0 also swaps a slow K-quant unpack for a fast 32-block one and could be
  a disproportionate part of the speed gain.
* **Drop `output.weight Q5_K -> Q4_0`.** If `output.weight` is still wanted smaller, `Q5_0` or `Q8_0`
  are the candidates to test - `Q4_0` on the LM head is what breaks the gate.
* Keep the quantizer patch: it is what makes `COPY` + `--tensor-type` usable at all, and it is
  additive (non-matching tensors stay bit-exact, proven by the byte-identical shard 2).
* Next measurement steps, in order: (1) a second and third cap-stable decode block for the full
  rewrite (n=1 pair is not enough), (2) the same on varH, (3) a prefill design that fits the thermal
  window.

## 8. Reproducing

```
upstream-llama/build-x86/bin/llama-quantize --allow-requantize --keep-split \
  --tensor-type 'hc_.*_(up|down)\.weight$=q8_0' \
  <shard1> out.gguf COPY 16

upstream-llama/build-x86/bin/llama-perplexity -m <shard1> \
  -f .ppl-corpus/wikitext-2-raw/wiki.test.raw -c 512 --chunks 32 -t 16
```

Device: `MoeDroid/tools/mqdense_arm.sh <A|B> <tag> <decode|prefill>` and
`mqdense_run.sh <decode|prefill> <tag>`; both models must be on `/sdcard` (see 6.1).
Analysis: `MoeDroid/tools/mqdense_table.py <log>`.

Logs: `moe-dense-q8/{dryrun,quantize,varH-quant,varO-quant}.log`,
`moe-dense-q8/{ppl32-orig,ppl32-new,varH-ppl,varO-ppl}.log`, `moe-dense-q8/sha256.txt`,
`md-decode-full.txt`, `md-prefill-P1.txt`.
