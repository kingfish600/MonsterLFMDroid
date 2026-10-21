# Stage 5: `iqk_mul_mat` port plan

Design document only. **No kernel was modified in this stage.** It records what
`ik_llama.cpp` does differently from our fork on this device, why the measured oracle gap is
what it is, and what a port would have to change. All numbers are measured on the RedMagic 10S
Pro (Snapdragon 8 Elite, CPU only) against the same `varH` model unless marked otherwise.

---

## 0. Why this document exists: the measured oracle gap

`ik_llama.cpp` was cross-compiled for Android arm64 with `GGML_IQK_MUL_MAT=ON` (its default)
and **loaded our `varH` model unmodified** - no type alias, no enum shim. Both engines were run
on the same device, same model, greedy, 64-token decode and a 262-token prefill:

| engine | decode t/s | prefill t/s |
|---|---|---|
| our fork (`varH`, shipping flags incl. batch pinning) | **5.61** | **9.07** |
| our fork (`varH`, unpinned, comparable to ik's flag set) | 5.61 | 7.47 |
| **`ik_llama.cpp`** (`varH`, `-t 6 -tb 8 --cpu-mask 0x3f -ub 512`) | **8.89 - 9.48** | **12.00 - 12.61** |

So ik is **~+65% on decode** and **~+35 to +65% on prefill** on identical weights. That is the
headroom this port is chasing, and it is large enough that the port is clearly worth doing.

Caveats on that table, stated plainly:

* ik was loaded **without** `--cache-type-k/v q8_0` (f16 KV) because the first load omitted
  them; q8_0 KV is the cheaper configuration, so this *understates* ik. A matched run is the
  next step, not a conclusion.
* ik does **not** implement `--cpu-strict`, `--cpu-mask-batch` or `--poll`. It therefore cannot
  take the batch-pool pinning that buys our fork +21.5% prefill, and it still wins.
* ik's numbers come from its own `slot print_timing` lines (its `/completion` response omits
  the `timings` object our harness normally reads).

---

## 1. How `ik_llama.cpp` structures its ARM kernels

Two things matter, and neither is a single clever instruction.

### 1.1 It is templated over the weight type, not a `vec_dot` per type

Our fork follows mainline: one `ggml_vec_dot_<wtype>_<atype>` per pair, dispatched through
`type_traits[].vec_dot`, called once per row of `x` (`nrc == 1`). ik instead has a small set of
**templated multi-row kernels** instantiated per weight type:

```
mul_mat_qX_0_q8_0_T     (T = Dequantizer)      // q8_0 activations
mul_mat_qX_1_q8_2_T     (T = Dequantizer)      // q8_2 activations
mul_mat_q2_0_q8_0_x4 / mul_mat_q2_0_q8_2_x4    // q2_0, explicit x4
```

registered through `IQK_SET_MUL_MAT_FUNCTIONS[_T/_T2]`. The `_x4` suffix is the point: the
kernel consumes **four rows of `x` per call** and keeps `acc[nrc_y]` accumulators, so the
activation side is dequantized and scaled once for four weight rows instead of four times.

### 1.2 Its inner loop is two structural tricks, not a different algorithm

`mul_mat_q2_0_q8_0_x4` (`iqk_gemm_legacy_quants.cpp:4511`):

```c
const int8_t  k_shifts[16] = {0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6};
const uint8_t k_shuff[64]  = { 0,0,0,0, 1,1,1,1, ... 15,15,15,15 };   // 64 entries
const auto shuff = vld1q_u8_x4(k_shuff);                              // 4 registers at once
...
float dx1 = GGML_FP16_TO_FP32(q2[2*ib+0].d);
float dx2 = GGML_FP16_TO_FP32(q2[2*ib+1].d);
float32x4_t dx = {dx1, dx1, dx2, dx2};        // TWO block scales packed into one vector
auto bits = vld1q_u8(q2[2*ib+0].qs);
qx[0] = vqtbl1q_u8(bits, shuff.val[0]);
```

Two differences from our kernel fall straight out:

1. **`dx` packs two `q2_0` block scales into one `float32x4`.** One `vmlaq` therefore retires
   **128 weights** (2 blocks x 64) instead of 64. Our kernel does one
   `vcvtq_f32_s32 -> vmlaq_n_f32` per 32 weights.
2. **`vld1q_u8_x4` shuffles four registers per instruction**, expanding 4 blocks (256 weights)
   in one pass, where we issue one `vqtbl1q_u8` per 16 output bytes.

### 1.3 What our ARM `q2_0` kernel actually is (correcting the brief)

The brief describes `ggml_vec_dot_q2_0_q8_0` as a "14-op" scalar loop. It is **not scalar** -
our fork already has a hand-written NEON kernel at `ggml/src/ggml-cpu/arch/arm/quants.c:222`,
and it already carries a measured micro-optimisation:

```c
// Two accumulators. Measured on Snapdragon 8 Elite with q2_0_bench: the
// single-accumulator form is latency-bound on the vcvtq_f32_s32 ->
// vmlaq_n_f32 chain (one per 32 weights), and an independent second
// accumulator takes the kernel from 6.2 to 9.4 Mrow/s (1.52x).
// Set BONSAI_Q20_UNROLL=0 for the original single-accumulator kernel.
```

So the fork has already independently discovered the exact bottleneck ik attacks - the
per-32-weight `vcvtq_f32_s32 -> vmlaq_n_f32` dependency chain - and fixed it the cheap way
(a second accumulator, 1.52x). The real structural gap is `nrc`:

| | our fork | ik |
|---|---|---|
| `q2_0` `nrows` | **1** (`ggml-cpu.c:239`) | 4 (`_x4`) |
| `q2_0` `vec_dot` | `ggml_vec_dot_q2_0_q8_0`, `assert(nrc == 1)` (`arch/arm/quants.c:227`) | `mul_mat_q2_0_q8_0_x4` |
| activation format | `block_q8_0` | `block_q8_0` **or** `block_q8_2` |
| scale/accumulate | one `vmlaq` per 32 weights | one `vmlaq` per 128 weights (`dx` packs 2 blocks) |
| table expand | 2 x `vqtbl1q_u8` per 32 weights | 4-wide `vld1q_u8_x4` shuffle |

Per 32 weights our kernel issues roughly: 1 `vld1_u8`, 1 `vcombine_u8`, 2 `vqtbl1q_u8`,
2 `vshlq_u8`, 2 `vandq_u8`, 2 `vsubq_s8`, 2 `vld1q_s8`, 2 `vdotq_s32`, 1 `vcvtq_f32_s32`,
1 `vmlaq_n_f32` (~16 vector ops). ik does not use fewer ops per weight for the *extract*; it
uses **far fewer accumulate/convert ops per weight** and **amortises the activation side over
4 rows**.

Note also that `q4_0`/`q8_0` get `nrows = 2` under `__ARM_FEATURE_MATMUL_INT8` in our fork,
but `q2_0` is pinned at `nrows = 1` - the i8mm path was never extended to it.

---

## 2. 2-bit and 3-bit: which functions handle what

| type | GEMV (`n=1`, decode) | GEMM (`n>1`, prefill) |
|---|---|---|
| `q2_0` | `mul_mat_q2_0_q8_0_x4` / `mul_mat_q2_0_q8_2_x4` | same kernels, `nrc_y` grows with the batch |
| `pq2_0` (Prism ternary, 34-byte block, `QK=128`) | the `qX_1_q8_2` template family | same |
| `q3_K` | the `_kquants` template family (`iqk_gemm_kquants.cpp`) | same |

There is no separate GEMV and GEMM path: the same `_x4` kernel serves `n=1` and `n>1`, and the
batch dimension simply increases `nrc_y` (how many `y` rows are packed into `Q8<nrc_y, ...>`).
That is the single most important structural difference from our fork, where
`mul_mat_id`/`mul_mat` fall back to a `nrc == 1` `vec_dot` loop over rows.

The activation side is pre-packed (`block_q8_0_x4`, `block_q8_2`) so the kernel receives `y`
already in the layout it wants, rather than re-quantising and re-laying-out per weight row.

---

## 3. Why `nrc = 2/4` is worth only a few ms on `n=1` decode

This is the crux, and it is worth being precise because it sets the expectation for the port.

On a **single-token decode** (`n = 1`), `nrc_y = 1`. There is exactly one activation row. So
the `_x4` structure buys **nothing** on the `y` side - there is no second row of `y` to share
the load with. What remains is:

* the `dx` packing (one `vmlaq` per 128 weights instead of per 32) - a real win, but bounded by
  the fact that the accumulate chain is already partially hidden by our dual accumulator;
* the 4-wide table shuffle - fewer `vqtbl1q_u8` issues per block.

Measured against our kernel that is worth a few ms/token, not a step change. Our own
`BONSAI_Q20_UNROLL` experiment already captured most of the available accumulate-chain win
(6.2 -> 9.4 Mrow/s = 1.52x on the isolated kernel), and the end-to-end gain from that was
small because the expert path is **not** purely ALU-bound: Stage 4 measured the decode expert
path at 79.5 ms/token for 632.8 MiB (7.8 GiB/s) with **31% of its excess over the dense rate
attributable to page-fault stalls**, leaving ~43 ms/token in the Q2_0 compute path itself.

So the honest expectation for a `nrc=2/4` port on decode is **single-digit percent**, and it
must be validated on-device rather than assumed from kernel-level Mrow/s.

Where `nrc` genuinely pays is **prefill** (`n > 1`), where there are many activation rows to
amortise the weight-side dequant and scale conversion over. That is consistent with the
measured oracle split: ik's advantage is **larger on prefill (+35 to +65%) than on decode
(+65% but from a different mechanism** - see §4).

---

## 4. What ik does to make `n=1` 2-bit decode faster

Given §3, the decode gap cannot be explained by `nrc` alone. The remaining candidates, in the
order they should be investigated:

1. **`GGML_EXPERT_CHUNKING`** is a compile-time define in ik (`ggml/src/ggml.c:18704`,
   `:19026`) that we do not have. Our `BONSAI_MULMATID_FLAT` addresses the same problem from
   the other end: with one token the stock path walks experts in order and every thread steals
   16-row chunks inside the *same* expert, so all threads miss the same TLB entries; flattening
   to contiguous half-expert units claimed from one counter lets each thread stream a different
   expert. ik's chunking may do this better, or may interact with the kernel selection.
2. **Activation format `q8_2`.** ik has a second activation type alongside `q8_0`. A 2-bit
   weight x 8-bit activation dot only needs ~2 bits of activation dynamic range per block; a
   narrower activation block cuts the `y` traffic and the convert work. This is a *different
   algorithm* on the activation side, not a tiling change, and is the most likely single source
   of a large decode win.
3. **Kernel selection at runtime.** `GGML_IQK_MUL_MAT` gates a dispatch that picks among
   several kernel families; our fork has one kernel per type.
4. **Flash attention.** ik defaults to flash attention enabled and has its own `iqk/fa`
   implementation; our fork runs with `-fa on` too, but the implementations differ. Part of the
   decode difference may sit outside `mul_mat` entirely.

**A port should therefore not start with `nrc`.** It should start by profiling ik's decode
graph against ours on the same device with the same `BONSAI_PROF` instrumentation, to find
where the 9.4 vs 5.6 t/s actually goes. Writing `nrc=2/4` first would optimise the part of the
budget that Stage 4 already showed is only ~43 ms/token of compute.

---

## 5. `mul_mat_id` and token grouping during `n>1` prefill

Our fork: `mul_mat_id` has exactly one `ggml_barrier` before the per-expert loop and uses a
per-expert atomic counter (handoff §5). `BONSAI_MULMATID_FLAT` flattens the work into
contiguous half-expert units claimed from a single counter, so each thread streams a different
expert.

ik: the equivalent work distribution lives behind `GGML_EXPERT_CHUNKING`, and the compute side
goes through the same `_x4` kernels with `nrc_y` set from the batch. During a long prompt the
router touches ~all 512 experts per layer, so the prefill expert path reads essentially the
whole 32,400 MiB pool once per prompt (Stage 4: 70.5 ms/token, 61.9% of prefill). A port should
compare **how tokens are grouped per expert** before touching the kernel: grouping tokens that
share an expert into one `nrc_y` block is what makes the `_x4` kernel useful in prefill, and it
is a scheduling change, not a numerical one.

---

## 6. Recommended port order (when kernels are eventually touched)

1. **Profile first.** Run both engines on `varH` with `BONSAI_PROF_EVERY=1` and diff the
   per-op budget. Do not write a kernel until the 9.4 vs 5.6 t/s gap is attributed.
2. **Matched-flag oracle re-run** (cheap, do before any code): ik with
   `-ctk q8_0 -ctv q8_0` against our engine with the same, both unpinned, mirrored A/B/B/A.
   The current 8.89-9.48 t/s was measured with f16 KV and is a lower bound.
3. **Activation-side work first** (`q8_2`-style narrower activation blocks, or `nrc=4`
   activation packing) because §3 says that is where the structural reuse is.
4. **`nrows = 2` for `q2_0` under `__ARM_FEATURE_MATMUL_INT8`**, matching what `q4_0`/`q8_0`
   already get in our own table - the smallest, best-understood step.
5. Only then the `dx`-packing / 4-wide-shuffle micro-optimisations.

Correctness gates for any of these, per the project's rules: kernel changes must be
**token-identical** (`--temp 0` md5), and speed must be judged from mirrored A/B/B/A pairs with
the cap gate held, never from a single arm.

---

## 7. SHA-256 enforcement in `ModelDownloader.kt` (documented, deliberately NOT enabled)

`ModelAsset.sha256` is currently **documentation only** - `grep` finds no consumer anywhere
outside `ModelCatalog.kt`, and `isOneDownloaded()` short-circuits to a plain
`fileFor(one).exists()` whenever a `.meta.json` sidecar is present.

The change to enforce it would be small:

```kotlin
// ModelDownloader.kt, in the download finalize path, after dest.length() is known
val expected = asset.sha256
if (expected.isNotEmpty()) {
    val actual = MessageDigest.getInstance("SHA-256").apply {
        dest.inputStream().use { it.copyTo(ChunkedSink(this)) }   // stream, never read 37 GB into RAM
    }.digest().joinToString("") { "%02x".format(it) }
    if (!actual.equals(expected, ignoreCase = true)) {
        dest.delete()                       // do not keep a bad shard
        File(dest.absolutePath + ".part").delete()
        throw IOException("sha256 mismatch for ${asset.fileName}")
    }
}
```

plus a matching check in `isOneDownloaded()` so an existing file is re-verified (or at least
size-checked) rather than trusted on existence alone.

**Do not enable strict failure in the shipping branch yet.** Shard 1 on the device is now the
`varH` rewrite (`e7c69452...ded12e2`, 37,033,916,192 B) while the hosted HuggingFace URL still
serves the un-rewritten shard (`69820c02...b199b720`, 37,623,740,192 B). Turning on enforcement
today would make every fresh install **reject the hosted file and fail to provision**. The
order must be: publish the rewritten shard (or repoint the asset URL) -> then enable
enforcement -> then re-provision a device from scratch to prove it.

Note this is also why the current swap is transparent: because the hash is never checked, a
device that already holds `varH` runs it regardless of what the catalog declares.
