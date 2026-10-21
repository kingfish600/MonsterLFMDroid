# Stage 7 daytime prep (host-only) + Stage 6 wrap-up

Date: 2026-10-17, 04:45-05:30 CDT (Part A, phone connected) then host-only.

Part A: install the Stage 6 engine on the phone and run one diagnostic. Part B: host-only work.

---

## PART A: app upgrade and the `5.39 vs 6.90` diagnostic

### A.1 The updated app is installed

| step | result |
|---|---|
| build | `tools/gradle-8.14/bin/gradle assembleDebug -g tools/gradle-home` in `MoeDroid` -> **BUILD SUCCESSFUL in 8s**, `mergeDebugNativeLibs` + `packageDebug` executed |
| APK | `MoeDroid/app/build/outputs/apk/debug/app-debug.apk`, 126,665,764 B, sha256 `7fd0a8340a96ec57...` |
| kernel actually in the APK | `libggml-cpu.so` **inside** the APK hashes to `b57412be3865b21...`, identical to the Stage 6 build - so the shipped kernels are really packaged, not assumed |
| install | `adb install -r` -> **Success** (in-place upgrade, never uninstall) |
| model preserved | all three files unchanged after the install: shard 1 `37033916192`, shard 2 `28800138432`, MTP `2786204800` |
| installed version | `versionCode=11`, `versionName=0.2.0-pinned`, `lastUpdateTime=2026-10-17 04:48:25` |
| installed native lib | the extracted `/data/app/.../lib/arm64/libggml-cpu.so` is **5,631,856 B**, exactly the Stage 6 kernel build (the previous one was 5,535,760 B) |
| app state after | launches and runs (217 MB RSS); its engine process starts only when a chat is opened, which is expected |
| clean copy for the user | `apk-out/MoeDroid-0.2.0-pinned-stage6-arm64.apk` |

Two operational facts worth recording, both cost a failed attempt on the night:

* **The Gradle wrapper does not work in this sandbox.** `./gradlew` tries to write
  `~/.gradle/wrapper/dists/...lc k` and the home directory is read-only; it fails with
  `FileNotFoundException ... (Read-only file system)`. Use the installed binary
  `tools/gradle-8.14/bin/gradle` with `-g tools/gradle-home` (inside the workspace) instead.
* **`nohup cmd &` dies here.** Each shell runs in a PID namespace with `--die-with-parent`, so any
  child of a foreground command is killed when that command returns (the first build attempt produced
  a 0-byte log and never ran). Long work must be a managed background job, not `nohup`.

### A.2 The `5.39` vs Stage 5's `6.90` question: first evidence

`filefrag` **does not exist on this device** (toybox has no such applet), so the extent count could
not be read directly. The block-allocation proxy instead:

| file | apparent size | allocated (`stat -c %b` x 512) | overhead |
|---|---|---|---|
| shard 1 | 37,033,916,192 | 37,070,336,000 | **+34.7 MiB (0.094%)** |
| shard 2 | 28,800,138,432 | 28,828,463,104 | **+27.0 MiB (0.094%)** |

Both files carry only ~0.1% allocation overhead, and identical overhead, so nothing here looks
pathological. That is consistent with f2fs preallocation rather than heavy fragmentation, but it is a
**proxy, not an extent count** - a file can be allocated exactly and still be badly fragmented. The
decisive test is therefore the one below, which does not depend on file layout at all.

### A.3 RESULT: the `5.39` vs `6.90` gap is a WARMUP effect, not fragmentation

One decode arm on the shipped engine, same file, same flags, same cap profile, but **14 warmup
requests instead of 4** (the Stage 6 protocol used 4):

| protocol | median | per-request series |
|---|---|---|
| Stage 6 shootout (4 warmups + 6 measured) | **5.3869** | 4.43 5.70 5.40 5.59 5.37 5.37 |
| **deep warmup (14 + 6)** | **6.5245** (wall 6.3210) | 5.56 6.51 **7.06** 6.51 6.54 6.84 |

That is **+21.1%** from warmup depth alone, and the deep-warmup series reaches 7.06 t/s - i.e. **Stage
5's 6.90 band**. So:

* **The app-path model copy is NOT slow.** It converges to the same steady state the Stage 5 copy
  reached. The 92%-full f2fs staging and the `cp`+`mv` at Stage 4 did not leave it crippled
  (consistent with the ~0.1% allocation overhead in A.2).
* **The Stage 6 protocol was warmup-limited by ~21%.** The ~12-15 GB expert working set needs roughly
  ten warmup requests to settle, not four: the first measured request of the deep arm still read 5.56
  and only the third onwards was above 6.5.
* **What this means for Stage 6's numbers:** its *relative* deltas remain internally valid (every arm
  shared the same 4-warmup protocol, and pairs are within-pass), but they were measured in a
  cold-biased regime where page-cache misses compete with the kernels for the bottleneck. That is a
  plausible reason `A1` showed only +2.7-6.3% and `A3` showed -5.7%: a kernel change is least visible
  when the measurement is dominated by cold-cache stalls. **Any Stage 7 shootout must use >= 10-14
  warmups**, and Stage 6's absolutes should be read as ~21% low.
* The shipped app, in real use after its cache warms, should sit around **6.5 t/s** rather than the
  5.5 reported in `STAGE6-report.md` §2.



---

## PART B: host-only work

Per the Stage 7 priority override, in this order:

| # | item | decision |
|---|---|---|
| 1 | `BONSAI_MOE_X4=2` x86 replica + 32-chunk PPL (Gate 3b) | **DEFERRED to 0 minutes** - mode 2 is only +1.26% over the already-shipped mode 1, inside the noise floor, while the real gap is +23.6% prefill / +40.7% decode. Stays opt-in behind the env switch |
| 2 | `q3_K` four-row revival | **SKIPPED** - measured -5.72%/-5.74%; it is register pressure on an already latency-bound integer chain |
| 3 | **extend the bit-exact `q2_0 x4` kernel to PREFILL** (`ne11 > 1`, `nrc_y = 1..4`) | **DO FIRST** - `MUL_MAT_ID` is 61.9% of prefill and currently falls back to the 1-row `vec_dot` there |
| 4 | port `block_q8_0_x4` / `quantize_row_q8_0_x4` activation repacking | **DO SECOND** |
| 5 | `s7_shootout.sh`: warmup sweep + clean `B1`/`B2` without `--poll` | **DO THIRD** |



### B.3 (priority #3) The q2_0 prefill tile: kernel implemented and bit-exact, wiring deliberately not done

**First, a correction to the task's premise, because it changes the design.** The task says
"during prefill (`ne11 > 1`) our `iqk_x4.c` is completely disabled". It is not: the expert hook
(`ggml-cpu.c:1638`) has **no `ne11` guard** (only the dense hook at `:1257` has the
`num_rows_per_vec_dot == 1` one, which is always true for `q2_0`). So during prefill the x4 kernel
already runs - it just runs **4 weight rows x ONE token** per call, because the token loop
(`ir1`) is *outer* and therefore re-unpacks the same four weight blocks for every token.

What is actually missing is what ik does: **amortising the weight unpack across token columns**
(ik loops the weight row outside and keeps `acc[nrc_y]` for the tokens). Per 128 weights the unpack
is 16 vector ops plus 4 `vqtbl1q_u8` - and that repeated unpack, not the bytes, is the prefill cost.

**Implemented** (`arch/arm/iqk_x4.c`, `BONSAI_MOE_X4_PREFILL`, default 0):
`bonsai_vec_dot_q2_0_q8_0_x4t4(n, s, vx, bx, vy, by)` - 4 weight rows x 4 token columns,
row-outer with 8 live accumulators (`e0-e3`, `o0-o3`), writing `s[ix*4 + iy]`. The accumulator
layout is ik's shape precisely to avoid the `A3` failure mode.

**Verification under `qemu-aarch64-static`** (`MoeDroid/tools/s7_q2_0_x4t_test.c`, 16 reference
calls per size):

| nb | n | result |
|---|---|---|
| 1, 2, 3, 5, 8, 16 | 64 .. 1024 | **BIT-EXACT, `max|d| = 0.000e+00`, all 16 (row, token) pairs** |

`ALL CHECKS PASSED`, exit 0. Odd `nb` values exercise the single-block tail path, and the first
compile attempt failed usefully: I had written the 32-weight group as a horizontal-add scalar, but
the `q2_0` reference keeps the group as a **4-lane vector** and does one `vmlaq_n_f32` per group -
reproducing that order is exactly what makes the kernel bit-exact.

**Spill analysis (`clang -S`, counting `str`/`ldr` to `[sp]`)** - the check the `A3` episode demands:

| kernel | frame | stack stores | stack loads | verdict |
|---|---|---|---|---|
| **`x4t4` (new prefill)** | **176 B** | **4** | **5** | 9 folded spills, spread through the body (11%-79%) |
| `x4` (shipped decode kernel) | 448 B | 22 | 20 | 42, yet it measured **+4.5%**: spills are not automatically fatal |
| `q3_K x4` (rejected) | 800 B | 20 | 17 | 37, and it measured **-5.7%** |

So the new kernel is the **leanest of the three** and its frame is 4.5x smaller than the rejected
one. But it does **not** meet the "zero `sp` spills in the inner loop" bar: it holds 8 unpacked
weight registers (`a0-a3`, `b0-b3`) live across 8 accumulators, which exceeds what 32 vector
registers can hold alongside the activation loads. The fix is structural and identified: **unpack
one block at a time** (block `i` -> its four tokens -> block `i+1` -> its four tokens), which halves
the live `qx` registers to 4 and should reach zero spills; a 2-token tile would also do it, at the
cost of the amortisation factor falling from 4x to 2x. Neither was attempted here, because a kernel
that is bit-exact but not yet spill-clean should not be wired into the graph on the same day it was
written.

**Not done, and why it is the correct stopping point:** wiring this into the call graph is *not* a
matter of adding a call. Both `..._one_chunk` functions loop tokens **outside** rows
(`for ir1 { for ir0 { vec_dot(...) } }`), so reaching a 4-token tile requires restructuring that
nesting (and the `dst_col` write pattern, which is per-token). That is a graph-level change in two
places, and it should be done together with the spill fix above, then gated on a device arm -
exactly the sequence that produced tonight's two shipped wins.

**Priority #4 (`block_q8_0_x4` repack) and #5 (`s7_shootout.sh`) were not started.** Both remain as
specified in the override: the repack needs `GGML_TYPE_Q8_0_X4` + `quantize_row_q8_0_x4` + the
enlarged work-buffer contract, and #5's warmup sweep is now informed by the A.3 finding (>= 10-14
warmups, not 4).

### B.4 (priority #3, continued) The spill question, settled: 2-token tile is clean, 4-token is not

**My stated fix from yesterday did not work, and that is the first thing to record.** The plan was
"unpack one block at a time so only 4 `qx` registers live instead of 8, which should reach zero
spills". Implemented and re-measured:

| version | frame | stack stores | stack loads | spills inside the hot loop |
|---|---|---|---|---|
| 4-token, unpack both blocks up front (yesterday) | 176 B | 4 | 5 | 8 of 9 at 11%-79% of the body |
| 4-token, unpack one block at a time (the "fix") | 272 B | 4 | 9 | **11 of 13** (loop body spans ~7%-66%) |
| **2-token tile (new)** | **0 B** | **0** | **0** | **none - no stack frame at all** |
| shipped decode `x4` (for reference) | 448 B | 22 | 20 | wins +4.5% anyway |
| rejected `q3_K x4` | 800 B | 20 | 17 | loses -5.7% |

So the hypothesis was wrong: the live-range pressure in the 4-token tile is not the two unpacked
blocks, it is **eight accumulators plus the activation loads**, and clang hoists S7-group loads across
the unrolled body regardless of how the unpack is ordered. The 4-token tile cannot be made spill-free
at `-O2`; the **2-token tile has no stack frame at all**.

Both are still bit-exact, re-verified after every change under `qemu-aarch64-static`:

| kernel | pairs checked | result |
|---|---|---|
| `x4t4` (4 tokens) | 16 per size, nb in {1,2,3,5,8,16} | **BIT-EXACT, `0.000e+00`** |
| **`x4t2` (2 tokens)** | 8 per size, nb in {1,2,3,5,8,16} | **BIT-EXACT, `0.000e+00`** |

**Decision: `x4t2` is the variant to wire**, not `x4t4`. It gives half the unpack amortisation of the
4-token tile (2x instead of 4x) but costs zero spills and zero frame - and after `A3` (a bit-exact
kernel that lost 5.7% to register pressure) a spill-free 2x is worth more than a spilling 4x.

**Why it is still not wired today - a hazard found while planning the loop change.** Wiring the tile
is not a local edit, and in the expert path there is a correctness trap:

* The kernel takes a single `by` stride between its activation rows. In `mul_mat_id_one_chunk` the
  activation pointer is built per token as `wdata + (i11 + i12*ne11)*row_size`, where `i12` comes from
  the router's `matrix_rows` mapping. **Consecutive expert-row indices are not guaranteed to be
  consecutive batch rows**, so two adjacent tokens' activation pointers are not necessarily one fixed
  stride apart. The kernel API would silently read the wrong rows if they are not.
* A safe wiring must therefore check `ptr(t+1) - ptr(t) == stride` and fall back to per-token calls
  when it fails, on top of restructuring the token-outer/row-inner nesting and the per-token `dst_col`
  scatter (dst positions are permuted by the same mapping).
* The dense path (`mul_mat_one_chunk`) does have a well-defined `src1_col_stride`, but dense `q2_0` is
  only ~3.6 ms/token, so wiring there alone would be busywork.

That is a device-verified change, not a blind one: get it wrong and the output is quietly wrong, which
is precisely the class of bug the token-identity gate exists to catch. It stays behind
`BONSAI_MOE_X4_PREFILL` (default 0, and currently not called from anywhere) until a device arm can
prove it.

**Next session, in order:** (1) wire `x4t2` into `mul_mat_id_one_chunk` with the stride guard above,
gated off; (2) device token-identity + speed arm; (3) if clean, the `block_q8_0_x4` repack (priority
#4); (4) the `s7_shootout.sh` warmup sweep (priority #5).

---

### B.5 (priority #3, finished) `x4t2` wired into both `_one_chunk` paths, and it is bit-identical on a real multi-threaded graph

Host-only session, phone unplugged: **nothing below was measured on the device**, and no `adb`
command was run. Everything is static analysis plus execution of the real ARM64 code under
`qemu-aarch64-static` (`.stage6/qemu/ext/usr/bin/qemu-aarch64-static`, §0 of the Stage 6 doc).

**1. The signature is now two activation pointers and two result pointers.** The stride-API version
of `bonsai_vec_dot_q2_0_q8_0_x4t2` could not be wired safely, because in
`ggml_compute_forward_mul_mat_id_one_chunk` the row for a routed token is
`wdata + (i11 + i12*ne11)*row_size` with `i12` taken from the router's `matrix_rows` map. Two
adjacent `ir1` values are two arbitrary rows, not two rows one stride apart, and `dst_col` is
permuted by the same map. So the kernel now takes what the caller actually has:

```c
int bonsai_vec_dot_q2_0_q8_0_x4t2(int n, float * s0, float * s1,
                                  const void * vx, size_t bx,
                                  const void * vy0, const void * vy1);
/* s0[ix] = dot(weight row ix, vy0), s1[ix] = dot(weight row ix, vy1), ix = 0..3 */
```

**2. The signature change cost nothing in register pressure.** Same measurement method for every
row below (`clang -S` with the shipping flags, `-march=armv8.2-a+dotprod+i8mm+fp16+bf16 -O3`;
`.stage7/spill.py`): frame = the largest `sub sp, sp, #N`, and **local spills** = every `[sp]`
access whose register list is not a callee-saved save (`x19-x30`/`d8-d15`) - callee-save traffic
is reported separately because a frame made only of it is not a spill.

| kernel | frame | callee-save | local stores | local loads |
|---|---|---|---|---|
| `bonsai_vec_dot_q2_0_q8_0_x4t2` (new two-pointer API) | **0 B** | 8 | **0** | **0** |
| `bonsai_x4_rows2_mode3` (new, Task 2 §B.6) | **0 B** | 10 | **0** | **0** |
| `bonsai_x4_rows4_mode1` (shipped decode default, now its own function) | 432 B | 12 | 28 | 27 |
| `bonsai_x4_rows4_mode2` (packed, opt-in) | 400 B | 12 | 16 | 17 |
| `bonsai_vec_dot_q2_0_q8_0_x4t4` (4-token tile, not wired) | 240 B | 12 | 7 | 12 |
| `bonsai_vec_dot_q3_K_q8_K_x4` (rejected on speed) | 784 B | 20 | 28 | 29 |

Two notes on method, because the counts differ from §B.3/§B.4 and the difference is not noise:

* §B.3/§B.4 counted raw `str/ldr [sp]` lines, which includes the callee-save prologue. Re-measuring
  the preserved `.stage6/asm/iqk_x4.s` with the split above gives **448 B / 24 stores / 23 loads**
  for the old `x4`, so the method reproduces the old numbers once callee-saves are excluded.
* The `x4` function is now a three-way dispatcher (`mode 1` / `mode 3` / `mode 2`) into three
  `noinline` helpers, because sharing one body let clang allocate for the worst mode in all of
  them. The dispatcher itself is 0 B / 0 spills; mode 1's own body spills 28/27 rather than the 44
  the shared function showed.

**3. Arithmetic verification (`qemu-aarch64-static`, real NEON).** `MoeDroid/tools/s7_q2_0_x4t_test.c`
updated for the new signature, 8 (row, token) pairs per size, `nb in {1,2,3,5,8,16}`:

| kernel | result |
|---|---|
| `x4t2`, two-pointer API | **BIT-EXACT, `max|d| = 0.000e+00`, all 48 pairs** (`rc=1`, 8 per size) |
| `x4t4` (unchanged, re-run as a control) | **BIT-EXACT, `0.000e+00`**, 96 pairs |

**4. Wiring, gated by `BONSAI_MOE_X4_PREFILL` (default `0`).** Both
`ggml_compute_forward_mul_mat_id_one_chunk` and `ggml_compute_forward_mul_mat_one_chunk` gained a
paired-token pre-loop that runs only when `bonsai_q2_0_x4_prefill_mode() == 1`, `src0->type ==
GGML_TYPE_Q2_0`, and a second token is inside both the current `blck_1` window and `ir1_end`.
Structure, identical in both places:

* the outer token loop advances by 1 in the stock path, so the paired loop consumes token pairs
  and **at most one token is left for the stock loop**, which is untouched and runs with
  `ir1` where the paired loop stopped. One-token prefills therefore behave exactly as before.
* `ir0` still walks in 4-row groups (`ir0 + 4 <= MIN(iir0 + blck_0, ir0_end)`) into two 16-float
  scratch arrays; the remaining 1-3 rows of a 16-row block, and the case where the expert has
  fewer than 4 rows, fall through to two 1-row `vec_dot` calls. In the model under test `ne01`
  (`n_ff`) is a multiple of 4, so that tail only appears on the last chunk.
* `src0_row` for the dense path is now computed per token (`a02/a03` and `b02/b03`), and the tile
  is only used when both tokens resolve to the same weight slice; otherwise that pair takes the
  1-row path. This is the dense broadcast case `r2`/`r3` can produce and the stock code handles
  implicitly by recomputing per `ir1`.
* `BONSAI_MOE_X4` (the decode kernel mode) is untouched by this switch: the two are independent.

**5. The gate: byte-identical on a real multi-threaded `MUL_MAT_ID` graph.**
`MoeDroid/tools/s7_mmid_prefill_verify.sh` cross-compiles `MoeDroid/tools/s6_mmid_prefill_test.c`
as a **static Android aarch64 binary linked against the `build-arm64-ship` object files** and runs
it under `qemu-aarch64-static`. (Static linking needs five `dl*` symbols that a static Android
binary cannot resolve; `MoeDroid/tools/s7_qemu_dl_stubs.c` provides failing stubs for code the tests
never reach - the dynamic backend loader and the abort-path backtrace printer.)

| case | shape | x4t2 calls (`=0` / `=1`) | output |
|---|---|---|---|
| 0 | 256x512, 512 experts, 10 used, 16 tokens, 8 threads | 0 / **3200** | **byte-identical** |
| 1 | 256x512, 256 experts, 4 used, 16 tokens, 8 threads | 0 / **768** | **byte-identical** |
| 2 | 128x256, 512 experts, 8 used, 32 tokens, 8 threads | 0 / **2560** | **byte-identical** |
| 3 | 256x512, 512 experts, 10 used, 16 tokens, **1 thread** | 0 / **2176** | **byte-identical** |
| 4 | 256x512, 512 experts, 2 used, 4 tokens, 8 threads | 0 / 0 | **byte-identical** |
| 5 | 256x512, 512 experts, 10 used, **1 token**, 6 threads | 0 / 0 | **byte-identical** |

with the stability controls also passing (stock == stock, prefill == prefill on a repeat run) and
`BONSAI_MOE_X4=3` producing the same bytes as `=1`.

**The comparison is not vacuous, and that is now proven rather than assumed.** The first version of
this gate could have passed while never running the new path, so `ggml-cpu.c` now counts tile calls
(`bonsai_q2_0_x4_prefill_t2_calls()`) and the test prints the per-case delta. That immediately
changed the reading of the table:

* cases 0-3 genuinely exercise the tile (768-3200 calls each), including the single-thread case;
* **case 4 exercises it zero times**, and that is correct rather than a defect: it is the "fewer
  units than threads" shape where no expert receives two rows, so every chunk takes the stock tail.
  It is kept as the tail-path control instead of being counted as coverage;
* **case 5 is the single-token shape** and must stay on the decode path - asserted, not assumed.

**6. Incidental fix found while editing the file.** `bonsai_vec_dot_q2_0_q8_0_x4t4` and
`bonsai_q2_0_x4_prefill_mode` were sitting **outside** the `#if defined(__aarch64__) &&
defined(__ARM_NEON)` block in `iqk_x4.c` (the `#endif` was above them, and the file had no closing
`#endif` at all). They call NEON intrinsics unconditionally, so the x86 host build would have
failed to compile the file. The guard now encloses them. It was never observed because
`build-x86` was last rebuilt before `x4t4` was added.

**7. Not done here, and why.** No device arm was run (phone away), so `BONSAI_MOE_X4_PREFILL`
stays **off by default** and its speed is unmeasured. §B.7's harness carries it as arm `AP2`.

---

### B.6 (Task 2) The 42-spill decode kernel is now spill-free (`BONSAI_MOE_X4=3`), and the aligned activation repack works (`BONSAI_Q8_0_X4=1`)

Host-only; no device arm was possible (phone away). All verification is `qemu-aarch64-static`
executing the shipping ARM64 code.

#### B.6.1 Why `x4` spills, established by experiment rather than by reading the list

The starting point (§B.3): the shipped decode kernel has a 432 B frame and 55 local stack
accesses. I tested four hypotheses in order, which is the cheapest way to find the real cause:

| attempt | frame | local st/ld | verdict |
|---|---|---|---|
| 1. isolate mode 1 and mode 2 into their own functions (the shared body was allocating for the worst mode) | 432 B | 28 / 27 | real but not sufficient: the mode-1 body alone still spills |
| 2. four rows, but apply the two 32-weight halves one after the other | 128 B | 2 / 3 | better; the two remaining spills are two *constant* vectors evicted from the preheader |
| 3. as 2, plus a leaner dequant so the constant set drops from 7 registers to 3 | 128 B | 2 / 3 | no change: clang re-hoists the constants and spills two of them anyway |
| 4. unpack and apply each half per row instead of sharing one expansion across four rows | 288 B | 15 / 16 | **worse, and it was also wrong** - sharing one expansion across rows means row 1..3 read row 0's weights. The test caught it immediately |
| **5. two rows per pass, lean dequant, expansion still once per row** | **0 B** | **0 / 0** | **spill-free** |

Two changes did the work, and both are worth recording because they generalise:

**(a) The constant set went from seven registers to three.** The old dequant held a 64-byte
`vqtbl1q_u8` table as four registers plus `shift`, `mask` and `one`. Two observations collapse that:

* group `k` is the four bytes at `qs + 4k` repeated four times, so **one** 16-byte index vector plus a
  `vextq_u8` 4-byte rotation of the source produces every group - no 64-byte table;
* `(bits >> 2f) & 3` needs a mask register, but `((bits << (6 - 2f)) >> 6)` keeps exactly the same
  field with **no mask**. The `-1` bias stays, and unsigned wrapping makes it work on `int8`:
  `vsubq_u8(x, 1)` maps 0 to 255, which is `-1` read as `int8`, and 1/2/3 to 0/1/2 - the same values
  the signed subtract produced.

So the constants are now `idx4`, `shlft_left`, `one`: three registers instead of seven, and the op
count per group is unchanged (`tbl`, `shl`, `shr`, `sub` versus `tbl`, `shl`, `and`, `sub`).

**(b) The accumulator count is what actually binds.** Mode 1 holds eight accumulators (two per row
for the even/odd block alternation the reference uses), and eight accumulators + four expanded
weight registers + four activation registers + six scale floats is 22-24 live values before
temporaries, against 32 usable vector registers. Halving the rows per pass halves that to four
accumulators; the 2-bit expansion still runs **once per row**, so nothing is recomputed, and the
only cost is that the activation blocks for a block index are loaded twice instead of once
(8 loads per 128 weights instead of 4, all L1-resident).

Result (`clang -S`, shipping flags, `.stage7/spill.py`):

| kernel | frame | callee-save | local stores | local loads |
|---|---|---|---|---|
| `bonsai_x4_rows2_plain` = **`BONSAI_MOE_X4=3`** | **0 B** | 10 | **0** | **0** |
| `bonsai_x4_rows2_packed` = `BONSAI_Q8_0_X4=1` | **0 B** | 8 | **0** | **0** |
| `bonsai_vec_dot_q2_0_q8_0_x4t2` (prefill tile, §B.5) | **0 B** | 8 | **0** | **0** |
| `bonsai_x4_rows4_mode1` (shipped decode default) | 432 B | 12 | 28 | 27 |
| `bonsai_x4_rows4_mode2` (packed, opt-in) | 400 B | 12 | 16 | 17 |
| `bonsai_vec_dot_q3_K_q8_K_x4` (rejected on speed) | 784 B | 20 | 28 | 29 |

**Arithmetic verification of mode 3** (`qemu-aarch64-static`, real NEON, `nb in {1,2,3,5,8,16,32,64}`,
4 rows per size):

| kernel | result |
|---|---|
| `BONSAI_MOE_X4=3` | **BIT-EXACT, `max|d| = 0.000e+00`, all 32 row results** |
| `BONSAI_MOE_X4=1` (control) | BIT-EXACT, `0.000e+00` |
| `BONSAI_MOE_X4=2` (control, near-exact by construction) | `<= 7.63e-05` |
| reference vs the double oracle | `<= 4.30e-05` |

**What this is and is not.** Zero spills is a register-allocation property, not a performance claim.
`A3` (`BONSAI_Q3K_X4`) was bit-exact too and lost 5.7%, so mode 3 is **opt-in** (`BONSAI_MOE_X4=3`)
and §B.7 carries it as arm `A5`. The expectation is that it recovers what mode 1 was paying for 55
stack accesses per 4-row call; whether that is positive on this device is exactly what `A5` decides.

Two structural notes on the code, since they change the shipped path's shape without changing its
arithmetic:

* `bonsai_vec_dot_q2_0_q8_0_x4` is now a three-way `noinline` dispatcher, so the mode-1 and mode-2
  bodies no longer share one register allocation. The dispatcher itself is 0 B / 0 spills.
* Mode 1's own body is byte-for-byte the same arithmetic and is still the default, so `A0` in the
  shootout measures what the app ships today.

#### B.6.2 The aligned activation repack prototype (`BONSAI_Q8_0_X4=1`)

`block_q8_0` is 34 bytes (`{fp16 d; int8 qs[32]}`), so in a quantised activation row the 16-byte
quant loads land at `2 + 34k mod 16` and roughly half of them straddle a 16-byte boundary. The
prototype repacks the row into records designed so that **every load in the kernel is aligned**:

```
one record (144 B, 16-byte aligned) = exactly one even/odd iteration of the kernel
  qs[4][32]   128 B   activation blocks 4r..4r+3, each 32 B, each half 16-byte aligned
  d[4]          8 B   their four fp16 scales
  pad[8]              brings the record to a multiple of 16
```

* `bonsai_pack_q8_0_x4()` copies (never recomputes) the row; `packed block` and `packed scale` are
  pure accessor changes, which is why bit-exactness is structural rather than lucky.
* the repack runs **once per 16-row weight block** in `ggml_compute_forward_mul_mat_id_one_chunk`,
  because the same activation row is then read by all sixteen weight rows in that block (four
  four-row kernel calls). Per 4-row call it would be a pure loss, so it is not done there.
* the scratch buffer is a **thread-local 8 KB, 16-byte-aligned** buffer (one per ggml worker), sized
  by `bonsai_q8_0_x4_bytes()`; a row that does not fit or is not a multiple of 64 returns 0 and the
  block stays on the stock x4 path, so the switch can never silently truncate.
* the kernel it drives is the **spill-free row-pair arithmetic of mode 3** (`bonsai_x4_rows2_packed`
  is generated from the same macro body as `bonsai_x4_rows2_plain`, with a different activation
  accessor). That is deliberate: it makes `A6 - A5` a measurement of the repack alone.
* prototype scope, stated so it is not over-read: **expert path only**, hence 45.9% of decode. The
  dense `q2_0` hook is untouched (~2.5% of decode), and prefill is not packed yet, because in
  prefill every token has its own activation row and there is no sixteen-row reuse to amortise it.

**Verification, two levels.**

| level | check | result |
|---|---|---|
| kernel | 4 rows x 8 sizes, `nb in {1,2,3,5,8,16,32,64}`, vs `ggml_vec_dot_q2_0_q8_0` | **BIT-EXACT, `0.000e+00` on all 32**, including the odd `nb` sizes that exercise the padded partial record (144/288/432 B buffers) |
| graph | `s7_mmid_prefill_verify.sh` packed arm vs prefill arm, 6 `MUL_MAT_ID` shapes | **byte-identical on all 6**, with 256-4032 repacks per shape, i.e. the path demonstrably ran |

**Honest expectation, so the arm is not oversold.** The repack adds one 144 B write and one read per
record per 16 rows, against four reads of the same 136 B of quant data; that is roughly +18% of the
activation traffic for that block, and it buys aligned loads instead of boundary-straddling ones.
That could be a wash. The measured evidence for why it might still be worth it is that this kernel
is already close to its load pipes: `A1` moved `MUL_MAT_ID` from 61.9 to 52.5 ms/token while the
wall-clock decode only moved 4.5%, which says the kernel is not purely issue-bound. **`A6` decides,
and no number is claimed here.**

---

### B.7 (Task 3) Tonight's turnkey harness: `s7_shootout.sh`, `s7_arm_ll.sh`, `s7_ll_table.py`

Host-only. Nothing here was run on a device. **Disclosure:** the driver's preflight was executed
once by hand to prove it fails fast with no phone attached; that ran `adb get-state` only, no device
was present and no device state was touched. Everything else below is host-side.

#### B.7.1 The three requirement changes, and where each one lives

| requirement | where | how it is enforced |
|---|---|---|
| **12 warmups + 6 measured** for decode arms | `s7_arm_ll.sh` `WARM_N=12` | the default, and the warmups are now **timed and logged** (`S7WARM`) instead of discarded, so a run that has not plateaued is visible. `s7_ll_table.py` prints a `last warmup / measured median` ratio per arm and marks `STILL CLIMBING` below 0.90 |
| **no `--poll` on any ik arm** | `s7_shootout.sh` `IK_EXTRA_*` and `s7_arm_ll.sh`'s `ik` flag line | `--poll` appears only in the `ours` flag line, which is the shipping configuration there. A comment at each site records the Stage 6 §2.2 measurement (7.45 -> 2.53 t/s, -66%) as the reason |
| **identity gates + mirrored decode and prefill passes** | `s7_shootout.sh` | 8 identity arms then 7 timed configs in both phases, both directions (`A0 A5 A6 AP2 B0 B1 B2`, then reversed) |

One arm is deliberately *not* in the default set: `A1`/`A2`/`A3`/`A4` from Stage 6. `MOE_X4=1` is
already the shipped default, `MOE_X4=2` was never Gate-3-cleared, and `Q3K_X4` was refuted at
-5.7%; `ARMS=` still accepts them for a targeted re-run.

#### B.7.2 The arm matrix and what each arm decides

| arm | engine | configuration | what it decides |
|---|---|---|---|
| `A0` | ours | no env (shipped defaults: `MOE_X4=1`, `MULMATID_FLAT=2`) | the reference every delta is quoted against, in that pass |
| **`A5`** | ours | `BONSAI_MOE_X4=3` | whether the **zero-spill** bit-exact kernel beats the shipped spilling one (§B.6.1). Both are token-identical, so this is a pure speed question with zero correctness risk |
| **`A6`** | ours | `A5` + `BONSAI_Q8_0_X4=1` | the **aligned activation repack** (§B.6.2). Because `A6` is `A5` plus one switch, `A6 - A5` isolates the repack and nothing else |
| **`AP2`** | ours | `BONSAI_MOE_X4_PREFILL=1` | the **2-token prefill tile** (§B.5). Measured in *both* phases on purpose: it should be exactly neutral on decode (single-token decode never enters the paired loop), and a non-zero decode delta would mean the gate ran the path when it should not have |
| `B0` | ik | baseline, **no `--poll`** | the oracle re-run under the corrected protocol and the corrected warmup depth |
| `B1` | ik | `--cpu-strict 1 --cpu-mask-batch 0xff --cpu-strict-batch 1`, **no `--poll`** | the **clean** best-vs-best prefill comparison that Stage 6 could not make: `B1` there was poisoned by `--poll` |
| `B2` | ik | `B1` + `--no-fused-moe`, **no `--poll`** | the fused up+gate worth, measured at identical flags and without the `--poll` contamination |

Identity gates (all on the shipping build, greedy, `temperature 0`, `seed 1`):

| gate | reference | test | prompt |
|---|---|---|---|
| `MOE_X4=3` vs shipped | `id-A0` | `id-A5` | short, 64 tokens |
| `MOE_X4=3` + repack vs shipped | `id-A0` | `id-A6` | short, 64 |
| repack repeatability | `id-A6` | `id-A6b` | short, 64 |
| prefill tile vs shipped | `id-pf-A0` | `id-pf-AP2` | **~256-token prefill**, 16 tokens |
| prefill tile + repack vs shipped | `id-pf-A0` | `id-pf-A6` | ~256-token prefill, 16 |
| prefill tile repeatability | `id-pf-AP2` | `id-pf-AP2b` | ~256-token prefill, 16 |

The long prompt is not optional: the paired loop needs two routed rows, so a short prompt would
leave `AP2` never entered and "identity pass" would be vacuous. The `S7IDTOK` token array is hashed
and kept, so a divergence is reported as an index rather than "differs".

#### B.7.3 Two failure modes this driver is built to make impossible

1. **A silently unset environment variable.** An arm whose `BONSAI_*` variable did not reach the
   engine measures the baseline and reads as "no effect". So `s7_arm_ll.sh` prints `S7ENV` with the
   variables actually in force, the analyzer carries them as a column in the main table, and the arm
   definitions exist in exactly one place (`A0_ENV` ... `AP2_ENV`).
2. **Staging a build without the kernels.** `s7_shootout.sh` greps the shipping `libggml-cpu.so` for
   `bonsai_vec_dot_q2_0_q8_0_x4t2`, `bonsai_x4_rows2_plain`, `bonsai_x4_rows2_packed`,
   `bonsai_pack_q8_0_x4` and `bonsai_q2_0_x4_prefill_mode` and **aborts** if any is missing, before
   anything is pushed. It also aborts if ik lacks `--cpu-mask-batch`, because then `B1`/`B2` would
   silently degrade to `B0`.

Plus a host-only prerequisite mode, `HOST_CHECK=1 s7_shootout.sh 0`, which needs no phone at all:

```
$ HOST_CHECK=1 MoeDroid/tools/s7_shootout.sh 0
  OK   ship libggml-cpu.so            OK   ik libggml/libllama/server
  OK   ship llama-server launcher     OK   ik-prof libggml.so
  OK   prof libggml-cpu.so            OK   ik build has batch pinning
  OK   libc++_shared.so               OK   libomp.so
  OK   device arm script              OK   analyzer
  OK   BONSAI_PROF only in prof       OK   symbol bonsai_* (6 of 6)
  OK   analyzer selftest
=== host-side prerequisites: 0 failures ===
```

#### B.7.4 The analyzer is validated against synthetic logs before it is trusted

`s7_ll_table.py --selftest` builds a complete synthetic night (7 configs x 2 phases x 2 passes plus
identity arms) with **known** injected deltas and asserts the parser recovers each one:

```
PASS  A5 median parsed
PASS  A5 delta vs A0 is +5.00%
PASS  A6 delta vs A5 is +2.86%
PASS  AP2 decode contrast present
PASS  B1 vs B0 contrast present
PASS  steady-state table flags a plateau
PASS  identity: A5 identical to A0
PASS  identity: prefill tile diverges at index 2
PASS  env column recorded
SELFTEST PASSED: the parser recovers every injected delta
```

A synthetic log set is also kept at `.stage7/fakelogs/` (33 arm files) so the rendered tables can be
inspected without a phone. The report contains: the medians table with the env column, a
**steady-state table** (warmup tail vs measured median), the per-request series, within-pass deltas
against that pass's own `A0`, the named contrasts, group means with the pass-to-pass spread (the
night's position noise floor), the identity gate grouped by prompt family, and the engines' own
configuration lines.

#### B.7.5 Builds

| build | command | result |
|---|---|---|
| `upstream-llama/build-arm64-ship` | `ninja -C build-arm64-ship` | up to date, 30/30 targets; `libggml-cpu.so` 5,699,976 B, contains all six Stage 7 symbols |
| `upstream-llama/build-arm64-prof` | `make -j24` (Unix Makefiles, `-DBONSAI_PROF=1`) | green; `BONSAI_PROF` strings present in prof (2), **absent** in ship (0), `libggml-cpu.so` 5,712,688 B |

#### B.7.6 Time budget, stated before the night

Per timed arm: ~60-90 s model load + cap gate (up to 30 s) + 12 warmups (~2 min at 6.5 t/s over 64
tokens) + 6 measured requests (~1 min) + 120 s cooldown, i.e. **~5 min**. So:

| run | arms | wall |
|---|---|---|
| `s7_shootout.sh 0` (staging + 8 identity arms) | 8 | ~35 min |
| `ARMS="A0 A5 A6 AP2 B0 B1 B2" s7_shootout.sh 1` (decode only) | 7 | ~45 min |
| `s7_shootout.sh 2` (decode + prefill, both directions) | 28 | **~2.7 h** |
| `PROF=1` adds 5 profiler arms | +5 | ~+30 min |

The 12-warmup protocol is what makes each arm ~2.5 min more expensive than Stage 6's, and §A.3 says
that cost is not optional: at 4 warmups the same engine read 21% low.

#### B.7.7 Files delivered in this section

| file | role |
|---|---|
| `MoeDroid/tools/s7_arm_ll.sh` | device arm: 12 timed warmups, `S7ENV`, cap gate, ik without `--poll`, `S7*` output lines |
| `MoeDroid/tools/s7_shootout.sh` | host driver: host-check mode, model sha256 gate, symbol gate, staging, 8 identity gates, mirrored decode + prefill passes, optional profiler pass |
| `MoeDroid/tools/s7_ll_table.py` | analyzer with `--selftest`; steady-state, contrasts, position control, identity, config |
| `MoeDroid/tools/s7_mmid_prefill_verify.sh` | graph-level bit-identity of the tile on `MUL_MAT_ID` (§B.5) |
| `MoeDroid/tools/s7_dense_prefill_verify.sh` | graph-level bit-identity of the tile on the **dense** `mul_mat` call site |
| `MoeDroid/tools/s7_dense_prefill_test.c` | the dense-path test program |
| `MoeDroid/tools/s7_qemu_dl_stubs.c` | failing `dl*` stubs so a static Android test binary can link ggml |

#### B.7.8 What is verified tonight versus what needs the phone

| claim | status |
|---|---|
| `x4t2` two-pointer API: bit-exact, 0 B frame, 0 spills | **verified** (qemu + `clang -S`) |
| `x4t2` wired into `MUL_MAT_ID`: byte-identical, 4 of 6 shapes exercise it | **verified** (qemu, 6 shapes, 8 threads + 1 thread) |
| `x4t2` wired into dense `mul_mat`: byte-identical, 5 of 6 shapes exercise it | **verified** (qemu, includes an odd token count and a single-token control) |
| `MOE_X4=3`: bit-exact, 0 B frame, 0 spills | **verified** (qemu, 8 sizes) |
| `Q8_0_X4=1` repack: bit-exact at kernel level and graph level | **verified** (qemu; 6 shapes, 256-4032 repacks each) |
| harness syntax, staging logic, symbol gates, analyzer arithmetic | **verified** on the host (`HOST_CHECK=1`, `--selftest`, synthetic logs) |
| **speed of A5 / A6 / AP2**, and the clean `B0/B1/B2` oracle gap | **not measured** - needs the device. That is what the 2.7 h run above buys |

---

### B.8 What did not happen in this session, and why (open items for the night)

| item | state | why, and what it needs |
|---|---|---|
| **no device measurement at all** | open | the phone is unplugged and away for 12 hours. `A5`, `A6`, `AP2` and the clean `B0/B1/B2` numbers exist only as arms in §B.7; not one speed number in this document was measured on hardware |
| `BONSAI_MOE_X4=3` is **not** the default | open | it is bit-exact and spill-free, but `A3` is the precedent for a bit-exact kernel that lost 5.7%. It flips to default only after `A5` shows a positive within-pass pair |
| `BONSAI_MOE_X4_PREFILL` is **off** | open | wired and proven byte-identical on both call sites, but unmeasured. `AP2` decides |
| `BONSAI_Q8_0_X4` is **off** | open | prototype scope is the expert decode path; prefill is not packed (no 16-row reuse there) and dense `q2_0` is not wired (2.5% of decode). `A6 - A5` decides whether to extend it |
| `patches/llama.cpp-moedroid.patch` is stale | open, and it was already stale before this session | the tracked patch is from Oct 14 and contains none of the Stage 6/7 kernels: `arch/arm/iqk_x4.c` is an **untracked new file**, so regenerating the patch needs `git add -N` on it plus a reverse-apply test against a clean `1b70350` checkout. Constraint 12 in `GOAL.md` therefore does not currently hold for this tree, and that is worth fixing before any of this is handed to someone else |
| `build-arm64-ik` / `build-arm64-ik-prof` not rebuilt | **not needed** | no `ik_llama.cpp` source was touched this session, so the existing binaries are current |
| `s6_*` harnesses not updated | intentional | `s7_*` supersedes them for the Stage 7 run (12 warmups, no `--poll` on ik); the `s6` scripts stay as the record of what Stage 6 actually ran |
| 32-chunk paired PPL for mode 3 / packed | **not needed** | both are bit-exact (`0.000e+00` at kernel level and byte-identical at graph level), so Gate 1 applies and Gate 2/Gate 3 do not |

### B.9 Session result in one paragraph

The three host-only tasks are done and every claim in them is either a compile-time fact or a
byte-comparison on real ARM64 code under `qemu-aarch64-static`; none is a speed claim. The `x4t2`
prefill tile now takes the two activation pointers the expert path can actually supply and is wired
into both `_one_chunk` call sites behind `BONSAI_MOE_X4_PREFILL`, with byte-identical output on 12
synthetic `MUL_MAT_ID` and dense `MUL_MAT` shapes and a call counter proving the new path ran. The
decode kernel's 55 stack accesses are gone: `BONSAI_MOE_X4=3` compiles to a zero-byte frame with zero
local spills and is bit-exact on 8 sizes, and the reason it took five attempts is recorded because
the four failed ones are the useful part. The aligned activation repack (`BONSAI_Q8_0_X4`) exists,
is bit-exact at both levels, and is presented as an experiment rather than a win. Tonight's harness is
built, self-tested on synthetic logs, and gated so that the two ways this kind of run usually lies - a
silently unset env var, and a "bit-identical" comparison that never executed the new path - cannot
happen quietly. What is missing is the only thing the host cannot supply: 2.7 hours on a cool phone.
