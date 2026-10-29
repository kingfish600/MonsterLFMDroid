# MonsterLFMDroid — handoff

**Last updated: 2026-10-29 16:20 America/Chicago.** Written to survive a context reset.
Everything below is measured on the device unless marked otherwise.

Owner: github.com/kingfish600. Repo: github.com/kingfish600/MonsterLFMDroid (MIT).
Device: RedMagic 10S Pro (NX789J), Snapdragon 8 Elite (SM8750), Adreno 830, 24 GB, 1 TB UFS.

> **Superseded numbers.** An earlier version of this file quoted **25.45 t/s** on the stock
> `e86f5fd` engine. Both are out of date. Current: **35.8 t/s** on `8642541` + perf mode, and
> **GPU offload is now on** (see §2). If you read a number elsewhere, check its date.

---

## 1. What this is

Fork of MonsterQwenDroid rebuilt around **LFM2.5-24B-A2B** (QAD-Q4_0, 12.65 GiB) with the
stock llama.cpp `8642541` engine. **The parent repo is read-only by owner instruction** — do
not modify `../MonsterQwenDroid`. All work happens here.

Sibling forks, all built and documented, none under active development:
`MonsterQwen36Droid` (Qwen3.6-35B, `a866ad3`), `MonsterFlintDroid` (flint, `95ed668`).

**MonsterQwenDroid is the partner project.** It is a separate, active CPU-only line of work
(Qwen3.8-Flash-Next varH). See §8 for the mutual testing strategy — the two share one phone
and must not test simultaneously.

---

## 2. Headline results (all on device)

| model | decode | prefill | notes |
|---|---|---|---|
| **LFM2.5-24B-A2B QAD-Q4_0** | **35.8 t/s** | **106.7 t/s** | CPU, perf mode, `--cpu-mask 0xff --cpu-strict 1` |
| **LFM3-1.7B QAD-Q4_0** | **43.5 t/s** | **228.8 t/s** | 993 MiB, loads in 0.74 s |
| LFM2.5-24B, GPU offload | 20.75 t/s (llama-bench tg64) | 76.86 t/s (pp256) | **Adreno 830 via OpenCL** |
| LFM3-1.7B, GPU offload | 28.97 t/s (tg128) | 330.72 t/s (pp512) | 3.58× prefill |
| monster (Qwen, parent) | 6.945 t/s | — | baseline for comparison |
| flint | 3.52 t/s | — | corrected from an earlier wrong 0.79 |
| Qwen3.6 IQ4_XS | 9.55 t/s | — | |

Sustained 24B decode over 5×300 tok: 36.15 / 35.73 / 36.14 / 36.02.

### GPU offload: measured, and it works

llama.cpp's **OpenCL backend names the Adreno 830 in the Snapdragon 8 Elite explicitly**
(`docs/backend/OPENCL.md`). The build now includes it (`-DGGML_OPENCL=ON`,
`-DGGML_OPENCL_USE_ADRENO_KERNELS=ON`). llama-bench, 3 reps, governors on `walt`:

```
24B   pp256 30.25 -> 76.86 (2.54x)   tg64  16.97 -> 20.75 (1.22x)
LFM3  pp512 92.50 -> 330.72 (3.58x)  tg128 23.41 -> 28.97 (1.24x)
```

**Decode gains least** — it is memory-bandwidth-bound and the Adreno shares the same LPDDR5X
as the CPU. **Prefill gains most** (compute-bound), which is what the agent loop pays on every
tool round. Flash attention makes no measurable difference (76.70 vs 76.86 pp).

Verified in the app's own path: ran `llama-server` on the device with the app's exact
arguments (`-ngl 99 --device GPUOpenCL`) — loads, listens, generates; 46.59 t/s decode on a
short prompt. The device supplies `libOpenCL.so` itself (`/vendor/lib64/`, listed in
`/vendor/etc/public.libraries.txt`), so the app ships `libggml-opencl.so` but **not**
`libOpenCL.so`.

### The +49% that was sitting on the table

`NubiaperformanceMode` (GameSpace) is a **+49% decode** switch — 24.1 → 35.9 t/s.

```bash
adb shell settings put global NubiaperformanceMode \
  "com.pocketpalai+300,com.prismml.moedroid+300,io.github.kingfish600.monsterlfmdroid+300,"
```

Original value backed up in `~/.perfmode-backup.txt`. Effect: cpu0 1996→3532 MHz, cpu6 (prime)
1017–1958→2841–3840 MHz.

**Always record the regime with every measurement**: governor per policy (`powersave` at idle,
`walt` under load; `performance` is root-only), GameSpace on/off. Cross-regime comparisons are
meaningless — the MonsterQwenDroid GOAL.md makes the same point.

---

## 3. Device layout

```
/sdcard/Android/data/io.github.kingfish600.monsterlfmdroid/files/models/
    LFM2.5-24B-A2B-QAD-Q4_0.gguf   13,578,190,880  sha d1c03f73ecaf61017d70419e...
    LFM3-1.7B-QAD-Q4_0.gguf         1,049,615,680  sha 43a7d09a0e9cba91b63eb705...
    LFM3-Audio-2B-Q4_0.gguf         1,049,613,856  (TTS backbone)
    mmproj-LFM3-Audio-2B-Q8_0.gguf    337,862,080  (TTS codec decoder — required)
```

Models must be `chmod 666` and the directory `chmod 777`, or the app (uid not in group 1078)
gets EACCES on `.part` and downloads appear to freeze.

Host model store: `/mnt/nvme4tb/models/`. Engine builds: `/mnt/nvme4tb/builds/`.
Device scratch: `/data/local/tmp/lfm3/` (engine), `/data/local/tmp/ocl/` (OpenCL build).

---

## 4. Build and deploy

```bash
# arm64 engine (NDK 28.2.13676358), the flags that matter:
cmake <src> -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-O3 -march=armv8.2-a+dotprod+i8mm+fp16+bf16" \
  -DCMAKE_CXX_FLAGS="-O3 -march=armv8.2-a+dotprod+i8mm+fp16+bf16" \
  -DGGML_OPENMP=OFF -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF
```

- **`-DGGML_OPENMP=OFF` is required.** The default pulls `libomp.so`, which is absent on Android.
- **Ninja is not installed** on this host — use `make -j$(nproc)`. The upstream OpenCL docs
  assume ninja; substitute Makefiles.
- OpenCL needs Khronos headers + an arm64 ICD loader in the NDK sysroot. Both are already
  built at `/mnt/nvme4tb/opencl/`; headers are installed into the NDK sysroot.
- `apk` build: `./gradlew assembleDebug -x lint` with
  `ANDROID_HOME=/home/kingfish600/workspace/.toolchain/android-sdk`.

**Reversible states** (engine swaps):
- `/mnt/nvme4tb/builds/arm64-e86f5fd-jniLibs-backup/` — original stock engine
- `/mnt/nvme4tb/builds/arm64-lfm3-jniLibs-preopencl/` — pre-OpenCL `jniLibs` (11 files)

---

## 5. App features added (beyond the parent)

- **Web search** (`tools/WebSearch.kt`) — DuckDuckGo `html/` then `lite/`, Wikipedia fallback.
  Runs **fully on-device**. See §6 for why the headers matter.
- **Tool calling / agent loop** (`ui/AppViewModel.kt`) — `MAX_TOOL_ROUNDS = 4`; tools are
  withheld on the final round so the model must answer instead of looping.
- **TTS** (`engine/TtsEngine.kt`) — LFM3-Audio-2B via `llama-tts`, packaged as
  `jniLibs/arm64-v8a/libllamatts.so` (Android only extracts `lib*.so`; the name is a packaging
  requirement, not a shared library). Measured 4.72 s of speech in 2.95 s (1.60× real-time).
- **ASCII speaking animation** (`ui/AsciiSpeaker.kt`) — travelling-sine waveform in block
  glyphs, two detuned waves, held flat when idle. Owner confirmed working ("green equalizer").
- **Model management UI** (`ui/ModelsScreen.kt`) — `● LOADED` badge, `Loading…`/`Unloading…`
  button states, and delete feedback that updates immediately.

---

## 6. Bugs found and fixed (with causes — do not regress these)

| symptom | cause | fix |
|---|---|---|
| Web search returned nothing | DDG answered an "anomaly" challenge page — **the request looked like a bot** (bare User-Agent). Not a block. | Full browser header set (`Accept-Language`, `Sec-Fetch-*`, `sec-ch-ua*`, `Upgrade-Insecure-Requests`). 10 results, zero anomaly. |
| DDG results cited duckduckgo.com | Results are `//duckduckgo.com/l/?uddg=…` redirects | Unwrap the `uddg` param |
| **TTS spoke the wrong language** | Text was passed verbatim, including `[🔍 web_search: …]` notes and `【2†L1-L4】` **CJK citation brackets** — CJK punctuation in English prose derails a 19-language speech model | `TtsEngine.cleanForSpeech()`: strip notes/citations/code/markdown, then a final sweep keeping only letters, digits, whitespace and ordinary punctuation |
| Delete gave no UI feedback | `isDownloaded()` reads the filesystem, so it is not observable. Collecting a revision flow was **not enough** — the `items()` lambda never *read* it, so Compose skipped the item and `isDownloaded()` was never called again | `val downloaded = remember(asset.id, diskRevision) { … }` — the read inside the item is the part that matters |
| `llama-server` OOM-killed (exit 137) | `--cache-ram 24576` = a **24 GB prompt cache on a 24 GB phone**. Upstream default is 8192 | Set to 1024 MiB. Swap free went 816 kB → 10.8 GB |
| Offload would have failed instantly | The offload path passed `--device Vulkan0`, left over from a Vulkan-based lineage this engine has no backend for | `--device GPUOpenCL` |
| App sampler defaults wrong | temp 1.0 / top_k 20 / repeat 1.0 inherited from the parent | Model-declared 0.1 / 50 / 1.05 |

### Measured and rejected — do not re-try

| thing | result |
|---|---|
| **`-ncmoe`** (MoE experts on CPU, Pithos-style) | **0.41 t/s** on the 24B. Unusable. Offload everything. |
| `--load-mode none` vs `mmap` | `none` untouched swap but `mmap` is more robust; the OOMs blamed on `none` were my own orphaned servers |
| `mlock` | Impossible — Android caps `RLIMIT_MEMLOCK` at 64 KB |
| DSpark speculation | Loses on x86 (34.36 vs 36.40); **100× slower on phone** (0.35 vs 35.9 t/s). Do not ship. |
| `--cache-ram 24576` | See §6 — it was the OOM cause |
| LFM3-VL-2B vision | Loads and processes images, but decodes **3.7 t/s with tower / 5.7 without** vs 43.5 for the same-size text model. Slow even *without* an image ⇒ upstream `lfm3.cpp`/mtmd. Re-test when upstream fixes it. |
| Pithos engine | Its win is disk-streaming for models that **don't fit RAM**. The 24B fits in 24 GB. Its NEON path is untested upstream. Not the right tool — but it is what pointed at the Adreno/OpenCL path. |

---

## 7. Operational hazards

- **Do not blind-tap UI coordinates.** The list scroll shifts which card sits under a given y.
  Twice I deleted the wrong model (the 24B, then LFM3-1.7B) this way. **Pair each Delete button
  with its own card title** from the uiautomator dump before tapping. Both were restored from
  the host and sha-verified.
- **The chat input row moves** between y≈292 and y≈818 depending on whether the chat has
  content. The send button is at the input's row, x≈2571. Re-dump before tapping.
- **"new" is at x≈439, "chats (N)" at x≈185** (y≈224). Tapping the wrong one opens Chat history.
- `pkill -f "llama-server"` self-matches the calling shell. Kill by explicit PID, or pattern
  `^ *[0-9]+ +llama-server`.
- **Clean up device `llama-server` processes after every test.** Orphans held 11.7 GB of swap
  and caused OOM kills that I initially misattributed to load mode.
- `python3` is absent on the device — device scripts must parse with grep/cut only.
- toybox `stat` has no `%n`.
- logcat is often empty for this app; the engine writes `files/runtime.log` instead.
- Do not kill the host's `flash_serve` (root PID, ~68 GB host RAM).

---

## 8. Mutual testing strategy (shared with MonsterQwenDroid)

One phone, two active projects. **They must not test at the same time** — different engines,
different apps, and both are memory-hungry.

### The rule
1. **Take the lock before touching the device**, release when done:
   `/mnt/nvme4tb/shared-test/device.lock` — write `{project, pid, purpose, started}`, delete when finished.
   If the lock exists and is fresh (<30 min), do not start; if stale, note it and take over.
2. **Record the regime in every result.** Non-negotiable, because it swings numbers by ~50%:
   governor per policy, GameSpace/perf mode on/off, and whether the app or a headless server ran.
3. **Report as mean ± stddev over ≥3 reps**, with the exact command line.
4. **Clean up**: kill device `llama-server` processes, release the lock.

### Shared resources
- **Models**: canonical store is `/mnt/nvme4tb/models/`. Do not duplicate; reference it.
- **Engine builds**: `/mnt/nvme4tb/builds/`. The **OpenCL build is reusable by both projects**
  (`arm64-opencl/`) — MonsterQwenDroid is deliberately CPU-only by its own GOAL, but its
  findings and this one's should still be cross-referenced.
- **Results**: append JSONL to `/mnt/nvme4tb/shared-test/results.jsonl` with
  `{ts, project, device_regime, model, quant, engine, cmd, metric, value, unit, reps, notes}`.
- **Findings**: cross-link the two handoffs. A fix in one is often relevant to the other —
  the header fix, the `--cache-ram` bug and the speech-cleaning bug were all app-level and
  apply to both apps.

### Known cross-project facts
- `NubiaperformanceMode` is worth **+49%** and applies to whichever package is listed. Both
  apps should be listed simultaneously so switching between them does not silently drop the boost.
- The device has **no little cores**: 2 prime (cpu6/7) + 6 perf (cpu0–5). Pin with
  `--cpu-mask 0xff --cpu-strict 1` (worth 1.65× vs bare `-t 8`).
- MonsterQwenDroid's GOAL.md carries the "regime caveat" and the long-lived-server protocol;
  use the same discipline here.

---

## 9. Open items

1. **Verify the OpenCL build in the app UI** — the engine path is proven with the app's exact
   arguments, but the app's own Load button has not been exercised end-to-end since the swap.
2. **LFM3-VL vision** — blocked on upstream; re-test when `lfm3.cpp`/mtmd improves.
3. **`GOAL.md` in this repo is stale** — it is a copy of MonsterQwenDroid's Qwen varH goal and
   does not describe this project. Needs rewriting or deleting.
4. Decide whether GPU offload should be user-toggleable (it helps prefill a lot but decode only
   ~1.2×, and it adds a 15 MB lib).
