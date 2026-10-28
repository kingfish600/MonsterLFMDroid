# MonsterLFMDroid — handoff (2026-10-27 ~16:45 America/Chicago)

Status: **installed and verified on the phone.** This fork is one of three
sibling apps split out of MonsterQwenDroid; LFM is the finished one.

## What this is
Fork of `MonsterQwenDroid` (Kotlin/Compose llama.cpp front-end, RedMagic
10s Pro) rebuilt around **LFM2.5-24B-A2B QAD-Q4_0** with the **stock
llama.cpp `e86f5fd` engine** and today's on-device tuning baked in.
Headline: **25.45 ± 1.33 t/s decode** (`-lm none`, t=8, all-core mask) vs
6.945 t/s for the 176B monster app — ~3.6×.

## Identity (so it coexists with the original app)
| | value |
|---|---|
| app label | `MonsterLFMDroid` |
| applicationId | `io.github.kingfish600.monsterlfmdroid` |
| version | code 3, `0.2.0-lfm` |
| namespace | `com.prismml.bonsaidroid` (kept — Kotlin untouched by rename) |
| APK | `app/build/outputs/apk/debug/app-debug.apk` (122.9 MB) |
| device | RedMagic 10s Pro `91251291018D`, Android 15/16, arm64, 24 GB RAM |
| model on device | `/storage/emulated/0/Android/data/io.github.kingfish600.monsterlfmdroid/files/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf` (13,578,190,880 B, pre-seeded — downloader sees it, no download needed) |

## Engine swap (the core trick)
- `app/src/main/jniLibs/arm64-v8a/` now holds the master build from
  `/mnt/nvme4tb/builds/arm64/bin/`: libllama, libllama-common,
  libllama-server-impl, libmtmd, libggml{,-base,-cpu}, plus the APK's old
  `libllamaserver.so` **stub replaced by master's `llama-server` stub**
  (7,320 B; its DT_NEEDED names match the app's layout 1:1 — pure drop-in)
  and the APK's `libbonsai_native.so` + `libc++_shared.so` (glue, llama-free).
- Stock llama.cpp knows arch key `lfm2moe`; the Prism fork binary does not.
  jniLibs were gitignored in the source repo → force-added in commit `166eec6`.
- **No NDK needed to build**: `cpp/CMakeLists.txt` is not wired into gradle;
  the engine is prebuilt .so only. Toolchain: JDK 17, gradle 8.14 (cached),
  `local.properties` → `sdk.dir=/home/kingfish600/workspace/.toolchain/android-sdk`.

## Code changes (commit `166eec6`)
- `build.gradle.kts`: new applicationId/version.
- `strings.xml`, `ModelsScreen.kt`, `SettingsScreen.kt`, `LlamaServerEngine.kt`:
  MonsterQwenDroid → MonsterLFMDroid.
- `ModelCatalog.kt`: single asset, LiquidAI QAD file, full sha256
  `d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5`,
  `hasMtp=false`, URL `huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/resolve/main/LFM2.5-24B-A2B-QAD-Q4_0.gguf`.
- `LlamaServerEngine.kt`: cpu-mask `0x3f`→`0xff` (old mask excluded the 2×
  4.32 GHz primes, −30% decode).
- **`-ctxcp 0` restored (2026-10-27 late, commit on top of the original build).**
  The earlier claim in this doc — "flag removed upstream, would crash startup" —
  was **wrong**. `-ctxcp` is defined at `common/arg.cpp:1709`, takes an int, and
  defaults to **32**; checkpoints are gated on `n_ctx_checkpoints > 0`
  (`server-context.cpp:1594`). So `-ctxcp 0` validly disables them, which is what
  the original app did on purpose — a checkpoint copies the whole KV mid-decode
  and stutters a linear chat. Dropping it silently re-enabled 32 of them.
  Settled empirically with the **x86 build of the same engine**
  (`/mnt/nvme4tb/builds/x86/bin`): an unknown flag is rejected before the model
  loads, so the app's exact argv with a fake `-m` path isolates parser errors.
  With `-ctxcp 0` the parser accepts everything and fails only on the fake path.
- `EngineProfile.kt`: presets `-t 6`→`-t 8`; `threadsBatch 8`→`0` (passing
  `-tb` silently defeats the cpu-mask — verified on device); SPEED `moeTopK`
  8→0 (BONSAI-only knob). `cacheRamMb` stays 24576 default (mmap-reclaimable,
  harmless).

## Verified on device (app's exact argv, llama-server from /data/local/tmp/lfm)
- load **1.2 s** RAM-resident (page cache warm), health OK, chat OK with the
  app's sampler params; q8_0 KV prompt-cache confirmed (`cached_tokens 19/23`).
- **Reasoning gotcha**: LFM thinks before answering — a 64-token budget
  returns empty content. App default `outputBudget=16384` is safe; SPEED's
  4096 is fine; never drop below ~512.
- Known-good bench config: `-ngl 0 -fa on --cache-type-k q8_0 --cache-type-v
  q8_0 --cpu-mask 0xff --cpu-strict 1 -t 8 -lm none` (−lm none = full-RAM,
  champion; app uses `-lm auto`, ~24.7 t/s, still fine).

## Build / install one-liner
```
cd MonsterLFMDroid && ANDROID_HOME=/home/kingfish600/workspace/.toolchain/android-sdk ./gradlew assembleDebug -x lint   # ~50 s
adb -s 91251291018D install -r -t app/build/outputs/apk/debug/app-debug.apk
```
`local.properties` is gitignored — create it on a fresh clone (one line, above SDK path).

## Pending / next
1. **Push to GitHub**: remotes in all three forks are deliberately poisoned
   (`https://INVALID.do-not-push-to-original.invalid/fork.git`). Needs real
   forks of kingfish600/MonsterQwenDroid + auth, then `git remote set-url`.
   **Never push to the original repo — hard user rule.** Original clone at
   `../MonsterQwenDroid` @ `88aaff6` stays untouched.
2. Tonight (night queue, `tools/deploy_lfm.sh` etc. at `/mnt/nvme4tb`):
   flint-nvfp4 + Qwen3.6-35B-A3B-UD-Q4_K_XL first runs (models staged on
   nvme4tb, device night.sh needs re-push); fill "Night result" rows in
   the Qwen36/flint `docs/FORK-NOTE.md`s; merge results into
   `../NIGHT-PLAN.md` (master notes file, workspace root — not in a repo).
3. Later ideas: MTP/spec-decode retest on LFM (negative so far), BONSAI
   MOE_SUM patch port (burst >30 t/s target), UD-IQ4_XS as Qwen alt quant.
4. Quant floor: **never below Q4** (user hard rule).

## Don'ts
- Don't kill host `flash_serve` (root PID 1541988, holds ~68 GB host RAM).
- One big GGUF in host RAM at a time; device has 24 GB — one 13–14 GB model max.
- Benchmarks must be same-session A/B (thermals dominate across hours; an
  idle "sustained" single-shot can read ~11 t/s after a long bench run —
  that's GameSpace scheduling, not the config).
- `git add -A` in these forks skips `.gitignore`d jniLibs — use `-f` if
  engine libs ever change again.
