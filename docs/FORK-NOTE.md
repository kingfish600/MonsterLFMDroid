# MonsterLFMDroid

Fork of MonsterQwenDroid engine/bench tooling, retargeted at **LiquidAI/LFM2.5-24B-A2B** (`lfm2moe`, 24B total / ~2.4B active). Do not push this repo to the original MonsterQwenDroid remote (push URL is deliberately poisoned).

## Thesis
On the RedMagic 10s Pro (24 GB RAM) the current 66 GB Q2_0 Qwen3.8 streams from UFS at ~1.33 GB/s and caps at ~7 t/s. LFM2.5 Q4_K_M is **14.53 GB — fits fully in the page cache** (~15.5 GB budget), removing the UFS wall AND freeing ~9 GB of RAM vs the current model, with ~4× fewer bytes touched per token. Expectation: decode 15–25 t/s, no more "dumb" Q2 mix.

## What's here
- `tools/deploy_lfm.sh` (host): pushes ARM64 engine + model to `/data/local/tmp/lfm` over adb.
- `tools/night_lfm.sh` (device): thermal/mem snapshot → llama-bench pp/tg sweep at `-t 4/6/8` with the proven flags (`-fa on`, q8_0 KV, `--cpu-mask 0x3f --cpu-strict 1`), then cold-vs-warm decode to prove RAM residency. `night.sh server` = long-lived GOAL.md-protocol server on :8080.

## Engine
Stock llama.cpp `master` @ `e86f5fd` (ndk r28, `-march=armv8.2-a+dotprod+i8mm+fp16+bf16`), built at `/mnt/nvme4tb/builds/arm64`. No BONSAI patches — they don't apply to this base and the RAM-resident regime changes what matters; patch-port (MOE_SUM, flat mul_mat_id) is the first optimization candidate once baseline numbers exist.

## Model file
`/mnt/nvme4tb/models/LFM2.5-24B-A2B-Q4_K_M.gguf` (14,526,258,464 B) from `LiquidAI/LFM2.5-24B-A2B-GGUF`. Fallback swap if page-cache pressure appears: `LFM2.5-24B-A2B-QAD-Q4_0.gguf` (13.58 GB, QAD = LiquidAI's quality-tuned Q4).

## Night result
(fill in after on-device run: llama-bench table, cold/warm decode, t/s vs Build 14 baseline 6.945/7.48)

Host x86 reference (16t, e86f5fd): decode 33.6 t/s, prefill 219 t/s.
Fallback if page-cache residency fails on-device: `LFM2.5-24B-A2B-QAD-Q4_0.gguf` (13.58 GB, sha d1c03f73… verified) is on the host at /mnt/nvme4tb/models/ — push it to /data/local/tmp/lfm/ and run night.sh with M pointed at it.
