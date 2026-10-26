# Changelog

## 0.1.1-alpha - signing fix

**Fixes an install failure reported on a OnePlus 13 / Android 16:**
`Parsing failed: APK contains no signature files.`

The cause was not a missing signature but a missing *scheme*. AGP disables v1
(JAR) signing whenever `minSdk >= 24`, and this app is `minSdk 31`, so the
0.1.0-alpha APK carried only an APK Signature Scheme v2 signature in the signing
block - with no `META-INF/CERT.SF` / `CERT.RSA` files at all. Android 7+ accepts
that, but some OEM installer paths (OxygenOS/ColorOS, and third-party file
managers) read the JAR manifest and report exactly that error.

Fixed by signing with all three schemes:

```kotlin
enableV1Signing = true
enableV2Signing = true
enableV3Signing = true
```

Verified on the rebuilt APK:

| | 0.1.0-alpha | 0.1.1-alpha |
|---|---|---|
| `META-INF/CERT.SF` + `CERT.RSA` | absent | **present** |
| v1 scheme (JAR signing) | false | **true** |
| v2 scheme | true | **true** |
| v3 scheme | false | **true** |
| `jarsigner -verify` | n/a | **jar verified** |

Note when checking this yourself: `apksigner verify --verbose` reports
`Verified using v1 scheme: false` on an APK that *does* have a v1 signature,
because it skips v1 verification when `minSdk >= 24` (v2/v3 suffice there). Pass
`--min-sdk-version 21` to force it, or check for `META-INF/CERT.RSA` directly.

The signing key is unchanged (same debug keystore), so this installs as an in-place
update over 0.1.0-alpha without losing app data - which matters here because the
model lives in the app's external files directory and is deleted on uninstall.

## 0.1.0-alpha - first public build

First alpha. Runs a ~176B-parameter hybrid MoE (Qwen3.8-Flash-Next, 512 experts,
top-10 routed, ~6B active per token) entirely on-device on a Snapdragon 8 Elite
phone, CPU only, from a 66 GB mixed-quant GGUF.

Measured on a RedMagic 10S Pro (24 GB, 1 TB): **4.4 t/s decode at full quality**,
**~7.3-8.6 t/s prefill on a 966-token prompt**, model loaded and streaming from storage.

Engine work on top of stock llama.cpp:

- BF16 dot product had no ARM SIMD path at all (x86-only); added a NEON
  `vshll` path and a hardware `vbfdotq_f32` path. **+40% decode, +55% prefill**
  measured in a mirrored A/B; BF16 matmul time in the profile fell 85%.
- Cluster pinning (`--cpu-mask 0x3f --cpu-strict 1`) keeps every worker inside
  one L2 cluster. This device has no L3, so unpinned threads pay a cross-cluster
  coherency round-trip on every graph node. **+26% decode** (3.5 -> 4.4 t/s),
  quality-neutral.
- `--poll` matters far less once threads are pinned (it was worth everything at
  unpinned `-t 6`); the app defaults to 0 and only the SPEED profile spins.
- Added a single-pass expert reduction op (`GGML_OP_MOE_SUM`) replacing a chain
  of 9 views+adds per layer. Removed ~30% of the decode graph's nodes with
  bit-identical output (and, measurably, zero speed change - see PERFORMANCE.md).
- Optional expert-count cap (`BONSAI_MOE_TOPK`) and expert-slice prefetch
  (`BONSAI_MOE_PREFETCH`), both off by default.

Honest notes: speculative decoding via an MTP head is **4x slower** on this
device (the 2.8 GB sidecar pushes the expert working set past the page-cache
cliff - measured, see PERFORMANCE.md), so it ships disabled. A uniform expert
cap of 6 is +27% speed for +11.3% perplexity and is opt-in behind SPEED.
