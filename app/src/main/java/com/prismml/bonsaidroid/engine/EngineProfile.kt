package com.prismml.bonsaidroid.engine

import kotlinx.serialization.Serializable

/**
 * Tuned server/engine configuration. Presets encode the guidance from the
 * Bonsai-2 model card + KNOWN_ISSUES.md, adapted to Snapdragon 8 Elite.
 *
 * Bonsai-2 invariants (prismml docs):
 *  - reasoning counts against the output limit: budget >= 16384 or replies
 *    come back empty ("thinks forever, never answers")
 *  - reasoning_effort: low/medium/xhigh valid; "high" HTTP 500s; "low"
 *    behaves like xhigh; "medium" is the practical default
 *  - thinking sampler: temp 1.0, top_p .95, top_k 20, min_p .05,
 *    presence 0.0, repetition 1.0 (penalties are NOT in the GGUF)
 *  - q5_0 KV cache is several times slower; q8_0/q4_0/f16 are fine
 *  - leave the server at --reasoning auto so per-request reasoning_effort wins
 *  - PTQ1_0 on Vulkan currently decodes < 2 tok/s on non-Xe2 GPUs (upstream
 *    kernel status) -> the CPU path is the phone-class fast lane.
 */
@Serializable
data class EngineProfile(
    val name: String,
    val contextTokens: Int = 16_384,
    /** Output budget incl. thinking; below 16k Bonsai-2 answers come back empty. */
    val outputBudget: Int = 16_384,
    /** 0 = auto (all detected big cores). 8 Elite: 8 big cores. */
    val threads: Int = 0,
    /** GPU layers via Vulkan; 0 keeps everything on CPU (validated default). */
    /**
     * Layers offloaded to the GPU. 0 = CPU only.
     *
     * **Kept at 0 deliberately.** The OpenCL backend works and is measurably faster
     * (see below), but shipping it breaks the app: libggml-opencl.so has a DT_NEEDED on
     * libOpenCL.so, and an Android app cannot resolve /vendor/lib64/libOpenCL.so for a
     * DT_NEEDED link - not even though it is listed in /vendor/etc/public.libraries.txt.
     * The engine then fails to start entirely:
     *   CANNOT LINK EXECUTABLE "libllamaserver.so": library "libOpenCL.so" not found
     * A headless run from /data/local/tmp with LD_LIBRARY_PATH works, which is what made
     * this look fine during development; the app's linker namespace is not the same.
     *
     * To re-enable: libOpenCL.so must be bundled into jniLibs (and must then be able to
     * find the Adreno ICD), or the backend must be dlopen'd rather than DT_NEEDED.
     *
     * Measured when it did run, llama-bench, 3 reps, governors on walt:
     *   24B   pp256 30.25 -> 76.86 (2.54x)   tg64  16.97 -> 20.75 (1.22x)
     *   LFM3  pp512 92.50 -> 330.72 (3.58x)  tg128 23.41 -> 28.97 (1.24x)
     * Keeping MoE experts on CPU (-ncmoe) measured 0.41 t/s - unusable.
     */
    val nGpuLayers: Int = 0,
    val kvCacheTypeK: String = "q8_0",
    val kvCacheTypeV: String = "q8_0",
    val flashAttn: Boolean = true,
    /**
     * Prompt-cache RAM for single-user slot reuse (MiB). Sent as `--cache-ram`.
     *
     * Upstream's own default is 8192. This fork had been setting **24576 — 24 GB on a
     * 24 GB phone**, which is what kept exhausting swap and getting the model
     * OOM-killed: the cache is ordinary anonymous memory, so it is swappable, and it
     * grows as prompts are processed. Measured: an app server left running 11h21m had
     * **11.3 GB of swap** attributed to it.
     *
     * The cache is a nicety (it saves re-processing a repeated prompt) and is not worth
     * any of that. 1 GiB still holds roughly a dozen 8k-context prompt states, which is
     * far more reuse than a single-user chat can generate, and it costs 1/24th of the
     * memory. Raise it only with the model's footprint in mind: the 24B already takes
     * 12.6 GB before any cache.
     */
    val cacheRamMb: Int = 1024,
    /** DSpark drafter — only exists for Bonsai-1 packs. */
    val useDraftModel: Boolean = false,
    /** MTP head grafted into the target GGUF (Bonsai-2 mtp-lean). No -md. */
    /**
     * MTP default OFF. Measured on the RedMagic with a clean alternating A/B
     * (beta49): speculation off 1.65/1.45 tok/s (mean 1.55) versus draft-mtp
     * 0.83/0.79 (mean 0.81) - 0.52x, i.e. MTP HALVES throughput. Draft
     * acceptance was only 36-40%, far below the ~0.6 a viable setup needs, so the
     * drafting cost plus half-rejected drafts costs far more than the verified
     * tokens save. Enable with the MTP A/B harness to re-measure.
     */
    val useMtp: Boolean = false,
    /** Suppress ALL speculation (MTP and DSpark). Used by the MTP A/B baseline arm. */
    val noSpeculation: Boolean = false,
    /** Must equal the drafter's block size; smaller crashes per upstream docs. */
    val draftMax: Int = 4,
    /**
     * Sampling. These are NOT arbitrary and NOT inherited from the monster app -
     * they are the values **the model itself declares** in its GGUF metadata:
     *
     *     general.sampling.temp           = 0.1
     *     general.sampling.top_k          = 50
     *     general.sampling.penalty_repeat = 1.05
     *
     * Liquid's release notes and independent testing agree (temp 0.1 / top-k 50 /
     * repeat-penalty 1.05). The previous values here (temp 1.0, top-k 20, repeat 1.0)
     * were tuned for the 176B Flash-Next model this fork was split from, and running
     * a reasoning model at temperature 1.0 mostly buys variance and hallucination.
     * top_p/min_p are left as they were; the model declares nothing for them.
     */
    val temperature: Float = 0.1f,
    val topP: Float = 0.95f,
    val topK: Int = 50,
    val minP: Float = 0.05f,
    val presencePenalty: Float = 0.0f,
    val repetitionPenalty: Float = 1.05f,
    /** low|medium|xhigh (never "high" - HTTP 500). "" leaves the template default. */
    val reasoningEffort: String = "medium",
    val seed: Long = -1L,
    /** Runtime hygiene */
    /**
     * llama-server `--load-mode` (`-lm`): auto | none | mmap | mlock | mmap+mlock.
     *
     * **`mmap` is the default, and that is a reliability decision, not a speed one.**
     *
     * `none` is genuinely faster when the phone is quiet — it loads the weights into
     * *anonymous* memory and measured 36.63 ± 0.35 t/s against mmap's 30.53 ± 4.21.
     * But anonymous memory cannot be reclaimed without swapping, and on a 24 GB phone
     * with 12.58 GB of swap already exhausted, that makes this process the largest
     * killable object on the device. Observed twice, reproducibly:
     *
     *     llama-server exited (code=137)      <- SIGKILL, low-memory killer
     *
     * An app whose model dies mid-conversation is not usable, so the 20% is not worth
     * it. With `mmap` the weights are file-backed **clean** pages: under pressure the
     * kernel simply drops them and re-reads from flash, which costs speed but never
     * kills the process. Slower beats dead.
     *
     * Note `mlock` is NOT an option: it needs RLIMIT_MEMLOCK, which Android caps at
     * 64 KB for a non-root process (the engine only reads the limit; it does not raise
     * it).
     */
    val loadMode: String = "mmap",
    val noWarmup: Boolean = false,
    /** Expose the local server on Wi-Fi (LAN serving mode). */
    val exposeOnWifi: Boolean = false,
    /**
     * PTQ1_0 ARM kernel selector, forwarded as $GGML_PTQ1_ARM to llama-server.
     * 0 v4 (shipped default) | 1 v4b | 2 v5 | 3 v6 | 4 v7 | 5 v8 | 6 v9 | 7 v10
     * (needs GGML_PTQ1_NROWS=2) | 8 v11 | 9 v12. Applied on engine restart.
     * Only meaningful for PTQ1_0 weights; harmless otherwise.
     */
    val kernelVariant: Int = 0,
    /** Forwarded as $GGML_PTQ1_NROWS; 2 enables the nrc=2 tile (v10 only). */
    val nrows: Int = 1,
    /** Let $GGML_PTQ1_ARM_FILE switch kernels without restarting the server. */
    val liveKernelSwitch: Boolean = false,
    /** Extra env for the llama-server child (Vulkan workaround knobs etc.). */
    val vulkanEnv: Map<String, String> = emptyMap(),
    /**
     * llama-server `--poll` (0-100): how long the CPU threadpool spins before it
     * sleeps on a futex. Every MoE layer hits several barriers per token, so a
     * thread that has been de-prioritised can pay the futex wake latency at each
     * one. Higher values trade battery for steadier decode. -1 omits the flag.
     */
    val pollPercent: Int = 50,
    /** llama-server -tb: threads for prompt processing. Prefill wants more
     *  workers than generation does; measured optimum here is -t 2 -tb 8
     *  (3.4-3.6 t/s decode vs 1.6 at -t 6), because two workers stay inside one
     *  L2 cluster while six thrash across both clusters on every graph node.
     *
     *  pollPercent is nearly irrelevant at t=2: 100 -> 3.4 t/s, 0 -> 3.3,
     *  50 -> 3.2. The futex-wake disaster that made --poll 100 essential was a
     *  -t 6 effect, so only SPEED spins; everything else sleeps between tokens
     *  and stops burning power. */
    val threadsBatch: Int = 0,
    /** Cap on experts routed to per layer (BONSAI_MOE_TOPK); 0 = the model's own
     *  value. Measured on this phone: 8 gives ~+8% decode for ~+3-5% perplexity,
     *  6 gives ~+27% for ~+11% perplexity (PPL 7.67 -> 8.53, 8 chunks). It changes
     *  routing, so it is never on by default. */
    val moeTopK: Int = 0,
    /** Pin every worker inside the performance cluster (cpu0-5, one shared
     *  12 MB L2) instead of letting the scheduler spread them across both
     *  clusters. Measured on this phone, 64-token runs, full quality:
     *    -t 2 unpinned            3.5 t/s
     *    -t 6 --cpu-mask 0x3f     4.4 t/s   (+26%)
     *  Snapdragon 8 Elite has no L3, so a barrier line or activation shared
     *  across clusters pays a coherency round-trip on every graph node - which
     *  is what made small thread counts look optimal. Pinned, more workers buy
     *  memory-level parallelism instead. */
    val pinPerfCluster: Boolean = true,
    /** llama-server --cpu-mask-batch/--cpu-strict-batch: pin the prompt-processing
     *  pool to all eight cores, one worker per core.
     *
     *  Passing -tb does NOT make the batch pool inherit --cpu-mask. In
     *  postprocess_cpu_params the whole cpu-params struct is copied into the batch
     *  params only while the batch thread count is unset, so with -tb 8 the batch
     *  pool keeps an all-zero cpumask and no affinity is ever applied - it floats
     *  across both clusters. The CLI help's "default: same as --cpu-mask" holds only
     *  when -tb is absent. Confirmed on device: 19 of 24 threads carried
     *  Cpus_allowed_list 0-7 and the seven prefill workers ran on cpu6/7.
     *
     *  Measured on this phone on the varH dense rewrite, 256-token prompts, mirrored
     *  A/B/B/A, every arm on one cap profile: 7.47 -> 9.07 t/s prefill (+21.5%, 4/4
     *  pairs, t = 9.08). A single 256-token-prefill + 64-token-decode request measured
     *  +17.9% prefill and +3.5% decode, so generation does not regress. Measured only
     *  at -t 6 -tb 8; THERMAL_SAVER's -tb 4 is uncharacterised and opts out. */
    val pinBatchAllCores: Boolean = true,
) {
    companion object {
        // LFM2.5-24B-A2B: 2B active params per token (~1.2 GB touched at
        // Q4_0); the whole 13.58 GB file lives in RAM. Hybrid conv keeps KV
        // small, so context is cheap and decode wants all 8 cores.
        val QUALITY = EngineProfile(
            name = "QUALITY",
            contextTokens = 32_768,
            reasoningEffort = "xhigh", // most accurate with a big budget
            pollPercent = 0,
            threads = 8,
            threadsBatch = 0, // no -tb: batch pool then inherits -t 8 and the 0xff mask
        )
        val BALANCED = EngineProfile(
            name = "BALANCED",
            contextTokens = 8_192,
            reasoningEffort = "medium",
            pollPercent = 0,
            threads = 8,
            threadsBatch = 0, // no -tb: batch pool then inherits -t 8 and the 0xff mask
        )
        val SPEED = EngineProfile(
            name = "SPEED",
            contextTokens = 4_096,
            outputBudget = 4_096,
            reasoningEffort = "low",
            pollPercent = 100,
            threads = 8,
            moeTopK = 0,          // BONSAI-fork knob; no-op on the stock lfm2moe engine
            threadsBatch = 0, // no -tb: batch pool then inherits -t 8 and the 0xff mask
        )
        val THERMAL_SAVER = EngineProfile(
            name = "THERMAL_SAVER",
            contextTokens = 4_096,
            threads = 4,   // fewer workers = less heat, still inside one cluster
            threadsBatch = 4,
            pinBatchAllCores = false, // -tb 4 was never measured with a batch mask
            reasoningEffort = "low",
            pollPercent = 0,
        )
        // No Vulkan preset: this app ships a CPU-only engine (the Adreno shares
        // the same LPDDR5X, so it adds compute but no capacity), and QSA has no
        // Vulkan kernel upstream.

        val presets = listOf(QUALITY, BALANCED, SPEED, THERMAL_SAVER)
    }
}
