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
    val nGpuLayers: Int = 0,
    val kvCacheTypeK: String = "q8_0",
    val kvCacheTypeV: String = "q8_0",
    val flashAttn: Boolean = true,
    /** Prompt-cache RAM for single-user slot reuse (MiB). */
    val cacheRamMb: Int = 24576,
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
    /** Sampling: thinking-mode set from the Bonsai-2 model card. */
    val temperature: Float = 1.0f,
    val topP: Float = 0.95f,
    val topK: Int = 20,
    val minP: Float = 0.05f,
    val presencePenalty: Float = 0.0f,
    val repetitionPenalty: Float = 1.0f,
    /** low|medium|xhigh (never "high" - HTTP 500). "" leaves the template default. */
    val reasoningEffort: String = "medium",
    val seed: Long = -1L,
    /** Runtime hygiene */
    // mlock keeps weight pages resident so decode never refaults through
    // FUSE storage. If RLIMIT_MEMLOCK blocks it the server warns and
    // continues unlocked, so default-on is strictly better than off.
    val mlockWeights: Boolean = false,
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
