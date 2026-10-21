package com.prismml.bonsaidroid.data

import kotlinx.serialization.Serializable

enum class AssetKind { TARGET_MODEL, DRAFT_MODEL, MMPROJ }

@Serializable
data class ModelAsset(
    val id: String,
    val kind: AssetKind,
    val fileName: String,
    val url: String,
    val sizeBytes: Long,
    val summary: String,
    /** sha256 from the HF LFS pointer; empty = not verified on finalize. */
    val sha256: String = "",
    /** For drafters: id of the target model this drafter was trained against. */
    val compatibleWith: String = "",
    /**
     * True only if this GGUF carries an MTP head. Guards `--spec-type draft-mtp`:
     * a plain quant has no head, so the flag makes the draft context fail to
     * init (the old DSpark-hijack class of mistake, in reverse).
     */
    val hasMtp: Boolean = false,
    /**
     * Additional files that must sit beside [fileName] for the model to load.
     * llama.cpp resolves the sibling shards of a split GGUF by name, but only
     * if they exist on disk - so the downloader has to fetch them all.
     */
    val extraFiles: List<String> = emptyList(),
    val extraSizes: List<Long> = emptyList(),
    /** For drafters: llama-server `--spec-type` ("draft-mtp", "draft-dspark"). */
    val specType: String = "",
)

/**
 * One model, split across two GGUF shards.
 *
 * Qwen3.8-Flash-Next is a 176B-parameter hybrid MoE (125B MoE + 51B n-gram
 * embedding table + 4B MTP head) with only ~6B parameters activated per token:
 * 10 routed experts of 512, per layer, plus one shared expert. The n-gram table
 * is indexed by bigrams/trigrams and Qwen designed it to be offloadable, so the
 * streaming problem is the experts, not that table.
 *
 * This app ships a stock-llama.cpp engine, not the Prism fork, because
 * `qwen4exp` only exists upstream. That also means the weights must be a type
 * stock llama.cpp can read - Q2_0 here, not a Prism-only PTQ1_0.
 */
object ModelCatalog {

    const val DEFAULT_TARGET_ID = "flashnext-q2_0"

    private const val REPO =
        "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Q2_0"

    private const val REPO_MTP =
        "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP"

    private const val SHARD1 = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
    private const val SHARD2 = "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"

    /**
     * Shard 1 is the varH dense rewrite, not the byte stream the upstream repo serves.
     * varH requantises the 192 `blk.N.hc_{attn,ffn}_{up,down}.weight` matrices from BF16
     * to Q8_0, which removes 562.5 MiB of dense bytes per token for +0.092% perplexity
     * (32 chunks, t = 1.02) and measures +8.19% decode on device (4/4 mirrored pairs,
     * t = 7.09). Shard 2 is untouched and byte-identical to upstream, so only shard 1
     * differs: 37,623,740,192 -> 37,033,916,192 bytes.
     *
     * The hosted URL still serves the un-rewritten shard 1, so a fresh download will not
     * match [ModelAsset.sha256] below and the rewritten shard must be published before
     * this asset can be re-downloaded. A device that already holds varH is unaffected.
     */
    val assets: List<ModelAsset> = listOf(
        ModelAsset(
            id = "flashnext-q2_0",
            kind = AssetKind.TARGET_MODEL,
            fileName = SHARD1,
            url = "$REPO/$SHARD1",
            sizeBytes = 37_033_916_192L,
            summary = "Qwen3.8-Flash-Next, 176B MoE at ~2.4 bpw (Q2_0), with the 192 " +
                "hc_* dense matrices requantised BF16 -> Q8_0 (-562.5 MiB). " +
                "~6B params active per token, so decode touches ~0.7 GB of " +
                "experts; the other ~34 GB streams from flash. Both shards are " +
                "required - shard 2 alone is a 28.8 GB n-gram table.",
            sha256 = "e7c694526e82f633eda7f37b5e35ed7075fd39140018c99d73e305818ded12e2",
            extraFiles = listOf(SHARD2),
            extraSizes = listOf(28_800_138_432L),
        ),
        /**
         * MTP head: drafts up to N tokens which the target then verifies in one
         * pass. Verification is exact, so this can only change speed, never the
         * output - a weak head merely drafts less often.
         *
         * This is the SELF-CONTAINED variant on purpose. The `shared-` heads
         * borrow the target's embedding and output projection, which needs
         * cross-model tensor borrowing that stock llama.cpp does not have; they
         * fail to load with "tensor 'token_embd.weight' not found".
         */
        ModelAsset(
            id = "flashnext-mtp-q4km",
            kind = AssetKind.DRAFT_MODEL,
            fileName = "mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf",
            url = "$REPO_MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf",
            sizeBytes = 2_786_204_800L,
            summary = "Multi-token-prediction head for Flash-Next. Drafts the next " +
                "tokens, the target verifies them in one pass, so output is " +
                "byte-identical - it only changes speed. Pass --spec-draft-n-max 2. " +
                "Expect less benefit above temperature 0.",
            sha256 = "8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231",
            compatibleWith = "flashnext-q2_0",
            specType = "draft-mtp",
        ),
    )
}
