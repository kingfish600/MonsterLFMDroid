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
 * LFM2.5-24B-A2B - Liquid AI's 24B MoE with only 2B parameters active per
 * token (arch key `lfm2moe`). At Q4_0 the whole model is 13.58 GB: it fits
 * fully in the RedMagic's 24 GB RAM, so decode is pure RAM-bandwidth -
 * measured 25.45 +/- 1.33 t/s at -t 8 on the prime+perf cores, ~3.6x the
 * 176B monster this fork was split off from.
 *
 * The Q4_0 here is the official QAD (Quantization-Aware Distillation)
 * checkpoint from LiquidAI, trained to recover accuracy lost to Q4_0 -
 * the right pick under the "no lower than Q4" rule. Single file, no shards,
 * no drafter: LFM2.5 ships no MTP head and speculation measured negative.
 *
 * Stock llama.cpp (e86f5fd+) knows lfm2moe - that is the engine this fork
 * ships in place of the Prism fork binary.
 */
object ModelCatalog {

    const val DEFAULT_TARGET_ID = "lfm25-qad-q4_0"

    private const val REPO =
        "https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF/resolve/main"

    private const val QAD = "LFM2.5-24B-A2B-QAD-Q4_0.gguf"

    val assets: List<ModelAsset> = listOf(
        ModelAsset(
            id = "lfm25-qad-q4_0",
            kind = AssetKind.TARGET_MODEL,
            fileName = QAD,
            url = "$REPO/$QAD",
            sizeBytes = 13_578_190_880L,
            summary = "LFM2.5-24B-A2B QAD Q4_0 - 24B MoE, 2B active per token. " +
                "Quantization-Aware Distillation Q4_0 recovers the perplexity " +
                "lost to plain Q4_0. 13.58 GB fits fully in RAM, where decode " +
                "measures ~25 t/s on the Snapdragon 8 Elite.",
            sha256 = "d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5",
        ),
    )
}
