package com.prismml.bonsaidroid.engine

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

/** Everything needed to boot the runtime. */
data class ModelSpec(
    val target: File,
    val draft: File? = null,   // drafter / MTP head GGUF
    /** llama-server `--spec-type` for [draft]: "draft-mtp" for an MTP head
     *  (self-contained, carries its own embedding), "draft-dspark" for a
     *  DSpark drafter. Empty falls back to draft-dspark. */
    val draftSpecType: String = "",
    val mmproj: File? = null,  // vision tower; unused for text-only v1
    /** Target GGUF carries an MTP head, so `--spec-type draft-mtp` can work. */
    val hasMtp: Boolean = false,
)

sealed interface CompletionEvent {
    data class Token(val text: String) : CompletionEvent
    /** Qwen3.x thinking content, when the server separates it out. */
    data class Reasoning(val text: String) : CompletionEvent
    /**
     * The model asked to call tools instead of answering. The harness executes them
     * and re-enters [BonsaiEngine.streamChat] with the results appended as
     * [ChatMessage.role] == "tool". Not terminal: a turn can produce tool calls and
     * then, once fed the results, a normal answer.
     */
    data class ToolCalls(val calls: List<ToolCall>) : CompletionEvent
    data class Done(
        val fullText: String,
        val reasoningText: String,
        val timings: Timings,
    ) : CompletionEvent
    data class Error(val message: String) : CompletionEvent
}

/** One function invocation requested by the model. */
data class ToolCall(
    val id: String,
    val name: String,
    /** Raw JSON object of arguments, exactly as the model emitted it. */
    val arguments: String,
)

/** Mirrors llama-server `timings` (fields optional across revisions). */
data class Timings(
    val promptN: Int = 0,
    val promptMs: Long = 0,
    val promptPerSecond: Float = 0f,
    val predictedN: Int = 0,
    val predictedMs: Long = 0,
    val predictedPerSecond: Float = 0f,
    /** DSpark speculative telemetry; zero/absent when drafting is off. */
    val draftN: Int = 0,
    val draftNAccepted: Int = 0,
)

enum class EngineState { IDLE, STARTING, LOADED, FAILED }

/** Observable runtime lifecycle — drives UI without log-diving. */
data class EngineStatus(
    val state: EngineState,
    val detail: String? = null,
    /** Seconds since start() began; meaningful while STARTING. */
    val elapsedSec: Int = 0,
)

interface BonsaiEngine {
    val isRunning: StateFlow<Boolean>
    val status: StateFlow<EngineStatus>
    val lastTimings: StateFlow<Timings?>
    /** Ring buffer of the underlying runtime's stdout/stderr for diagnostics. */
    val logTail: StateFlow<List<String>>

    /** Boots the runtime; suspends until it is ready to serve or throws. */
    suspend fun start(spec: ModelSpec, profile: EngineProfile)

    suspend fun stop()

    /**
     * Streams one chat completion. The server applies the model's chat template;
     * callers pass structured messages.
     */
    fun streamChat(
        messages: List<ChatMessage>,
        profile: EngineProfile,
        maxTokens: Int = 0, // 0 = use profile.outputBudget
        temperature: Float = profile.temperature,
        stop: List<String> = emptyList(),
        /** OpenAI-style function schemas (raw JSON). Empty disables tool calling. */
        tools: List<String> = emptyList(),
    ): Flow<CompletionEvent>
}

/**
 * A chat turn.
 *
 * @param toolCallId set on `tool` role messages so the server can match a result back
 *   to the call that produced it.
 * @param toolCallsJson on an assistant turn that requested tools: the raw JSON array
 *   of `tool_calls` exactly as the server emitted them. The server requires the
 *   assistant's request to be echoed back before the matching `tool` results, or it
 *   rejects the conversation as malformed.
 */
data class ChatMessage(
    val role: String,
    val content: String,
    val toolCallId: String? = null,
    val toolCallsJson: String? = null,
)
