package com.prismml.bonsaidroid.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.prismml.bonsaidroid.data.ChatTurn
import com.prismml.bonsaidroid.data.ModelAsset
import com.prismml.bonsaidroid.data.ModelDownloader
import com.prismml.bonsaidroid.engine.ChatMessage
import com.prismml.bonsaidroid.engine.CompletionEvent
import com.prismml.bonsaidroid.engine.EngineProfile
import com.prismml.bonsaidroid.engine.LlamaServerEngine
import com.prismml.bonsaidroid.engine.ModelSpec
import com.prismml.bonsaidroid.harness.BenchHarness
import com.prismml.bonsaidroid.service.InferenceService
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import android.content.Context
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private val Context.bonsaiPrefs by preferencesDataStore(name = "bonsai_settings")
private val KEY_PROFILE = stringPreferencesKey("engine_profile")
// Persisted separately from the preset NAME: setProfile only saved the name, so any
// field tweak (including the baked-in PTQ1_0 kernel) was discarded on app restart.
private val KEY_KERNEL = intPreferencesKey("ptq1_kernel_variant")
private val KEY_NROWS = intPreferencesKey("ptq1_nrows")

/**
 * Single view-model shared by all screens. Holds the engine lifecycle, the
 * active chat, download state, and the harness wiring.
 */
class AppViewModel(app: Application) : AndroidViewModel(app) {

    private val ctx get() = getApplication<Application>()

    val engine = LlamaServerEngine(app)
    val downloader = ModelDownloader(app)
    val chats by lazy { (app as com.prismml.bonsaidroid.BonsaiApp).chats }
    val governor = (app as com.prismml.bonsaidroid.BonsaiApp).governor

    // ---- engine / model state ------------------------------------------ //

    val isRunning = engine.isRunning
    val engineStatus = engine.status
    val lastTimings = engine.lastTimings
    val logTail = engine.logTail
    val downloadState = downloader.state

    data class Loaded(val target: ModelAsset, val draft: ModelAsset?)
    private val _loaded = MutableStateFlow<Loaded?>(null)
    val loaded: StateFlow<Loaded?> = _loaded.asStateFlow()

    private val _busy = MutableStateFlow(false)
    val busy: StateFlow<Boolean> = _busy.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    /** Set when we degraded gracefully (e.g. dropped a failing MTP drafter). */
    private val _fallbackNote = MutableStateFlow<String?>(null)
    val fallbackNote: StateFlow<String?> = _fallbackNote.asStateFlow()

    private var currentSpec: ModelSpec? = null
    private val _profile = MutableStateFlow(EngineProfile.BALANCED)
    /** Observable profile — Settings collects this; persists via DataStore. */
    val profileFlow: StateFlow<EngineProfile> = _profile.asStateFlow()
    val profile: EngineProfile get() = _profile.value

    /** Filename of the currently loaded target GGUF, or null if idle. */
    val loadedModelName: String? get() = currentSpec?.target?.name

    init {
        // Restore last-selected profile so force-stop/reopen keeps intent.
        viewModelScope.launch {
            runCatching {
                val prefs = ctx.bonsaiPrefs.data.first()
                prefs.get(KEY_PROFILE)?.let { saved ->
                    EngineProfile.presets.firstOrNull { it.name == saved }?.let {
                        _profile.value = it
                    }
                }
                // re-apply the tuned kernel selector over the restored preset
                _profile.value = _profile.value.copy(
                    kernelVariant = prefs.get(KEY_KERNEL) ?: _profile.value.kernelVariant,
                    nrows = prefs.get(KEY_NROWS) ?: _profile.value.nrows,
                )
            }
        }
    }

    /**
     * Profile changes apply on the NEXT load by default. Hot-restarting a
     * 7 GB runtime to flip a flag caused memory-spike crashes (old process
     * still dying while the new one mmaps); explicit Unload+Load is safe.
     */
    fun setProfile(p: EngineProfile, restartIfNeeded: Boolean = false) {
        _profile.value = p
        governor.setBaseline(p)
        viewModelScope.launch {
            runCatching {
                ctx.bonsaiPrefs.edit {
                    it[KEY_PROFILE] = p.name
                    it[KEY_KERNEL] = p.kernelVariant
                    it[KEY_NROWS] = p.nrows
                }
            }
        }
        if (restartIfNeeded && isRunning.value) restart()
    }

    /**
     * Adopt the fastest kernel from a Kernel A/B run as the persisted default, so
     * the winning kernel is integrated without a rebuild. The server caches its
     * choice once, so the engine must be Unloaded and Loaded again to apply it.
     */
    fun adoptWinningKernel(variant: Int) {
        setProfile(_profile.value.copy(
            kernelVariant = variant,
            nrows = if (variant == 7) 2 else 1,
        ))
    }

    fun loadModel(target: ModelAsset, draft: ModelAsset?) {
        viewModelScope.launch {
            _busy.value = true
            _error.value = null
            _fallbackNote.value = null
            try {
                // Drafter only when the profile asks for it — a downloaded
                // drafter must NOT ride along in drafter-off profiles.
                val wantDraft = draft != null && _profile.value.useDraftModel
                val spec = ModelSpec(
                    target = downloader.fileFor(target),
                    draft = if (wantDraft) downloader.fileFor(draft!!) else null,
                    draftSpecType = if (wantDraft) draft!!.specType else "",
                    hasMtp = target.hasMtp,
                )
                currentSpec = spec
                engine.start(spec, _profile.value)
                _loaded.value = Loaded(target, if (wantDraft) draft else null)
                InferenceService.start(ctx, _profile.value.name)
            } catch (first: Throwable) {
                // Graceful degradation: if the DSpark drafter is why the boot
                // failed, retry without it rather than dead-ending.
                val usedDraft = currentSpec?.draft != null
                if (!usedDraft) {
                    _error.value = first.message ?: first.javaClass.simpleName
                    return@launch
                }
                _fallbackNote.value =
                    "MTP drafter failed to init — retrying without it. " +
                        "Reason: ${(first.message ?: first.javaClass.simpleName).take(140)}"
                try {
                    engine.stop()
                    val bareSpec = ModelSpec(target = downloader.fileFor(target))
                    currentSpec = bareSpec
                    _profile.value = _profile.value.copy(useDraftModel = false)
                    engine.start(bareSpec, _profile.value)
                    _loaded.value = Loaded(target, null)
                    InferenceService.start(ctx, _profile.value.name)
                } catch (second: Throwable) {
                    _error.value = "without-drafter retry also failed: " +
                        (second.message ?: second.javaClass.simpleName)
                }
            } finally {
                _busy.value = false
            }
        }
    }

    fun unload() {
        viewModelScope.launch {
            // Teardown must never propagate: uncaught exceptions in this
            // Main-scope coroutine kill the whole app process.
            try {
                engine.stop()
            } catch (t: Throwable) {
                android.util.Log.e("BonsaiVM", "engine.stop() failed", t)
            }
            runCatching { InferenceService.stop(ctx) }
            _loaded.value = null
        }
    }

    /**
     * Fire-and-forget restart for the UI. It must swallow failures: viewModelScope
     * has no CoroutineExceptionHandler here, so an escaping exception would take the
     * app down. restartAndWait() rethrows deliberately so the benchmark harness can
     * see a failed reload - hence the catch on this path only.
     */
    fun restart() {
        viewModelScope.launch { runCatching { restartAndWait() } }
    }

    /**
     * Restart the engine and suspend until it is actually healthy.
     *
     * Polling a flag instead is racy: between calling [restart] and its coroutine
     * reaching stop(), isRunning is still true from the previous engine, so a
     * `while (busy || !isRunning)` loop can exit immediately and hand a dead
     * server to the caller. Awaiting start() removes the race entirely.
     */
    suspend fun restartAndWait() {
        val spec = currentSpec ?: return
        _busy.value = true
        try {
            engine.stop()
            // Release the old weight mapping before asking for a new one. The A/B
            // reloads the model a dozen times; mapping the next 6 GB while the
            // previous mapping is still being torn down can leave two live at once,
            // which is enough for the OOM killer to take the app down - and that
            // shows up as a force-close, not as an exception we can catch.
            System.gc()
            kotlinx.coroutines.delay(1_500)
            engine.start(spec, _profile.value)
        } catch (t: Throwable) {
            _error.value = t.message ?: t.javaClass.simpleName
            throw t
        } finally {
            _busy.value = false
        }
    }

    // ---- chat ----------------------------------------------------------- //

    private val _turns = MutableStateFlow<List<ChatTurn>>(emptyList())
    val turns: StateFlow<List<ChatTurn>> = _turns.asStateFlow()

    private val _generating = MutableStateFlow(false)
    val generating: StateFlow<Boolean> = _generating.asStateFlow()

    /** Active session id; null = fresh chat, lazily created on first send. */
    private val _currentSessionId = MutableStateFlow<Long?>(null)
    val currentSessionId: StateFlow<Long?> = _currentSessionId.asStateFlow()

    private var genJob: Job? = null

    /**
     * Per-chat reasoning_effort override: "" = profile default | medium | xhigh | none.
     * "none" skips thinking entirely - ~1400 fewer decode tokens per answer, the
     * single biggest wall-clock lever at ~2 t/s. "low" is deliberately NOT offered:
     * it behaves like xhigh on this model (model card).
     */
    private val _reasoning = MutableStateFlow("")
    val reasoning: StateFlow<String> = _reasoning.asStateFlow()

    fun setReasoning(code: String) {
        _reasoning.value = code
        _currentSessionId.value?.let { sid ->
            chats.update(sid) { s -> s.copy(reasoningOverride = code) }
        }
    }

    /**
     * Start a NEW chat: clears the view and detaches from the previous session
     * (the old one stays in history). Before this was just a view clear and the
     * next message silently appended to the previous session.
     */
    fun newChat() {
        genJob?.cancel()
        _generating.value = false
        _turns.value = emptyList()
        // Eagerly open a fresh session; reuse the current one while it is still
        // empty so repeated taps do not spam history. (A null id here would
        // make the next send silently append to the previous chat again.)
        val cur = _currentSessionId.value?.let { chats.get(it) }
        _currentSessionId.value = when {
            cur != null && cur.turns.isEmpty() -> cur.id
            else -> chats.newSession(_loaded.value?.target?.fileName ?: "?").id
        }
        _reasoning.value = ""
    }

    /** Open a stored session from history. */
    fun openChat(id: Long) {
        genJob?.cancel()
        _generating.value = false
        val s = chats.get(id) ?: return
        _currentSessionId.value = id
        _reasoning.value = s.reasoningOverride
        _turns.value = s.turns.toList()
    }

    fun deleteChat(id: Long) {
        chats.delete(id)
        if (_currentSessionId.value == id) {
            _currentSessionId.value = null
            _turns.value = emptyList()
        }
    }

    /** Fire-and-forget suspending work from UI callbacks, errors surfaced. */
    fun launchUi(block: suspend () -> Unit) {
        viewModelScope.launch {
            try { block() } catch (t: Throwable) {
                if (t !is kotlinx.coroutines.CancellationException) _error.value =
                    t.message ?: t.javaClass.simpleName
            }
        }
    }

    fun send(text: String, systemPrompt: String = DEFAULT_SYSTEM_PROMPT) {
        if (text.isBlank()) return
        startGeneration(text.trim(), systemPrompt, persistUserTurn = true)
    }

    /**
     * Re-run the last user message. Drops the last assistant answer from both
     * the view and the stored session, then streams a fresh answer over the
     * same context. No-op while generating or without an assistant turn.
     */
    fun regenerate(systemPrompt: String = DEFAULT_SYSTEM_PROMPT) {
        if (_generating.value || !isRunning.value) return
        val list = _turns.value
        val lastAssistant = list.indexOfLast { it.role == "assistant" }
        if (lastAssistant < 0) return
        val trimmed = list.take(lastAssistant)
        val lastUser = trimmed.lastOrNull { it.role == "user" }?.content ?: return
        _turns.value = trimmed
        _currentSessionId.value?.let { sid ->
            chats.update(sid) { s -> s.copy(turns = trimmed.toMutableList()) }
        }
        startGeneration(lastUser, systemPrompt, persistUserTurn = false)
    }

    private fun startGeneration(
        userText: String,
        systemPrompt: String,
        persistUserTurn: Boolean,
    ) {
        if (!isRunning.value || _generating.value) return

        val userTurn = ChatTurn("user", userText)
        val assistantTurn = ChatTurn("assistant", "")
        _turns.value = _turns.value + userTurn + assistantTurn
        val sid = currentSessionId()
        // Persist the user turn immediately: a mid-stream crash used to lose
        // the whole exchange; only the (partial) answer is deferred.
        if (persistUserTurn) chats.append(sid, userTurn)
        // Per-chat reasoning override -> effective profile for THIS request.
        val eff = chats.get(sid)?.reasoningOverride.orEmpty()
        val prof = if (eff.isEmpty()) _profile.value else _profile.value.copy(reasoningEffort = eff)

        genJob = viewModelScope.launch {
            _generating.value = true
            // ADPF: hold boost for the whole generation (no-op below API 33).
            com.prismml.bonsaidroid.engine.PerfHint.acquire(ctx)

            // UI throttle. The streaming path appends to the turn list once per
            // token, and every append is a full Compose recomposition plus an O(n)
            // string rebuild of the answer so far. At the speeds this model now
            // reaches (~37 t/s) that is 37 recompositions/second and a quadratic
            // rebuild, and the UI thread competes with the engine for the same
            // cores. Measured cost: ~29.7 t/s rendered in the UI vs 36.96 t/s when
            // the identical server is driven directly. Buffer and flush at ~12 Hz
            // instead - visually identical, and the CPU goes back to the model.
            val pendingContent = StringBuilder()
            val pendingReasoning = StringBuilder()
            var lastFlushMs = 0L
            fun flushPending() {
                if (pendingContent.isNotEmpty()) {
                    appendToLastAssistant(pendingContent.toString())
                    pendingContent.setLength(0)
                }
                if (pendingReasoning.isNotEmpty()) {
                    appendReasoningToLast(pendingReasoning.toString())
                    pendingReasoning.setLength(0)
                }
                lastFlushMs = System.currentTimeMillis()
            }
            fun maybeFlush() {
                if (System.currentTimeMillis() - lastFlushMs >= 80) flushPending()
            }

            try {
                val apiMessages = buildList {
                    add(ChatMessage("system", systemPrompt))
                    addAll(_turns.value.dropLast(1).map { ChatMessage(it.role, it.content) })
                }
                // ---- agent loop -------------------------------------------------
                // The model may answer, or ask to call tools. On a tool call we run it,
                // append the assistant's own request plus the result, and go round
                // again. MAX_TOOL_ROUNDS bounds it so a confused model cannot spin
                // forever; every round also grows the context, and decode slows with
                // context, so this is a cost control as much as a safety one.
                val convo = apiMessages.toMutableList()
                val tools = if (toolEnabled()) listOf(
                    com.prismml.bonsaidroid.tools.WebSearch.schema()
                ) else emptyList()
                var round = 0
                while (true) {
                    var calls: List<com.prismml.bonsaidroid.engine.ToolCall>? = null
                    // On the final round, withhold the tools entirely. Otherwise the
                    // model happily searches forever - observed in practice: four
                    // near-identical queries ("newest iPhone model 2025", "latest
                    // iPhone release 2025", "iPhone 16", "Apple iPhone latest model")
                    // and then the cap fired with no answer given. Removing the tools
                    // leaves it no option but to answer from what it already has.
                    val roundTools = if (round >= MAX_TOOL_ROUNDS - 1) emptyList() else tools
                    engine.streamChat(convo, prof, tools = roundTools).collect { ev ->
                        when (ev) {
                            is CompletionEvent.Token -> {
                                pendingContent.append(ev.text); maybeFlush()
                            }
                            is CompletionEvent.Reasoning -> {
                                pendingReasoning.append(ev.text); maybeFlush()
                            }
                            is CompletionEvent.ToolCalls -> calls = ev.calls
                            is CompletionEvent.Error -> setError(ev.message)
                            else -> Unit
                        }
                    }
                    flushPending()
                    val requested = calls ?: break
                    if (round++ >= MAX_TOOL_ROUNDS - 1) break
                    // Echo the request back, then supply each result.
                    convo += ChatMessage(
                        role = "assistant",
                        content = "",
                        toolCallsJson = requested.joinToString(
                            prefix = "[", postfix = "]", separator = ","
                        ) { tc ->
                            """{"id":${jsonStr(tc.id)},"type":"function","function":{""" +
                                """"name":${jsonStr(tc.name)},"arguments":${jsonStr(tc.arguments)}}}"""
                        },
                    )
                    for (tc in requested) {
                        emitToolNote(tc)
                        val result = runTool(tc)
                        convo += ChatMessage(role = "tool", content = result, toolCallId = tc.id)
                    }
                }
                // Silent-empty completion = classic GPU numeric fault signature
                // (instant-EOS from corrupted logits). Never look like a reply.
                if (_profile.value.nGpuLayers > 0 && !lastAssistantProduced()) {
                    setError(
                        "GPU run returned no tokens (numeric fault suspected) — " +
                            "check ⧉ log / runtime.log",
                    )
                }
                chats.append(sid, ChatTurn("assistant", lastAssistantText()))
            } catch (_: kotlinx.coroutines.CancellationException) {
                // Stopped by user: keep and persist the partial answer.
                flushPending()
                chats.append(sid, ChatTurn("assistant", lastAssistantText()))
            } catch (t: Throwable) {
                flushPending()
                setError(t.message ?: t.javaClass.simpleName)
                val partial = lastAssistantText()
                if (partial.isNotBlank()) chats.append(sid, ChatTurn("assistant", partial))
            } finally {
                com.prismml.bonsaidroid.engine.PerfHint.release()
                _generating.value = false
            }
        }
    }

    fun stopGenerating() {
        genJob?.cancel()
        genJob = null
        _generating.value = false
    }

    private fun appendToLastAssistant(chunk: String) {
        val list = _turns.value.toMutableList()
        val idx = list.indexOfLast { it.role == "assistant" }
        if (idx >= 0) list[idx] = list[idx].copy(content = list[idx].content + chunk)
        _turns.value = list
    }

    private fun appendReasoningToLast(chunk: String) {
        val list = _turns.value.toMutableList()
        val idx = list.indexOfLast { it.role == "assistant" }
        if (idx >= 0) list[idx] = list[idx].copy(reasoning = list[idx].reasoning + chunk)
        _turns.value = list
    }

    private fun lastAssistantText(): String =
        _turns.value.lastOrNull { it.role == "assistant" }?.content.orEmpty()

    // ---- tools -------------------------------------------------------------

    /** Web search is always available; a settings toggle can gate this later. */
    private fun toolEnabled(): Boolean = true

    private fun jsonStr(s: String): String = buildString {
        append('"')
        for (c in s) when (c) {
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (c < ' ') append("\\u%04x".format(c.code)) else append(c)
        }
        append('"')
    }

    private fun toolArg(tc: com.prismml.bonsaidroid.engine.ToolCall, key: String): String? =
        runCatching {
            kotlinx.serialization.json.Json.parseToJsonElement(tc.arguments)
                .let { it as kotlinx.serialization.json.JsonObject }[key]
                ?.let { it as? kotlinx.serialization.json.JsonPrimitive }?.content
        }.getOrNull()

    /** Executes one requested tool. Never throws - failures become text for the model. */
    private suspend fun runTool(tc: com.prismml.bonsaidroid.engine.ToolCall): String =
        when (tc.name) {
            "web_search" -> {
                val q = toolArg(tc, "query").orEmpty()
                if (q.isBlank()) {
                    "Error: web_search needs a non-empty \"query\" string."
                } else {
                    com.prismml.bonsaidroid.tools.WebSearch.render(
                        com.prismml.bonsaidroid.tools.WebSearch.search(q), q
                    )
                }
            }
            else -> "Error: unknown tool \"${tc.name}\"."
        }

    /** Makes tool activity visible in the transcript instead of a silent pause. */
    private fun emitToolNote(tc: com.prismml.bonsaidroid.engine.ToolCall) {
        val shown = toolArg(tc, "query") ?: tc.arguments.take(80)
        appendToLastAssistant("\n\n[\uD83D\uDD0D ${tc.name}: $shown]\n\n")
    }

    private fun lastAssistantProduced(): Boolean {
        val t = _turns.value.lastOrNull { it.role == "assistant" } ?: return false
        // Reasoning-only output (open <think> block) counts as produced text.
        return t.content.isNotBlank() || t.reasoning.isNotBlank()
    }

    /** Public so the Harness NEON probe can report its numbers in the error slot. */
    fun setError(msg: String) { _error.value = msg }

    // ---- Vulkan op self-test -------------------------------------------- //

    private val _selfTest = MutableStateFlow<List<String>>(emptyList())
    val selfTest = _selfTest.asStateFlow()
    private var selfTestJob: Job? = null
    private var selfTestFile: java.io.File? = null
    private val selfTestAll = StringBuilder()

    /** Runs the packaged test-backend-ops binary (Vulkan vs CPU reference). */
    fun runGpuSelfTest() {
        if (selfTestJob?.isActive == true) return
        selfTestJob = viewModelScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            _selfTest.value = listOf("starting...")
            selfTestAll.setLength(0)
            selfTestFile = try {
                java.io.File(
                    ctx.getExternalFilesDir(null) ?: ctx.filesDir,
                    "selftest.log",
                ).apply { delete(); createNewFile() }
            } catch (_: Throwable) { null }
            val bin = java.io.File(
                ctx.applicationInfo.nativeLibraryDir,
                "libtestops.so",
            )
            if (!bin.exists()) {
                _selfTest.value = listOf("missing: $bin")
                return@launch
            }
            try {
                val proc = ProcessBuilder(listOf(bin.absolutePath))
                    .redirectErrorStream(true)
                    .apply {
                        environment()["LD_LIBRARY_PATH"] =
                            ctx.applicationInfo.nativeLibraryDir
                    }
                    .start()
                var lastPush = 0L
                java.io.BufferedReader(
                    java.io.InputStreamReader(proc.inputStream),
                ).useLines { lines ->
                    lines.forEach { line ->
                        selfTestAll.append(line).append('\n')
                        runCatching { selfTestFile?.appendText(line + "\n") }
                        // Throttle UI pushes: recomposition-per-line makes the
                        // text unselectable and burns battery.
                        val now = System.currentTimeMillis()
                        if (now - lastPush > 400) {
                            lastPush = now
                            _selfTest.value =
                                selfTestAll.split('\n').takeLast(30)
                        }
                    }
                }
                val code = proc.waitFor()
                _selfTest.value = selfTestAll.split('\n').takeLast(30) +
                    "exit=$code"
                selfTestFile?.appendText("exit=$code\n")
            } catch (t: Throwable) {
                _selfTest.value = listOf(
                    "error: " + (t.message ?: t.javaClass.simpleName),
                )
            }
        }
    }

    /** Full transcript to clipboard (the streamed view is tail-only). */
    fun copySelfTest() {
        val cm = ctx.getSystemService(android.content.ClipboardManager::class.java)
        cm?.setPrimaryClip(
            android.content.ClipData.newPlainText(
                "bonsai-selftest",
                selfTestAll.toString().ifEmpty { "(no output yet)" },
            ),
        )
    }

    fun clearSelfTest() {
        selfTestJob?.cancel()
        _selfTest.value = emptyList()
    }

    private fun currentSessionId(): Long {
        _currentSessionId.value?.let { return it }
        // Prefer the most recent stored session only when nothing is open yet
        // (first send after app start); otherwise a "new chat" opens a session.
        val id = chats.sessions.value.firstOrNull()?.id
            ?: chats.newSession(_loaded.value?.target?.fileName ?: "?").id
        _currentSessionId.value = id
        return id
    }

    // ---- harness --------------------------------------------------------- //

    val evalHarness: com.prismml.bonsaidroid.harness.EvalHarness by lazy {
        com.prismml.bonsaidroid.harness.EvalHarness(ctx, engine)
    }

    val benchHarness: BenchHarness by lazy {
        BenchHarness(
            appContext = getApplication(),
            engine = engine,
            specProvider = { currentSpec!! },
            thermalProbe = { governor.readMaxRelevantTemp() },
            memAvailableProbe = {
                com.prismml.bonsaidroid.nativeffi.BonsaiNative.readMemInfoKb()
                    .getOrNull(1)?.toLong()?.div(1024) ?: -1L
            },
            restartWithProfile = { p ->
                // setProfile MUST come first: restartAndWait() restarts using
                // _profile.value, so without this the profile the harness asks for is
                // silently ignored and every arm runs the previous one. That bug made
                // the MTP A/B run draft-mtp in BOTH arms.
                setProfile(p, restartIfNeeded = false)
                // restartAndWait() suspends on start(), which blocks in awaitHealthy,
                // so there is no need to poll or guess a delay.
                withTimeoutOrNull(240_000) { restartAndWait() }
                kotlinx.coroutines.delay(300)
            },
        )
    }

    override fun onCleared() {
        genJob?.cancel()
        viewModelScope.launch { runCatching { engine.stop() } }
        engine.shutdown()
        super.onCleared()
    }

    companion object {
        /**
         * Cap on tool round-trips per user message. Each round re-sends the whole
         * conversation, and decode slows as context grows, so this bounds both a
         * confused model and the cost of a long agent chain.
         */
        private const val MAX_TOOL_ROUNDS = 4

        const val DEFAULT_SYSTEM_PROMPT =
            "You are LFM2.5-24B-A2B running locally on an Android phone. " +
                "Be concise unless asked for depth. " +
                "You have a web_search tool. Use it for facts you are unsure of or " +
                "anything recent - but search at most twice, then answer with what " +
                "you have. Do not repeat the same query with slight rewording. " +
                "Cite the URLs you actually used. If a search returns nothing " +
                "useful, say so rather than guessing."
    }
}
