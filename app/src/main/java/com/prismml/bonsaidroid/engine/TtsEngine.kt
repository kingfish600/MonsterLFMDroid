package com.prismml.bonsaidroid.engine

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Text-to-speech via llama.cpp's `llama-tts` tool and the LFM3-Audio-2B model.
 *
 * `llama-tts` is a standalone binary rather than a server, so this runs it as a child
 * process and reads the WAV it writes. It is shipped as `libllamatts.so` because
 * Android only extracts files matching `lib*.so` from an APK's jniLibs; the name is a
 * packaging requirement, not a shared library.
 *
 * It links against exactly the libraries the app already ships (libllama-common,
 * libmtmd, libllama, libggml*) and has no separate impl library, so no extra engine
 * files are needed.
 *
 * Per the upstream README for this model: 19 languages via `--tts-lang`, `female` or
 * `male` via `--tts-voice`, no voice cloning, each frame is 80 ms, and sampling is
 * fixed by the model card (temperature 1.0, top-k 128, repeat-penalty 1.1) so the
 * usual sampler flags are ignored. Measured on host x86: 4.88 s of speech in 1.51 s.
 */
object TtsEngine {

    private const val TAG = "BonsaiDroid"
    private const val BIN = "libllamatts.so"

    /** Timeouts: model load is the slow part, generation is not. */
    private const val TIMEOUT_SEC = 600L

    data class Speech(val wav: File, val seconds: Double, val genSeconds: Double)

    /**
     * Synthesises [text] to a WAV in the cache dir. Blocking work runs on IO.
     *
     * @param voice "female" (default) or "male"
     * @param lang  ISO code, e.g. "en", "de", "ja" (19 supported)
     */
    suspend fun speak(
        context: Context,
        text: String,
        model: File,
        mmproj: File,
        voice: String = "female",
        lang: String = "en",
    ): Result<Speech> = withContext(Dispatchers.IO) {
        runCatching {
            require(text.isNotBlank()) { "nothing to say" }
            require(model.exists()) { "audio model missing: ${model.name}" }
            require(mmproj.exists()) { "audio mmproj missing: ${mmproj.name}" }

            val nativeDir = context.applicationInfo.nativeLibraryDir
            val exe = File(nativeDir, BIN)
            require(exe.exists()) { "$BIN not found in $nativeDir" }

            val out = File(context.cacheDir, "tts-${System.currentTimeMillis()}.wav")
            if (out.exists()) out.delete()

            val cmd = listOf(
                exe.absolutePath,
                "-m", model.absolutePath,
                "-mm", mmproj.absolutePath,
                "-p", text,
                "--tts-voice", voice,
                "--tts-lang", lang,
                "--output", out.absolutePath,
                "-t", "6",
            )
            Log.i(TAG, "tts: ${cmd.joinToString(" ").take(300)}")

            val pb = ProcessBuilder(cmd)
            pb.directory(context.cacheDir)
            pb.environment()["LD_LIBRARY_PATH"] = nativeDir
            val proc = pb.start()

            // Drain both streams or a chatty run can block on a full pipe.
            val stdout = StringBuilder()
            val t1 = Thread { proc.inputStream.bufferedReader().forEachLine { stdout.appendLine(it) } }
            val t2 = Thread { proc.errorStream.bufferedReader().forEachLine { stdout.appendLine(it) } }
            t1.start(); t2.start()

            val finished = proc.waitFor(TIMEOUT_SEC, TimeUnit.SECONDS)
            if (!finished) {
                proc.destroyForcibly()
                error("tts timed out after ${TIMEOUT_SEC}s")
            }
            t1.join(2000); t2.join(2000)

            val log = stdout.toString()
            if (proc.exitValue() != 0 || !out.exists() || out.length() == 0L) {
                error("tts failed (exit ${proc.exitValue()}): ${log.takeLast(300)}")
            }

            // The tool reports its own timings; parse them for the UI.
            val audioSec = Regex("""output audio = ([0-9.]+)s""").find(log)
                ?.groupValues?.get(1)?.toDoubleOrNull() ?: 0.0
            val genSec = Regex("""generation ([0-9.]+)s""").find(log)
                ?.groupValues?.get(1)?.toDoubleOrNull() ?: 0.0
            Log.i(TAG, "tts ok: ${out.length()} bytes, audio ${audioSec}s, gen ${genSec}s")
            Speech(out, audioSec, genSec)
        }
    }

    /** Deletes stale wavs so the cache does not grow without bound. */
    fun cleanup(context: Context) {
        runCatching {
            context.cacheDir.listFiles { f -> f.name.startsWith("tts-") && f.name.endsWith(".wav") }
                ?.sortedByDescending { it.lastModified() }
                ?.drop(3)
                ?.forEach { it.delete() }
        }
    }
}
