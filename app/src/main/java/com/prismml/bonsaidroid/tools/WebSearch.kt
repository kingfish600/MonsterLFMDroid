package com.prismml.bonsaidroid.tools

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

/**
 * Web search tool — the first real tool for the agent loop.
 *
 * DuckDuckGo's **lite** endpoint, via POST. Verified working from the device class of
 * client: GET to the same host returns a page whose links are obfuscated behind
 * `uddg=` redirects, and the Instant Answer JSON API answers `202` with no body
 * (rate-limited). POSTing a form body to `https://lite.duckduckgo.com/lite/` returns
 * plain HTML with real results, so that is what this uses. No API key, no quota.
 *
 * Result size is deliberately small. An agent loop *accumulates* context and decode
 * speed falls off with it (measured ~31 t/s at 500 tokens, ~25 at 4K, ~15-17 at 8K),
 * so every tool result is a permanent tax on the rest of the conversation. We return
 * a handful of results with truncated snippets rather than a full page of text.
 */
object WebSearch {

    private const val ENDPOINT = "https://lite.duckduckgo.com/lite/"
    private const val UA =
        "Mozilla/5.0 (Linux; Android 15) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"

    /** Keep snippets short — see the class comment about context cost. */
    private const val MAX_SNIPPET = 240

    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(25, TimeUnit.SECONDS)
        .build()

    data class Result(val title: String, val url: String, val snippet: String)

    private val LINK = Regex(
        """<a[^>]+class=['"]result-link['"][^>]*href="([^"]+)"[^>]*>(.*?)</a>""",
        setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE),
    )
    private val SNIPPET = Regex(
        """<td[^>]+class=['"]result-snippet['"][^>]*>(.*?)</td>""",
        setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE),
    )
    private val TAGS = Regex("""<[^>]+>""")
    private val ENTITIES = mapOf(
        "&amp;" to "&", "&lt;" to "<", "&gt;" to ">", "&quot;" to "\"",
        "&#39;" to "'", "&apos;" to "'", "&nbsp;" to " ", "&hellip;" to "…",
        "&mdash;" to "—", "&ndash;" to "–", "&#x27;" to "'",
    )

    private fun clean(s: String): String {
        var out = TAGS.replace(s, "")
        for ((k, v) in ENTITIES) out = out.replace(k, v)
        out = out.replace(Regex("""&#(\d+);""")) { m ->
            m.groupValues[1].toIntOrNull()?.toChar()?.toString() ?: ""
        }
        return out.replace(Regex("""\s+"""), " ").trim()
    }

    /** Runs the search. Never throws — a failed search returns empty, and the model is told so. */
    suspend fun search(query: String, max: Int = 5): List<Result> = withContext(Dispatchers.IO) {
        runCatching {
            val body = ("q=" + java.net.URLEncoder.encode(query, "UTF-8"))
                .toRequestBody("application/x-www-form-urlencoded".toMediaType())
            val req = Request.Builder()
                .url(ENDPOINT)
                .header("User-Agent", UA)
                .header("Accept", "text/html")
                .post(body)
                .build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) return@runCatching emptyList()
                val html = resp.body?.string().orEmpty()
                val links = LINK.findAll(html).toList()
                val snippets = SNIPPET.findAll(html).map { clean(it.groupValues[1]) }.toList()
                links.take(max).mapIndexed { i, m ->
                    val url = m.groupValues[1]
                    Result(
                        title = clean(m.groupValues[2]).take(140),
                        url = url,
                        snippet = (snippets.getOrNull(i) ?: "").take(MAX_SNIPPET),
                    )
                }.filter { it.url.startsWith("http") }
            }
        }.getOrDefault(emptyList())
    }

    /** Compact text form handed back to the model as the tool result. */
    fun render(results: List<Result>, query: String): String =
        if (results.isEmpty()) {
            "No results for \"$query\". The search endpoint returned nothing usable."
        } else {
            buildString {
                appendLine("Search results for \"$query\":")
                results.forEachIndexed { i, r ->
                    appendLine("${i + 1}. ${r.title}")
                    appendLine("   ${r.url}")
                    if (r.snippet.isNotBlank()) appendLine("   ${r.snippet}")
                }
            }.trimEnd()
        }

    /** OpenAI-style function schema advertised to the model. */
    fun schema(): String = """
        {
          "type": "function",
          "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this whenever the answer depends on facts you are not certain of, or anything recent.",
            "parameters": {
              "type": "object",
              "properties": {
                "query": {"type": "string", "description": "The search query"}
              },
              "required": ["query"]
            }
          }
        }
    """.trimIndent()
}
