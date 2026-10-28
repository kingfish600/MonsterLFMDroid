package com.prismml.bonsaidroid.tools

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import okhttp3.OkHttpClient
import okhttp3.Request
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

/**
 * Web search tool.
 *
 * **Why this no longer uses DuckDuckGo.** The original implementation scraped
 * `lite.duckduckgo.com` with a POST, which worked when written. It stopped working
 * shortly after: DDG began returning an "anomaly" challenge page (HTTP 202, ~14 KB,
 * zero result anchors) for automated requests from this client, so the tool silently
 * returned nothing and the model correctly reported finding no results. Verified
 * against both `lite.` and `html.duckduckgo.com` — both blocked.
 *
 * **What it uses instead.** The user's own SearXNG instance, which is already running
 * locally in Docker (host port 8093) with `formats: [html, json]` and, importantly,
 * `limiter: false`. The phone reaches it over Tailscale, where it appears as
 * 100.103.111.81. This is strictly better than a public endpoint: no rate limiting, no
 * API key, no third party seeing the queries, and 20 results with real snippets.
 *
 * **Wikipedia fallback.** If SearXNG is unreachable — the common case being the phone
 * off Tailscale or away from home — the tool falls back to the MediaWiki search API,
 * which needs no key and is never blocked. Wikipedia alone is narrower than a web
 * search, so the result text says which backend answered; the model is told to be
 * honest about the difference rather than implying it searched the whole web.
 */
object WebSearch {

    private const val TAG = "BonsaiDroid"

    /** The user's SearXNG, over Tailscale. Change if the host address changes. */
    private const val SEARX = "http://100.103.111.81:8093"

    private const val WIKI = "https://en.wikipedia.org/w/api.php"

    private const val UA =
        "Mozilla/5.0 (Linux; Android 15) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"

    /** Keep snippets short: agent loops accumulate context and decode slows with it. */
    private const val MAX_SNIPPET = 240

    private val client = OkHttpClient.Builder()
        .connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    data class Result(val title: String, val url: String, val snippet: String)

    /** Which backend actually answered, so the model can be honest about coverage. */
    data class SearchOutcome(
        val backend: String,
        val results: List<Result>,
        val error: String? = null,
    )

    // ---------------------------------------------------------------- SearXNG

    private suspend fun searx(query: String, max: Int): List<Result>? = withContext(Dispatchers.IO) {
        runCatching {
            val url = "$SEARX/search?q=${URLEncoder.encode(query, "UTF-8")}&format=json"
            val req = Request.Builder().url(url).header("User-Agent", UA)
                .header("Accept", "application/json").get().build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    Log.i(TAG, "searx HTTP ${resp.code}")
                    return@runCatching null
                }
                val body = resp.body?.string().orEmpty()
                val root = json.parseToJsonElement(body).jsonObject
                val arr = root["results"] as? JsonArray ?: return@runCatching null
                arr.take(max).mapNotNull { el ->
                    val o = el as? JsonObject ?: return@mapNotNull null
                    val u = (o["url"] as? JsonPrimitive)?.content.orEmpty()
                    if (!u.startsWith("http")) return@mapNotNull null
                    Result(
                        title = (o["title"] as? JsonPrimitive)?.content.orEmpty().take(140),
                        url = u,
                        snippet = (o["content"] as? JsonPrimitive)?.content.orEmpty()
                            .replace(Regex("""\s+"""), " ").trim().take(MAX_SNIPPET),
                    )
                }
            }
        }.getOrNull()
    }

    // -------------------------------------------------------------- Wikipedia

    private suspend fun wikipedia(query: String, max: Int): List<Result>? = withContext(Dispatchers.IO) {
        runCatching {
            val url = "$WIKI?action=query&list=search&srsearch=" +
                URLEncoder.encode(query, "UTF-8") + "&format=json&srlimit=$max"
            val req = Request.Builder().url(url).header("User-Agent", UA).get().build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) return@runCatching null
                val body = resp.body?.string().orEmpty()
                val root = json.parseToJsonElement(body).jsonObject
                val hits = ((root["query"] as? JsonObject)?.get("search")) as? JsonArray
                    ?: return@runCatching null
                hits.mapNotNull { el ->
                    val o = el as? JsonObject ?: return@mapNotNull null
                    val title = (o["title"] as? JsonPrimitive)?.content.orEmpty()
                    if (title.isBlank()) return@mapNotNull null
                    Result(
                        title = title,
                        url = "https://en.wikipedia.org/wiki/" +
                            URLEncoder.encode(title.replace(' ', '_'), "UTF-8"),
                        snippet = (o["snippet"] as? JsonPrimitive)?.content.orEmpty()
                            .replace(Regex("""<[^>]+>"""), "")
                            .replace("&quot;", "\"").replace("&amp;", "&")
                            .replace(Regex("""\s+"""), " ").trim().take(MAX_SNIPPET),
                    )
                }
            }
        }.getOrNull()
    }

    // ------------------------------------------------------------------ entry

    /** Runs the search. Never throws; a total failure is reported as an error string. */
    suspend fun searchOutcome(query: String, max: Int = 5): SearchOutcome {
        searx(query, max)?.takeIf { it.isNotEmpty() }?.let {
            Log.i(TAG, "web_search searx \"$query\" -> ${it.size} results")
            return SearchOutcome("searxng", it)
        }
        Log.i(TAG, "web_search searx unavailable, trying wikipedia")
        wikipedia(query, max)?.takeIf { it.isNotEmpty() }?.let {
            Log.i(TAG, "web_search wikipedia \"$query\" -> ${it.size} results")
            return SearchOutcome("wikipedia", it)
        }
        Log.i(TAG, "web_search FAILED \"$query\" - both backends empty")
        return SearchOutcome(
            "none", emptyList(),
            error = "Search is unavailable right now (SearXNG unreachable and Wikipedia " +
                "returned nothing). Tell the user you could not search rather than guessing.",
        )
    }

    /** Compact text form handed back to the model as the tool result. */
    fun render(outcome: SearchOutcome, query: String): String = when {
        outcome.error != null -> outcome.error
        outcome.results.isEmpty() ->
            "No results for \"$query\"."
        else -> buildString {
            appendLine(
                if (outcome.backend == "wikipedia")
                    "Wikipedia results for \"$query\" (Wikipedia only - this was a fallback, " +
                        "not a full web search):"
                else
                    "Web search results for \"$query\":"
            )
            outcome.results.forEachIndexed { i, r ->
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
