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
import java.net.URLDecoder
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

/**
 * Web search tool — runs entirely on the device.
 *
 * ## Why this is DuckDuckGo again
 *
 * An earlier version scraped DDG and then appeared to break: DDG started answering
 * with an "anomaly" challenge page (HTTP 202, ~14 KB, zero result anchors). The
 * conclusion drawn at the time — that DDG blocks automated clients — was wrong. The
 * request was simply *shaped* like a bot: a bare User-Agent and nothing else. With the
 * header set a real browser sends (`Accept`, `Accept-Language`, `Accept-Encoding`,
 * `Sec-Fetch-*`, `sec-ch-ua*`, `Upgrade-Insecure-Requests`), the same endpoint returns
 * HTTP 200 with 10 real results and no challenge at all. Verified both ways from the
 * host before changing this file.
 *
 * That matters because the intermediate fix routed searches through a SearXNG instance
 * on the user's home machine over Tailscale — which works, but means searches leave the
 * phone. This keeps the whole path on-device, which is the point of the app.
 *
 * ## Layla's approach, for the record
 *
 * Layla AI unlocks search through an "Agents" mini-app. Those mini-apps are HTML apps
 * running in a React Native WebView — their docs say to treat them as "browser apps
 * running in a WebView" — so their DuckDuckGo calls are made from a real browser
 * context. That is the robust version of what the header set approximates. If DDG ever
 * tightens up again, a WebView-backed fetch is the next step rather than another
 * endpoint.
 *
 * Wikipedia remains the fallback: no key, never blocked, and the result text says which
 * backend answered so the model can be honest about coverage.
 */
object WebSearch {

    private const val TAG = "BonsaiDroid"

    private const val DDG = "https://html.duckduckgo.com/html/"
    private const val WIKI = "https://en.wikipedia.org/w/api.php"

    /** A realistic Android Chrome UA. Paired with the headers below; either alone fails. */
    private const val UA =
        "Mozilla/5.0 (Linux; Android 15; NX789J) AppleWebKit/537.36 (KHTML, like Gecko) " +
            "Chrome/131.0.0.0 Mobile Safari/537.36"

    private const val MAX_SNIPPET = 240

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    data class Result(val title: String, val url: String, val snippet: String)

    data class SearchOutcome(
        val backend: String,
        val results: List<Result>,
        val error: String? = null,
    )

    /**
     * The header set that makes DDG answer a non-browser client. Omitting the
     * `Sec-Fetch-*` / `sec-ch-ua*` group is what produced the challenge page.
     */
    private fun browserRequest(url: String) = Request.Builder()
        .url(url)
        .header("User-Agent", UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9," +
                "image/avif,image/webp,image/apng,*/*;q=0.8",
        )
        .header("Accept-Language", "en-US,en;q=0.9")
        .header("Sec-Fetch-Dest", "document")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Site", "none")
        .header("Sec-Fetch-User", "?1")
        .header("Upgrade-Insecure-Requests", "1")
        .header("sec-ch-ua", "\"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"")
        .header("sec-ch-ua-mobile", "?1")
        .header("sec-ch-ua-platform", "\"Android\"")
        .get()
        .build()

    private val LINK = Regex(
        """<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>""",
        setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE),
    )
    private val SNIPPET = Regex(
        """class="result__snippet"[^>]*>(.*?)</a>""",
        setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE),
    )
    private val TAGS = Regex("""<[^>]+>""")

    private fun clean(s: String): String =
        TAGS.replace(s, "")
            .replace("&amp;", "&").replace("&quot;", "\"").replace("&#x27;", "'")
            .replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace(Regex("""\s+"""), " ")
            .trim()

    /**
     * DDG wraps result links as `//duckduckgo.com/l/?uddg=<percent-encoded>&rut=...`.
     * Unwrap to the real destination, or the model cites duckduckgo.com redirects.
     */
    private fun unwrap(href: String): String {
        val raw = if (href.startsWith("//")) "https:$href" else href
        val m = Regex("""[?&]uddg=([^&]+)""").find(raw) ?: return raw
        return runCatching { URLDecoder.decode(m.groupValues[1], "UTF-8") }.getOrDefault(raw)
    }

    private suspend fun duckduckgo(query: String, max: Int): List<Result>? =
        withContext(Dispatchers.IO) {
            runCatching {
                val url = "$DDG?q=${URLEncoder.encode(query, "UTF-8")}"
                client.newCall(browserRequest(url)).execute().use { resp ->
                    if (!resp.isSuccessful) {
                        Log.i(TAG, "ddg HTTP ${resp.code}")
                        return@runCatching null
                    }
                    val body = resp.body?.string().orEmpty()
                    if (body.contains("anomaly", ignoreCase = true) &&
                        !body.contains("result__a")
                    ) {
                        Log.i(TAG, "ddg returned a challenge page (${body.length} bytes)")
                        return@runCatching null
                    }
                    val links = LINK.findAll(body).toList()
                    val snippets = SNIPPET.findAll(body).map { clean(it.groupValues[1]) }.toList()
                    links.take(max).mapIndexedNotNull { i, m ->
                        val u = unwrap(m.groupValues[1])
                        if (!u.startsWith("http")) return@mapIndexedNotNull null
                        Result(
                            title = clean(m.groupValues[2]).take(140),
                            url = u,
                            snippet = (snippets.getOrNull(i) ?: "").take(MAX_SNIPPET),
                        )
                    }
                }
            }.getOrNull()
        }

    private suspend fun wikipedia(query: String, max: Int): List<Result>? =
        withContext(Dispatchers.IO) {
            runCatching {
                val url = "$WIKI?action=query&list=search&srsearch=" +
                    URLEncoder.encode(query, "UTF-8") + "&format=json&srlimit=$max"
                val req = Request.Builder().url(url).header("User-Agent", UA).get().build()
                client.newCall(req).execute().use { resp ->
                    if (!resp.isSuccessful) return@runCatching null
                    val root = json.parseToJsonElement(resp.body?.string().orEmpty()).jsonObject
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

    /** Runs the search. Never throws; a total failure is reported as an error string. */
    suspend fun searchOutcome(query: String, max: Int = 5): SearchOutcome {
        duckduckgo(query, max)?.takeIf { it.isNotEmpty() }?.let {
            Log.i(TAG, "web_search duckduckgo \"$query\" -> ${it.size} results")
            return SearchOutcome("duckduckgo", it)
        }
        Log.i(TAG, "web_search ddg unavailable, trying wikipedia")
        wikipedia(query, max)?.takeIf { it.isNotEmpty() }?.let {
            Log.i(TAG, "web_search wikipedia \"$query\" -> ${it.size} results")
            return SearchOutcome("wikipedia", it)
        }
        Log.i(TAG, "web_search FAILED \"$query\" - both backends empty")
        return SearchOutcome(
            "none", emptyList(),
            error = "Search is unavailable right now (DuckDuckGo and Wikipedia both " +
                "returned nothing). Tell the user you could not search rather than guessing.",
        )
    }

    fun render(outcome: SearchOutcome, query: String): String = when {
        outcome.error != null -> outcome.error
        outcome.results.isEmpty() -> "No results for \"$query\"."
        else -> buildString {
            appendLine(
                if (outcome.backend == "wikipedia")
                    "Wikipedia results for \"$query\" (Wikipedia only - this was a " +
                        "fallback, not a full web search):"
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
