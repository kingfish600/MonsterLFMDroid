package com.prismml.bonsaidroid.engine

import android.content.Context
import android.provider.Settings

/**
 * RedMagic / Nubia per-app performance mode.
 *
 * `NubiaperformanceMode` is a global setting listing the packages the OEM is
 * willing to run at full clocks, as `package+level` entries, e.g.
 * `com.pocketpalai+300,com.prismml.moedroid+300,`.
 *
 * **This is worth roughly 50% of decode speed and is invisible otherwise.**
 * Measured on the RedMagic 10s Pro with LFM2.5-24B-A2B QAD-Q4_0:
 *
 *   not listed : cpu0 1996 MHz, cpu6 (prime) 1017-1958 MHz -> 24.1 t/s
 *   listed +300: cpu0 3532 MHz, cpu6 (prime) 2841-3840 MHz -> 35.9 t/s
 *
 * The prime cores are the whole point of the `--cpu-mask 0xff` pin, and without
 * the entry the OEM simply never lets them boost. The original MonsterQwenDroid
 * shipped with this entry; a freshly built fork does not get one, so it silently
 * runs at half speed until the user adds it (Game Space UI is the supported path;
 * `adb shell settings put global NubiaperformanceMode ...` also works).
 *
 * Reading a global setting needs no permission. Non-Nubia devices simply have no
 * such key, so this returns null and the UI stays quiet.
 */
object PerfMode {

    private const val KEY = "NubiaperformanceMode"

    /** Raw setting value, or null if absent/unreadable. */
    fun raw(context: Context): String? =
        runCatching { Settings.Global.getString(context.contentResolver, KEY) }.getOrNull()

    /**
     * true  = this package is listed, so the OEM boost should apply
     * false = the setting exists but this package is missing (running at ~half speed)
     * null  = no such setting (not a Nubia device) or unreadable - say nothing
     */
    fun isBoosted(context: Context): Boolean? {
        val v = raw(context) ?: return null
        if (v.isBlank()) return null
        val me = context.packageName
        return v.split(',').any { it.substringBefore('+').trim() == me }
    }

    /** One-line human summary for the diagnostics bundle. */
    fun describe(context: Context): String = when (isBoosted(context)) {
        true -> "perf_mode: boosted (package listed in $KEY)"
        false -> "perf_mode: NOT BOOSTED - add this app in Game Space for ~50% more speed " +
            "(prime cores are capped without it). raw=$KEY:${raw(context)}"
        null -> "perf_mode: n/a (no $KEY on this device)"
    }
}
