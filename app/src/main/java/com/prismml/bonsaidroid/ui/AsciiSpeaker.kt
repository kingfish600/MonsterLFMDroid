package com.prismml.bonsaidroid.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

/**
 * ASCII "speaking" animation, shown while a reply is being read aloud.
 *
 * Deliberately drawn with block characters rather than a real waveform view: it renders
 * in the existing text surface with no extra dependency, and at 24 kHz mono there is no
 * meaningful amplitude data worth plotting from the UI thread anyway.
 *
 * The bars are a travelling sine sampled per column and quantised into the eight
 * partial-block glyphs (▁▂▃▄▅▆▇█), so consecutive frames look like a wave moving
 * through the text instead of noise. Two detuned waves are summed so it does not read
 * as a single clean tone.
 */
private val BLOCKS = listOf('▁', '▂', '▃', '▄', '▅', '▆', '▇', '█')

/** Rows of bars; more rows = taller animation. */
private const val ROWS = 3
private const val COLS = 34

private fun frameFor(tick: Int): List<String> = (0 until ROWS).map { row ->
    buildString {
        for (c in 0 until COLS) {
            val x = c.toDouble() / COLS
            // Two waves at different speeds, plus a per-row offset, so the three rows
            // do not move in lockstep.
            val a = kotlin.math.sin((x * 6.0) + tick * 0.45 + row * 0.9)
            val b = kotlin.math.sin((x * 11.0) - tick * 0.28 + row * 1.7) * 0.55
            val v = ((a + b) + 1.55) / 3.1              // -> roughly 0..1
            val idx = (v * (BLOCKS.size - 1)).toInt().coerceIn(0, BLOCKS.size - 1)
            append(BLOCKS[idx])
        }
    }
}

@Composable
fun AsciiSpeaker(
    playing: Boolean,
    modifier: Modifier = Modifier,
    tint: Color = MaterialTheme.colorScheme.primary,
) {
    var tick by remember { mutableIntStateOf(0) }
    LaunchedEffect(playing) {
        tick = 0
        while (playing) {
            delay(90)
            tick++
        }
    }
    // When idle, hold a flat line rather than an empty gap, so the block does not
    // jump the layout as playback starts and stops.
    val lines = if (playing) frameFor(tick) else List(ROWS) { "▁".repeat(COLS) }

    Column(modifier) {
        lines.forEach { line ->
            Text(
                text = line,
                color = tint,
                fontFamily = FontFamily.Monospace,
                fontSize = 15.sp,
                lineHeight = 15.sp,
            )
        }
    }
}
