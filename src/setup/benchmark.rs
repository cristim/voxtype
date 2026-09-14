//! `voxtype setup benchmark`: time the installed binary variants on this
//! machine and recommend one from the measurements.
//!
//! Each variant runs as its own process through `voxtype transcribe`, because
//! the packaged binaries in `/usr/lib/voxtype/` can be older than the binary
//! running the benchmark, and that command's output is the interface they all
//! share. The recording is of a known passage, so a variant that finishes
//! quickly with the wrong words (a GPU build silently failing, say) is not the
//! one recommended.

use super::binary::{self, Variant, LIB_DIR};
use crate::config::Config;
use anyhow::Context;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Text the user reads aloud. Plain words with no names or numbers, so the
/// word error rate measures the variant rather than the model's spelling.
pub const PASSAGE: &str = "Speech recognition runs on this computer without sending your voice \
anywhere. Please read these sentences at your normal pace and volume. Each installed build \
will transcribe the recording, and the fastest one that gets the words right will be \
recommended.";

/// A variant may be this much less accurate than the best result and still
/// be recommended for being faster.
const WER_TOLERANCE: f64 = 0.05;

/// Below this RMS the recording is treated as silence (roughly -46 dBFS).
const MIN_RMS: f32 = 0.005;

const MIN_RECORDING_SECS: f32 = 3.0;
const MAX_RECORDING_SECS: u64 = 60;
const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscribeOutput {
    pub model_load_secs: Option<f64>,
    pub transcribe_secs: Option<f64>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VariantResult {
    pub variant: Variant,
    pub binary_name: &'static str,
    /// Transcription time of each timed run, in seconds.
    pub runs_secs: Vec<f64>,
    pub median_secs: Option<f64>,
    pub model_load_secs: Option<f64>,
    /// `None` when no reference text was available to score against.
    pub word_error_rate: Option<f64>,
    pub transcript: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    engine: &'static str,
    audio_secs: f32,
    results: Vec<VariantResult>,
    recommended: Option<Variant>,
    warnings: Vec<String>,
}

pub async fn run(
    config: &Config,
    config_path: Option<&Path>,
    audio: Option<PathBuf>,
    runs: usize,
    save_audio: Option<PathBuf>,
    json: bool,
) -> anyhow::Result<()> {
    let engine = config.engine.name();
    let candidates = candidates(engine);
    if candidates.is_empty() {
        anyhow::bail!(
            "No installed variant in {} can run the {} engine on this machine, so there is \
             nothing to benchmark.",
            LIB_DIR,
            engine
        );
    }

    let mut warnings = Vec::new();
    if let Some(w) = load_warning() {
        warnings.push(w);
    }
    if crate::daemon_status::read_pid_if_alive().is_some() {
        warnings.push(
            "The voxtype daemon is running. It keeps a model loaded and competes for the \
             same CPU and GPU, which can slow every variant down."
                .to_string(),
        );
    }
    for w in &warnings {
        say(json, &format!("Note: {}", w));
    }

    // Holds the temporary recording until the benchmark finishes.
    let mut _recording: Option<tempfile::NamedTempFile> = None;
    let (wav_path, reference): (PathBuf, Option<&str>) = match audio {
        Some(path) => (path, None),
        None => {
            let samples = record_passage(config, json).await?;
            let file = tempfile::Builder::new()
                .prefix("voxtype-benchmark-")
                .suffix(".wav")
                .tempfile()?;
            write_wav(file.path(), &samples)?;
            if let Some(dest) = &save_audio {
                std::fs::copy(file.path(), dest)
                    .with_context(|| format!("Cannot save the recording to {}", dest.display()))?;
                say(json, &format!("Saved the recording to {}", dest.display()));
            }
            let path = file.path().to_path_buf();
            _recording = Some(file);
            (path, Some(PASSAGE))
        }
    };
    let audio_secs = wav_duration_secs(&wav_path)?;

    say(
        json,
        &format!(
            "\nBenchmarking {} variant(s) on {:.1}s of audio: one warm-up run each, then {} timed run(s).",
            candidates.len(),
            audio_secs,
            runs
        ),
    );

    let mut results: Vec<VariantResult> = candidates
        .iter()
        .map(|&v| VariantResult {
            variant: v,
            binary_name: v.binary_name(),
            runs_secs: Vec::new(),
            median_secs: None,
            model_load_secs: None,
            word_error_rate: None,
            transcript: None,
            error: None,
        })
        .collect();

    // Warm-up: fills the page cache with the model and builds any GPU shader
    // cache, which would otherwise be charged to whichever variant runs first.
    for result in &mut results {
        say(json, &format!("  warm-up  {}", result.variant.display()));
        if let Err(e) = run_variant(result.variant, &wav_path, config_path) {
            result.error = Some(e.to_string());
        }
    }

    // Alternate the order each round so thermal throttling and background
    // load do not consistently favour the same variant.
    for round in 0..runs {
        let mut order: Vec<usize> = (0..results.len()).collect();
        if round % 2 == 1 {
            order.reverse();
        }
        for i in order {
            let result = &mut results[i];
            if result.error.is_some() {
                continue;
            }
            say(
                json,
                &format!("  run {}/{}  {}", round + 1, runs, result.variant.display()),
            );
            match run_variant(result.variant, &wav_path, config_path) {
                Ok((output, wall_secs)) => {
                    result
                        .runs_secs
                        .push(output.transcribe_secs.unwrap_or(wall_secs));
                    if result.model_load_secs.is_none() {
                        result.model_load_secs = output.model_load_secs;
                    }
                    result.word_error_rate = reference.map(|r| word_error_rate(r, &output.text));
                    result.transcript = Some(output.text);
                }
                Err(e) => result.error = Some(e.to_string()),
            }
        }
    }
    for result in &mut results {
        result.median_secs = median(&result.runs_secs);
    }

    let recommended = pick_recommendation(&results);

    if json {
        let report = Report {
            engine,
            audio_secs,
            results,
            recommended,
            warnings,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&results, audio_secs, recommended);
    }
    Ok(())
}

/// Installed variants that can run `engine` on this CPU and GPU. Reads the lib
/// dir directly so a source build can still benchmark the packaged binaries.
fn candidates(engine: &str) -> Vec<Variant> {
    let cpu = binary::detect_cpu();
    let gpus = binary::detect_gpus();
    binary::enumerate_installed()
        .into_iter()
        .filter(|&v| {
            v.supports_engine(engine)
                && binary::variant_runs_on_cpu(v, &cpu)
                && binary::variant_gpu_available(v, &gpus)
        })
        .collect()
}

fn say(json: bool, line: &str) {
    // Keep stdout clean for the JSON report.
    if json {
        eprintln!("{}", line);
    } else {
        println!("{}", line);
    }
}

async fn record_passage(config: &Config, json: bool) -> anyhow::Result<Vec<f32>> {
    say(json, "Read this passage aloud:\n");
    say(json, &format!("  {}\n", PASSAGE));
    say(json, "Press Enter to start recording.");
    wait_for_enter().await?;

    let mut capture = crate::audio::create_capture(&config.audio)?;
    let mut chunks = capture.start().await?;
    // Drain the live chunk channel so it can never fill up; the full recording
    // comes back from stop().
    let drain = tokio::spawn(async move { while chunks.recv().await.is_some() {} });

    say(
        json,
        "Recording. Press Enter when you have finished reading.",
    );
    let _ = tokio::time::timeout(Duration::from_secs(MAX_RECORDING_SECS), wait_for_enter()).await;
    let samples = capture.stop().await?;
    drain.abort();

    let secs = samples.len() as f32 / SAMPLE_RATE as f32;
    if secs < MIN_RECORDING_SECS {
        anyhow::bail!(
            "The recording is only {:.1}s long. Read the whole passage before pressing Enter.",
            secs
        );
    }
    if rms(&samples) < MIN_RMS {
        anyhow::bail!(
            "The recording is silent, so the microphone is not delivering audio.\n\
             Check the input device with: voxtype info devices"
        );
    }
    Ok(samples)
}

async fn wait_for_enter() -> std::io::Result<()> {
    use tokio::io::AsyncBufReadExt;
    let mut line = String::new();
    tokio::io::BufReader::new(tokio::io::stdin())
        .read_line(&mut line)
        .await
        .map(|_| ())
}

fn write_wav(path: &Path, samples: &[f32]) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn wav_duration_secs(path: &Path) -> anyhow::Result<f32> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Cannot read {} as a WAV file", path.display()))?;
    let spec = reader.spec();
    Ok(reader.duration() as f32 / spec.sample_rate as f32)
}

/// Run one variant's `transcribe` on `wav`, returning its parsed output and the
/// process wall time.
fn run_variant(
    variant: Variant,
    wav: &Path,
    config_path: Option<&Path>,
) -> anyhow::Result<(TranscribeOutput, f64)> {
    let binary = Path::new(LIB_DIR).join(variant.binary_name());
    let mut cmd = Command::new(&binary);
    if let Some(path) = config_path {
        cmd.arg("--config").arg(path);
    }
    cmd.arg("transcribe").arg(wav);

    let start = Instant::now();
    let output = cmd
        .output()
        .with_context(|| format!("Cannot run {}", binary.display()))?;
    let wall_secs = start.elapsed().as_secs_f64();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last = stderr.lines().rev().find(|l| !l.trim().is_empty());
        anyhow::bail!(
            "{} exited with {}: {}",
            variant.binary_name(),
            output.status,
            last.unwrap_or("no error output")
        );
    }
    Ok((
        parse_transcribe_output(&String::from_utf8_lossy(&output.stdout)),
        wall_secs,
    ))
}

/// Parse `voxtype transcribe` stdout: timing comes from the "Model loaded in"
/// and "Transcription completed in" log lines, and the transcript is the text
/// after the final blank line.
pub fn parse_transcribe_output(stdout: &str) -> TranscribeOutput {
    let mut out = TranscribeOutput::default();
    let secs_after = |line: &str, marker: &str| -> Option<f64> {
        let rest = &line[line.find(marker)? + marker.len()..];
        let number: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        number.parse().ok()
    };

    let lines: Vec<String> = stdout.lines().map(super::accel::strip_ansi).collect();
    for line in &lines {
        if let Some(s) = secs_after(line, "Model loaded in ") {
            out.model_load_secs = Some(s);
        }
        if let Some(s) = secs_after(line, "Transcription completed in ") {
            out.transcribe_secs = Some(s);
        }
    }
    if let Some(blank) = lines.iter().rposition(|l| l.trim().is_empty()) {
        out.text = lines[blank + 1..].join(" ").trim().to_string();
    }
    out
}

fn normalize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Word error rate: word-level edit distance divided by the reference length.
/// Case and punctuation are ignored.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hw) in h.iter().enumerate() {
            let substitution = prev[j] + usize::from(rw != hw);
            cur[j + 1] = substitution.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len() as f64
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// The fastest variant whose word error rate is within `WER_TOLERANCE` of the
/// best. Without a reference text, the fastest variant that produced any text.
pub fn pick_recommendation(results: &[VariantResult]) -> Option<Variant> {
    let finished: Vec<&VariantResult> = results
        .iter()
        .filter(|r| r.error.is_none() && r.median_secs.is_some())
        .filter(|r| r.transcript.as_deref().is_some_and(|t| !t.is_empty()))
        .collect();

    let best_wer = finished
        .iter()
        .filter_map(|r| r.word_error_rate)
        .min_by(|a, b| a.total_cmp(b));

    finished
        .into_iter()
        .filter(|r| match (best_wer, r.word_error_rate) {
            (Some(best), Some(wer)) => wer <= best + WER_TOLERANCE,
            _ => true,
        })
        .min_by(|a, b| a.median_secs.unwrap().total_cmp(&b.median_secs.unwrap()))
        .map(|r| r.variant)
}

fn load_warning() -> Option<String> {
    let loadavg = std::fs::read_to_string("/proc/loadavg").ok()?;
    let load: f64 = loadavg.split_whitespace().next()?.parse().ok()?;
    let cores = std::thread::available_parallelism().ok()?.get() as f64;
    (load > cores * 0.75).then(|| {
        format!(
            "System load is {:.1} on {} cores. Other work will skew the timings; close it \
             or rerun later for results you can trust.",
            load, cores
        )
    })
}

fn print_report(results: &[VariantResult], audio_secs: f32, recommended: Option<Variant>) {
    println!();
    println!(
        "{:<22} {:>12} {:>11} {:>10} {:>11}",
        "Variant", "Transcribe", "Model load", "RT factor", "Word errors"
    );
    for r in results {
        let mark = if Some(r.variant) == recommended {
            "  ★"
        } else {
            ""
        };
        if let Some(err) = &r.error {
            println!("{:<22} failed: {}", r.variant.display(), err);
            continue;
        }
        let secs = |v: Option<f64>| v.map_or("n/a".to_string(), |s| format!("{:.2}s", s));
        println!(
            "{:<22} {:>12} {:>11} {:>10} {:>11}{}",
            r.variant.display(),
            secs(r.median_secs),
            secs(r.model_load_secs),
            r.median_secs.map_or("n/a".to_string(), |s| format!(
                "{:.2}x",
                s / audio_secs as f64
            )),
            r.word_error_rate
                .map_or("n/a".to_string(), |w| format!("{:.1}%", w * 100.0)),
            mark
        );
    }
    println!();

    let Some(best) = recommended else {
        println!("No variant produced a usable transcription, so there is no recommendation.");
        return;
    };
    println!(
        "Recommended on this machine: {} ({})",
        best.display(),
        best.binary_name()
    );
    if binary::active_variant() == Some(best) {
        println!("It is already the active variant.");
    } else {
        println!(
            "Switch with: sudo voxtype setup variant --to {}",
            best.binary_name()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured from `voxtype-vulkan transcribe` 1.0.1, colour codes included.
    const VULKAN_STDOUT: &str = "Loading audio file: \"speech_long.wav\"\n\
Audio format: 16000 Hz, 1 channel(s), Int\n\
Processing 76048 samples (4.75s)...\n\
\u{1b}[2m2026-09-14T09:47:02.928465Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m Using local whisper transcription mode\n\
\u{1b}[2m2026-09-14T09:47:03.983799Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m Model loaded in 1.05s\n\
\u{1b}[2m2026-09-14T09:47:05.211902Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m Transcription completed in 1.23s: \"This is a longer test ...\"\n\
\n\
This is a longer test of voice activity detection with multiple words and phrases.\n";

    #[test]
    fn parses_timing_and_transcript() {
        let out = parse_transcribe_output(VULKAN_STDOUT);
        assert_eq!(out.model_load_secs, Some(1.05));
        assert_eq!(out.transcribe_secs, Some(1.23));
        assert_eq!(
            out.text,
            "This is a longer test of voice activity detection with multiple words and phrases."
        );
    }

    #[test]
    fn no_speech_output_has_no_transcript() {
        let out = parse_transcribe_output(
            "Loading audio file: \"x.wav\"\nVAD: 0.00s speech (0.0% of audio)\nNo speech detected, skipping transcription.\n",
        );
        assert_eq!(out.text, "");
        assert_eq!(out.transcribe_secs, None);
    }

    #[test]
    fn word_error_rate_ignores_case_and_punctuation() {
        assert_eq!(word_error_rate("Hello, world.", "hello world"), 0.0);
        assert_eq!(
            word_error_rate("one two three four", "one too three four"),
            0.25
        );
        assert_eq!(
            word_error_rate("one two three four", "one three four"),
            0.25
        );
        assert_eq!(word_error_rate("one two", "one two three"), 0.5);
        assert_eq!(word_error_rate("one two", ""), 1.0);
        assert_eq!(word_error_rate("", ""), 0.0);
    }

    #[test]
    fn passage_scores_itself_perfectly() {
        assert_eq!(word_error_rate(PASSAGE, PASSAGE), 0.0);
    }

    fn result(v: Variant, median: f64, wer: Option<f64>) -> VariantResult {
        VariantResult {
            variant: v,
            binary_name: v.binary_name(),
            runs_secs: vec![median],
            median_secs: Some(median),
            model_load_secs: None,
            word_error_rate: wer,
            transcript: Some("text".to_string()),
            error: None,
        }
    }

    #[test]
    fn a_fast_but_wrong_variant_is_not_recommended() {
        let results = vec![
            result(Variant::WhisperNative, 1.3, Some(0.02)),
            result(Variant::WhisperVulkan, 0.4, Some(0.60)),
        ];
        assert_eq!(pick_recommendation(&results), Some(Variant::WhisperNative));
    }

    #[test]
    fn a_faster_variant_within_tolerance_wins() {
        let results = vec![
            result(Variant::WhisperNative, 1.3, Some(0.00)),
            result(Variant::WhisperVulkan, 0.7, Some(0.04)),
        ];
        assert_eq!(pick_recommendation(&results), Some(Variant::WhisperVulkan));
    }

    #[test]
    fn failed_and_empty_variants_are_skipped() {
        let mut failed = result(Variant::WhisperVulkan, 0.1, Some(0.0));
        failed.error = Some("exited with 1".to_string());
        let mut empty = result(Variant::WhisperAvx2, 0.2, None);
        empty.transcript = Some(String::new());
        let results = vec![
            failed,
            empty,
            result(Variant::WhisperNative, 1.3, Some(0.0)),
        ];
        assert_eq!(pick_recommendation(&results), Some(Variant::WhisperNative));
    }

    #[test]
    fn without_a_reference_the_fastest_wins() {
        let results = vec![
            result(Variant::WhisperNative, 1.3, None),
            result(Variant::WhisperVulkan, 0.7, None),
        ];
        assert_eq!(pick_recommendation(&results), Some(Variant::WhisperVulkan));
    }

    #[test]
    fn median_and_rms() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(rms(&[]), 0.0);
        assert!(rms(&vec![0.0; 1000]) < MIN_RMS);
        assert!(rms(&[0.1, -0.1, 0.1, -0.1]) > MIN_RMS);
    }
}
