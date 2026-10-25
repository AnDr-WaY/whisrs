//! Streaming stress harness for the `WaylandVkKeyboard` backend.
//!
//! Drives the backend exactly the way the daemon does under real streaming
//! transcription: a sentence mixing English + a non-Latin script is split into
//! MANY small `type_text` deltas (1–3 words each, like the deltas a streaming
//! ASR emits), all through ONE long-lived `WaylandVkKeyboard`. A separate
//! focused client (`vk_recorder`) records the exact text the compositor
//! delivered. We compare per iteration and report the exact-match rate and
//! which characters were dropped.
//!
//! This is the regression metric for the keymap-apply race: before the fix,
//! mixed-script streaming intermittently drops spaces and freshly-introduced
//! glyphs; after the fix it must be 100%.
//!
//! SAFETY: spawned by `run_stress.sh` against an ISOLATED headless compositor
//! (its own `XDG_RUNTIME_DIR` under /tmp). It refuses to run if pointed at the
//! real `/run/user/*` runtime dir. It never touches the user's live session.
//!
//! Usage:
//! ```text
//! vk_stress --recorder <path> --iters N --delay-ms M [--lang ar|ru] [--verbose]
//! ```

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

use xkb_type::wayland_vk::WaylandVkKeyboard;

/// A line-protocol handle to the spawned recorder client.
struct Recorder {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Recorder {
    fn spawn(recorder_path: &str) -> anyhow::Result<Self> {
        let mut child = Command::new(recorder_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut rec = Self {
            child,
            stdin,
            stdout,
        };
        // Wait for READY.
        let line = rec.read_line()?;
        anyhow::ensure!(
            line.trim() == "READY",
            "recorder did not report READY, got {line:?}"
        );
        Ok(rec)
    }

    fn read_line(&mut self) -> anyhow::Result<String> {
        let mut s = String::new();
        let n = self.stdout.read_line(&mut s)?;
        anyhow::ensure!(n > 0, "recorder stdout closed unexpectedly");
        Ok(s)
    }

    fn reset(&mut self) -> anyhow::Result<()> {
        writeln!(self.stdin, "RESET")?;
        self.stdin.flush()?;
        let line = self.read_line()?;
        anyhow::ensure!(line.trim() == "OK", "expected OK after RESET, got {line:?}");
        Ok(())
    }

    fn dump(&mut self) -> anyhow::Result<String> {
        writeln!(self.stdin, "DUMP")?;
        self.stdin.flush()?;
        let line = self.read_line()?;
        let rest = line
            .strip_prefix("TEXT ")
            .ok_or_else(|| anyhow::anyhow!("expected TEXT line, got {line:?}"))?;
        let text: String = serde_json::from_str(rest.trim_end())?;
        Ok(text)
    }

    fn keymaps(&mut self) -> anyhow::Result<u32> {
        writeln!(self.stdin, "KEYMAPS")?;
        self.stdin.flush()?;
        let line = self.read_line()?;
        let rest = line
            .strip_prefix("KEYMAPS ")
            .ok_or_else(|| anyhow::anyhow!("expected KEYMAPS line, got {line:?}"))?;
        Ok(rest.trim().parse()?)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "QUIT");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

/// Split `text` into streaming-style deltas of 1–3 whitespace-separated tokens,
/// preserving the spaces *between* tokens so the reconstructed stream equals
/// `text` exactly. The split boundaries vary by a rotating size so different
/// iterations exercise different delta groupings (mirroring real ASR jitter).
fn split_into_deltas(text: &str, seed: usize) -> Vec<String> {
    // Tokenise into (word, trailing-space?) preserving exact spacing. We keep
    // the space attached to the *preceding* word so concatenation is lossless.
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if ch == ' ' {
            tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }

    // Group tokens into deltas of 1..=3 tokens, rotating the group size by the
    // seed so iterations differ.
    let sizes = [1usize, 2, 3, 1, 2];
    let mut deltas: Vec<String> = Vec::new();
    let mut i = 0;
    let mut s = seed;
    while i < tokens.len() {
        let n = sizes[s % sizes.len()].max(1);
        s += 1;
        let end = (i + n).min(tokens.len());
        deltas.push(tokens[i..end].concat());
        i = end;
    }
    deltas
}

fn main() -> anyhow::Result<()> {
    // SAFETY GUARD: never run against the user's live session runtime dir.
    let xrd = std::env::var("XDG_RUNTIME_DIR").unwrap_or_default();
    if xrd.starts_with("/run/user/") {
        eprintln!(
            "vk_stress: refusing to run against real runtime dir {xrd:?}; \
             use an isolated XDG_RUNTIME_DIR under /tmp"
        );
        std::process::exit(3);
    }

    let mut recorder_path = String::from("vk_recorder");
    let mut iters = 100usize;
    let mut delay_ms = 4u64;
    let mut lang = String::from("ar");
    let mut verbose = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--recorder" => recorder_path = args.next().expect("--recorder needs a value"),
            "--iters" => iters = args.next().unwrap().parse().unwrap(),
            "--delay-ms" => delay_ms = args.next().unwrap().parse().unwrap(),
            "--lang" => lang = args.next().unwrap(),
            "--verbose" => verbose = true,
            other => panic!("unknown arg {other}"),
        }
    }

    // Mixed-script sentences: English words interleaved with non-Latin words,
    // dense and realistic. Spaces between English words are the prime
    // drop-victims in the bug.
    let sentence = match lang.as_str() {
        "ar" => {
            "the quick brown fox مرحبا بالعالم jumps over اهلا وسهلا the lazy dog \
                 صباح الخير and types مع السلامة fast every single time"
        }
        "ru" => {
            "the quick brown fox привет мир jumps over здравствуй the lazy dog \
                 доброе утро and types до свидания fast every single time"
        }
        // Dense multi-script: English + Arabic + Russian + Greek + CJK +
        // accented Latin + uppercase + shifted punctuation. >45 distinct
        // non-ASCII glyphs (exceeds the per-keymap AltGr capacity), so this
        // also exercises the over-capacity batch path — and must STILL never
        // drop a char.
        "mix" => {
            "The Quick BROWN مرحبا بالعالم اهلا صباح الخير السلامة привет \
                  мир здравствуй доброе утро свидания αβγδεζηθ Ωμω 日本語 中文 \
                  한국어 Größe café naïve résumé Ærø 100% & (mixed) EVERY Time?"
        }
        other => panic!("unknown lang {other}"),
    };

    eprintln!("vk_stress: lang={lang} iters={iters} delay_ms={delay_ms}");
    eprintln!("vk_stress: sentence = {sentence:?}");

    let mut rec = Recorder::spawn(&recorder_path)?;

    // ONE long-lived keyboard for the whole run, just like the daemon.
    // Creating it attaches a `zwp_virtual_keyboard_v1` to the seat, which is
    // what gives the seat its keyboard capability and lets the recorder acquire
    // the keyboard + receive focus.
    let mut kb = WaylandVkKeyboard::new(Duration::from_millis(delay_ms))?;

    // WARM-UP: drive one keystroke so the keymap is uploaded and the recorder
    // surface acquires keyboard focus, then settle and clear. Without this the
    // first measured iteration races focus establishment (a harness artefact,
    // not the bug under test). We retry until the recorder actually observes a
    // keystroke, so the real measurement starts from a known-focused state.
    {
        let mut focused = false;
        for _ in 0..50 {
            rec.reset()?;
            kb.type_text("x")?;
            std::thread::sleep(Duration::from_millis(20));
            let got = rec.dump()?;
            if got.contains('x') {
                focused = true;
                break;
            }
        }
        anyhow::ensure!(
            focused,
            "warm-up failed: recorder never received a keystroke (no focus?)"
        );
        rec.reset()?;
    }

    let mut exact = 0usize;
    let mut total_dropped_chars = 0usize;
    let mut drop_examples: Vec<String> = Vec::new();
    let mut first_keymaps = 0u32;
    let mut last_keymaps = 0u32;

    for iter in 0..iters {
        rec.reset()?;
        let before_km = rec.keymaps()?;
        if iter == 0 {
            first_keymaps = before_km;
        }

        let deltas = split_into_deltas(sentence, iter);
        for d in &deltas {
            kb.type_text(d)?;
        }

        let got = rec.dump()?;
        let after_km = rec.keymaps()?;
        last_keymaps = after_km;

        if got == sentence {
            exact += 1;
        } else {
            let dropped = char_diff(sentence, &got);
            total_dropped_chars += dropped.len();
            if drop_examples.len() < 8 {
                drop_examples.push(format!(
                    "iter {iter}: expected {} chars, got {} chars; re-uploads this iter={}; \
                     missing/garbled={:?}\n    EXP={sentence:?}\n    GOT={got:?}",
                    sentence.chars().count(),
                    got.chars().count(),
                    after_km.saturating_sub(before_km),
                    dropped,
                ));
            }
            if verbose {
                eprintln!("iter {iter}: MISMATCH got={got:?}");
            }
        }
    }

    let rate = (exact as f64) * 100.0 / (iters as f64);
    println!("==== vk_stress results (lang={lang}) ====");
    println!("iterations:            {iters}");
    println!("exact matches:         {exact}");
    println!("exact-match rate:      {rate:.1}%");
    println!("total dropped chars:   {total_dropped_chars}");
    println!(
        "keymap uploads (first iter total / final iter total): {first_keymaps} / {last_keymaps}"
    );
    if !drop_examples.is_empty() {
        println!("---- drop examples ----");
        for e in &drop_examples {
            println!("{e}");
        }
    }

    // Exit non-zero if not perfect, so the shell driver can gate on it.
    if exact != iters {
        std::process::exit(1);
    }
    Ok(())
}

/// Compute a simple character-level diff: the chars present in `expected` whose
/// multiset count exceeds that in `got` (i.e. the dropped/garbled characters).
fn char_diff(expected: &str, got: &str) -> Vec<char> {
    use std::collections::HashMap;
    let mut counts: HashMap<char, i32> = HashMap::new();
    for c in expected.chars() {
        *counts.entry(c).or_default() += 1;
    }
    for c in got.chars() {
        *counts.entry(c).or_default() -= 1;
    }
    let mut out: Vec<char> = Vec::new();
    for (c, n) in counts {
        for _ in 0..n.max(0) {
            out.push(c);
        }
    }
    out.sort_unstable();
    out
}
