use std::process;

use clap::{Args, Parser, Subcommand, ValueEnum};
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

use whisrs::history::HistoryEntry;
use whisrs::recording::{
    store, RecordingAction, RecordingMeta, RecordingOptions, RecordingPhase, RecordingState,
    RecordingStatus,
};
use whisrs::{
    encode_message, read_message, service::ServiceManager, socket_path, Command, Response,
    RestartOutcome, State,
};

const ASCII_BANNER: &str = concat!(
    "\n",
    "         __    _\n",
    "  _    _| |__ |_|___ _ __ ___\n",
    " \\ \\//\\ / '_ \\| / __| '__/ __|\n",
    "  \\  /\\ \\ | | | \\__ \\ |  \\__ \\\n",
    "   \\/  \\/|_| |_|_|___/_|  |___/\n",
    "\n",
    "  speak. type. done.\n",
    "\n",
    env!("CARGO_PKG_VERSION"),
);

// ANSI color codes.
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

#[derive(Parser)]
#[command(
    name = "whisrs",
    about = "Linux-first voice-to-text dictation tool",
    long_version = ASCII_BANNER,
)]
struct Cli {
    #[command(subcommand)]
    command: SubCmd,
}

#[derive(Subcommand)]
enum SubCmd {
    /// Interactive onboarding — pick a backend, set API key, test microphone
    Setup,
    /// Edit any part of ~/.config/whisrs/config.toml; restarts the daemon on save
    Config,
    /// Toggle dictation on/off (start, or stop and transcribe)
    Toggle {
        /// Override the transcription language for this session: an ISO 639-1
        /// code (e.g. `en`, `pl`), optionally with a region (`en-US`), or `auto`
        #[arg(short, long, value_parser = whisrs::validate_language_override)]
        language: Option<String>,
    },
    /// Cancel the current dictation and discard its audio
    Cancel,
    /// Query the dictation state (idle, recording, transcribing); for a
    /// recording see `whisrs recording status`
    Status,
    /// Show recent transcription history
    Log {
        /// Number of entries to show (default: 20)
        #[arg(short = 'n', long, default_value = "20")]
        limit: usize,
        /// Clear all history
        #[arg(long)]
        clear: bool,
    },
    /// Command mode: select text, speak an instruction, LLM rewrites it in place
    Command,
    /// Toggle a named custom LLM command (see [[llm_commands]] in config.toml):
    /// dictate, the LLM applies the configured instruction, result is typed
    /// at the cursor. Press again to stop recording, same as `toggle`.
    #[command(name = "llm-command")]
    LlmCommand {
        /// Name of the [[llm_commands]] entry to run.
        name: String,
    },
    /// Reprogram a named LLM command from the current selection: highlight the
    /// new instruction text, then run this — it's saved to config and applied
    /// live. Pairs with an entry's `set_hotkey`.
    #[command(name = "llm-command-set")]
    LlmCommandSet {
        /// Name of the [[llm_commands]] entry to reprogram.
        name: String,
    },
    /// Read the selected text aloud via TTS (press again to stop playback)
    #[command(alias = "read")]
    Speak,
    /// Restart the whisrs daemon (uses the systemd or OpenRC user service when present)
    Restart,
    /// Record your mic and the system audio, then turn them into a transcript
    #[command(subcommand)]
    Recording(RecordingCmd),
}

/// The transcription languages of a recording (`start`, `toggle`, `retry`).
#[derive(Args, Debug, Clone, Default, PartialEq, Eq)]
struct LanguageArgs {
    /// Language of both tracks: an ISO 639-1 code (e.g. `de`), optionally
    /// with a region (`en-US`), or `auto`
    #[arg(short, long, value_parser = whisrs::validate_language_override)]
    language: Option<String>,
    /// Language of your microphone track, over --language
    #[arg(long, value_name = "LANGUAGE", value_parser = whisrs::validate_language_override)]
    mic_language: Option<String>,
    /// Language of the system audio track, over --language
    #[arg(long, value_name = "LANGUAGE", value_parser = whisrs::validate_language_override)]
    system_language: Option<String>,
}

impl LanguageArgs {
    fn into_options(self, others: Option<String>) -> RecordingOptions {
        RecordingOptions {
            language: self.language,
            mic_language: self.mic_language,
            system_language: self.system_language,
            others,
        }
    }
}

#[derive(Subcommand)]
enum RecordingCmd {
    /// Start a recording (mic + system audio)
    Start {
        /// Recording title, used in the transcript and the folder name
        #[arg(short, long)]
        title: Option<String>,
        #[command(flatten)]
        languages: LanguageArgs,
        /// Name of the system audio track in the transcript (default: Others)
        #[arg(long, value_name = "NAME", value_parser = whisrs::recording::validate_others_label)]
        others: Option<String>,
    },
    /// Stop the recording and queue it for transcription
    Stop,
    /// Start a recording, or stop the one in progress (bind this to a hotkey)
    Toggle {
        /// Recording title when this starts a recording
        #[arg(short, long)]
        title: Option<String>,
        // The languages apply when this starts a recording.
        #[command(flatten)]
        languages: LanguageArgs,
        /// Name of the system audio track when this starts a recording
        /// (default: Others)
        #[arg(long, value_name = "NAME", value_parser = whisrs::recording::validate_others_label)]
        others: Option<String>,
    },
    /// Stop the recording and delete it
    Cancel,
    /// Show whether a recording is in progress or processing, and which
    /// recordings wait for the network
    Status,
    /// List recordings, newest first
    List {
        /// Print a JSON array instead of a table
        #[arg(long)]
        json: bool,
    },
    /// Print a recording's transcript
    Show {
        /// Recording id, a unique id prefix, or `last`
        id: String,
        /// Output format
        #[arg(short, long, value_enum, default_value_t = ShowFormat::Md)]
        format: ShowFormat,
    },
    /// Print a recording's folder
    Path {
        /// Recording id, a unique id prefix, or `last`
        id: String,
    },
    /// Copy a recording's transcript to the clipboard
    Copy {
        /// Recording id, a unique id prefix, or `last`
        id: String,
    },
    /// Transcribe a recording again (e.g. after a failure, or in another
    /// language: a language given here replaces the stored one)
    Retry {
        /// Recording id, a unique id prefix, or `last`
        id: String,
        #[command(flatten)]
        languages: LanguageArgs,
    },
}

/// Output format of `whisrs recording show`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ShowFormat {
    /// The rendered Markdown transcript (`<id>.md`)
    Md,
    /// The raw `recording.json`
    Json,
    /// Plain text, one line per utterance
    Txt,
}

/// Check if stdout is a TTY for color support.
fn is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Format a state for display with optional color.
fn format_state(state: State, use_color: bool) -> String {
    if !use_color {
        return format!("{state}");
    }

    match state {
        State::Idle => format!("{BOLD}idle{RESET}"),
        State::Recording => format!("{BOLD}{GREEN}recording{RESET}"),
        State::Transcribing => format!("{BOLD}{YELLOW}transcribing{RESET}"),
        State::Synthesizing => format!("{BOLD}{CYAN}synthesizing{RESET}"),
        State::Speaking => format!("{BOLD}{GREEN}speaking{RESET}"),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        SubCmd::Setup => {
            if let Err(e) = whisrs::config::setup::run_setup() {
                if is_tty() {
                    eprintln!("{RED}setup failed:{RESET} {e:#}");
                } else {
                    eprintln!("setup failed: {e:#}");
                }
                process::exit(1);
            }
        }
        SubCmd::Config => {
            if let Err(e) = whisrs::config::edit::run_config_menu() {
                if is_tty() {
                    eprintln!("{RED}config failed:{RESET} {e:#}");
                } else {
                    eprintln!("config failed: {e:#}");
                }
                process::exit(1);
            }
        }
        SubCmd::Toggle { language } => {
            send_command(Command::Toggle { language }).await?;
        }
        SubCmd::Cancel => {
            send_command(Command::Cancel).await?;
        }
        SubCmd::Status => {
            send_command(Command::Status).await?;
        }
        SubCmd::Log { limit, clear } => {
            if clear {
                send_command(Command::ClearHistory).await?;
            } else {
                send_command(Command::Log { limit }).await?;
            }
        }
        SubCmd::Command => {
            send_command(Command::CommandMode).await?;
        }
        SubCmd::LlmCommand { name } => {
            send_command(Command::LlmCommand { name }).await?;
        }
        SubCmd::LlmCommandSet { name } => {
            send_command(Command::SetLlmInstruction { name }).await?;
        }
        SubCmd::Speak => {
            send_command(Command::Speak).await?;
        }
        SubCmd::Restart => {
            cmd_restart()?;
        }
        SubCmd::Recording(cmd) => {
            cmd_recording(cmd).await?;
        }
    }

    Ok(())
}

/// Restart the whisrs daemon.
///
/// Delegates to whichever init system is managing the daemon (systemd or
/// OpenRC), and prints guidance when neither has a whisrs service installed.
/// We don't try to `pkill whisrsd` ourselves because that races with respawn
/// and silently breaks for users who launched the daemon under tmux/foot/etc.
fn cmd_restart() -> anyhow::Result<()> {
    let use_color = is_tty();
    let manager = ServiceManager::detect();

    let banner = format!("Restarting whisrs daemon ({})…", manager.name());
    if use_color {
        println!("{BOLD}{banner}{RESET}");
    } else {
        println!("{banner}");
    }

    match manager.restart() {
        RestartOutcome::Restarted => {
            if use_color {
                println!("{GREEN}Daemon restarted.{RESET}");
            } else {
                println!("Daemon restarted.");
            }
            Ok(())
        }
        RestartOutcome::Failed => {
            let hint = manager.restart_hint().unwrap_or("restart");
            if use_color {
                eprintln!("{RED}{hint} failed.{RESET}");
            } else {
                eprintln!("{hint} failed.");
            }
            process::exit(1);
        }
        RestartOutcome::NoService => {
            let detail = match manager.enable_hint() {
                Some(enable) => format!(
                    "No whisrs service installed for {}.\n\
                     \n\
                     Install it (run `whisrs setup` and accept the service step), then:\n\
                     \n\
                     \x20 {enable}\n\
                     \n\
                     Or restart the daemon manually:\n\
                     \n\
                     \x20 pkill whisrsd; sleep 0.2; whisrsd &",
                    manager.name()
                ),
                None => "No service manager detected (neither systemd nor OpenRC).\n\
                     \n\
                     Restart the daemon manually:\n\
                     \n\
                     \x20 pkill whisrsd; sleep 0.2; whisrsd &"
                    .to_string(),
            };
            if use_color {
                eprintln!("{YELLOW}{detail}{RESET}");
            } else {
                eprintln!("{detail}");
            }
            process::exit(1);
        }
    }
}

/// Connect to the daemon and send a command, printing the response.
async fn send_command(cmd: Command) -> anyhow::Result<()> {
    let use_color = is_tty();
    let response = request(cmd).await?;

    match response {
        Response::Ok { state } => {
            println!("{}", format_state(state, use_color));
        }
        Response::History { entries } => {
            if entries.is_empty() {
                println!("No transcription history.");
            } else {
                print_history(&entries, use_color);
            }
        }
        Response::Recording { recording } => {
            println!("{}", format_recording_status(&recording, use_color));
        }
        Response::Error { message } => exit_with_error(&message),
    }

    Ok(())
}

/// Print `error: <message>` to stderr and exit 1.
fn exit_with_error(message: &str) -> ! {
    if is_tty() {
        eprintln!("{RED}error:{RESET} {message}");
    } else {
        eprintln!("error: {message}");
    }
    process::exit(1);
}

/// Connect to the daemon, send a command and return its response.
/// Exits with a hint when the daemon is not running.
async fn request(cmd: Command) -> anyhow::Result<Response> {
    let path = socket_path();
    let use_color = is_tty();

    let stream = match UnixStream::connect(&path).await {
        Ok(s) => s,
        Err(_) => {
            // Name the command that actually exists on this machine — telling
            // an OpenRC user to run systemctl is a dead end.
            let service_hint = match ServiceManager::detect().enable_hint() {
                Some(enable) => format!(
                    "\n\
                     \n\
                     Or enable the service:\n\
                     \n\
                     \x20 {enable}"
                ),
                None => String::new(),
            };
            let msg = format!(
                "whisrsd is not running. Start it with:\n\
                 \n\
                 \x20 whisrsd &{service_hint}"
            );
            if use_color {
                eprintln!("{RED}{msg}{RESET}");
            } else {
                eprintln!("{msg}");
            }
            process::exit(1);
        }
    };

    let (mut reader, mut writer) = stream.into_split();

    // Send command.
    let encoded = encode_message(&cmd)?;
    writer.write_all(&encoded).await?;
    writer.shutdown().await?;

    // Read response.
    let response: Response = read_message(&mut reader).await?;
    Ok(response)
}

/// `whisrs recording <action>`. Recording control goes through the daemon;
/// `list`, `show` and `path` only read the recordings folder, so they work
/// with the daemon stopped.
async fn cmd_recording(cmd: RecordingCmd) -> anyhow::Result<()> {
    let use_color = is_tty();
    let none = RecordingOptions::default;
    let (action, title, id, options) = match cmd {
        RecordingCmd::Start {
            title,
            languages,
            others,
        } => (
            RecordingAction::Start,
            title,
            None,
            languages.into_options(others),
        ),
        RecordingCmd::Stop => (RecordingAction::Stop, None, None, none()),
        RecordingCmd::Toggle {
            title,
            languages,
            others,
        } => (
            RecordingAction::Toggle,
            title,
            None,
            languages.into_options(others),
        ),
        RecordingCmd::Cancel => (RecordingAction::Cancel, None, None, none()),
        RecordingCmd::Status => (RecordingAction::Status, None, None, none()),
        RecordingCmd::Copy { id } => (RecordingAction::Copy, None, Some(id), none()),
        RecordingCmd::Retry { id, languages } => (
            RecordingAction::Retry,
            None,
            Some(id),
            languages.into_options(None),
        ),
        RecordingCmd::List { json } => return recording_list(json),
        RecordingCmd::Show { id, format } => return recording_show(&id, format),
        RecordingCmd::Path { id } => {
            let dir = resolve_or_exit(&id);
            println!("{}", dir.display());
            return Ok(());
        }
    };

    let response = request(Command::Recording {
        action,
        title,
        id,
        options,
    })
    .await?;
    match response {
        Response::Recording { recording } => {
            match action {
                RecordingAction::Copy => println!("Transcript copied to the clipboard."),
                RecordingAction::Retry => println!("Queued for transcription."),
                _ => {}
            }
            println!("{}", format_recording_status(&recording, use_color));
        }
        Response::Error { message } => exit_with_error(&message),
        other => exit_with_error(&format!("unexpected daemon response: {other:?}")),
    }
    Ok(())
}

/// The recordings root: `[recording] dir` from config.toml, else the default.
///
/// Reads only that one key, so a config the daemon would reject for an
/// unrelated reason still lets you browse recordings.
fn recordings_root() -> std::path::PathBuf {
    let dir = std::fs::read_to_string(whisrs::config_path())
        .ok()
        .and_then(|text| recording_dir_from_toml(&text));
    store::recordings_dir_from(dir.as_deref())
}

/// `[recording] dir` from config.toml text, if set.
fn recording_dir_from_toml(text: &str) -> Option<String> {
    let table: toml::Table = text.parse().ok()?;
    table
        .get("recording")?
        .get("dir")?
        .as_str()
        .map(str::to_string)
}

/// Resolve `<id|last>` to a recording folder, or print the error and exit 1.
fn resolve_or_exit(query: &str) -> std::path::PathBuf {
    match store::resolve(&recordings_root(), query) {
        Ok(dir) => dir,
        Err(e) => exit_with_error(&format!("{e:#}")),
    }
}

/// `whisrs recording list`.
fn recording_list(json: bool) -> anyhow::Result<()> {
    let root = recordings_root();
    let recordings = match store::list(&root) {
        Ok(m) => m,
        Err(e) => exit_with_error(&format!("{e:#}")),
    };

    if json {
        let items: Vec<serde_json::Value> = recordings
            .iter()
            .map(|(dir, meta)| recording_list_json(dir, meta))
            .collect();
        println!("{}", serde_json::to_string_pretty(&items)?);
        return Ok(());
    }

    if recordings.is_empty() {
        println!("No recordings in {}.", root.display());
        return Ok(());
    }
    let id_width = recordings
        .iter()
        .map(|(_, m)| m.id.len())
        .max()
        .unwrap_or(0)
        .max("ID".len());
    println!(
        "{:<id_width$}  {:<16}  {:>8}  {:<10}  TITLE",
        "ID", "STARTED", "DURATION", "STATUS"
    );
    for (_, meta) in &recordings {
        println!("{}", format_recording_row(meta, id_width));
    }
    Ok(())
}

/// One row of the `recording list` table.
fn format_recording_row(meta: &RecordingMeta, id_width: usize) -> String {
    let duration = recording_duration_secs(meta)
        .map(format_duration)
        .unwrap_or_else(|| "-".to_string());
    format!(
        "{:<id_width$}  {:<16}  {:>8}  {:<10}  {}",
        meta.id,
        meta.started_at.format("%Y-%m-%d %H:%M"),
        duration,
        recording_state_name(meta.status),
        meta.title.as_deref().unwrap_or(""),
    )
}

/// One element of `recording list --json`: the metadata without segments.
fn recording_list_json(dir: &std::path::Path, meta: &RecordingMeta) -> serde_json::Value {
    serde_json::json!({
        "id": meta.id,
        "title": meta.title,
        "status": meta.status,
        "started_at": meta.started_at,
        "ended_at": meta.ended_at,
        "duration_secs": recording_duration_secs(meta),
        "dir": dir,
        "error": meta.error,
    })
}

/// Recorded length in whole seconds, or `None` while still recording.
fn recording_duration_secs(meta: &RecordingMeta) -> Option<i64> {
    let ended = meta.ended_at?;
    Some((ended - meta.started_at).num_seconds().max(0))
}

/// `3725` -> `1:02:05`, `125` -> `2:05`.
fn format_duration(secs: i64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

fn recording_state_name(state: RecordingState) -> &'static str {
    match state {
        RecordingState::Recording => "recording",
        RecordingState::Processing => "processing",
        RecordingState::Done => "done",
        RecordingState::Failed => "failed",
        RecordingState::Waiting => "waiting",
    }
}

/// `whisrs recording show`.
fn recording_show(query: &str, format: ShowFormat) -> anyhow::Result<()> {
    let dir = resolve_or_exit(query);
    match format {
        ShowFormat::Json => {
            let path = dir.join("recording.json");
            match std::fs::read_to_string(&path) {
                Ok(text) => print!("{text}"),
                Err(e) => exit_with_error(&format!("failed to read {}: {e}", path.display())),
            }
        }
        ShowFormat::Md | ShowFormat::Txt => {
            let meta = match store::read_meta(&dir) {
                Ok(meta) => meta,
                Err(e) => exit_with_error(&format!("{e:#}")),
            };
            if format == ShowFormat::Txt {
                if meta.status != RecordingState::Done {
                    exit_with_error(&not_ready_message(&meta));
                }
                print!("{}", whisrs::recording::render::render_text(&meta));
            } else {
                let path = store::transcript_path(&dir, &meta.id);
                match std::fs::read_to_string(&path) {
                    Ok(text) => print!("{text}"),
                    Err(e)
                        if e.kind() == std::io::ErrorKind::NotFound
                            && meta.status != RecordingState::Done =>
                    {
                        exit_with_error(&not_ready_message(&meta))
                    }
                    Err(e) => exit_with_error(&format!("failed to read {}: {e}", path.display())),
                }
            }
        }
    }
    Ok(())
}

/// Why `show` has no transcript to print for a recording that is not done.
fn not_ready_message(meta: &RecordingMeta) -> String {
    match meta.status {
        RecordingState::Failed => format!(
            "recording {} failed to process: {}. Run `whisrs recording retry {}` to try again.",
            meta.id,
            meta.error.as_deref().unwrap_or("unknown error"),
            meta.id
        ),
        RecordingState::Waiting => {
            let until = meta
                .network_deadline()
                .map(|d| format!(" until {}", d.format("%Y-%m-%d %H:%M")))
                .unwrap_or_default();
            format!(
                "recording {} is waiting for the network: {}. whisrs tries again every {} \
                 minutes{until}; `whisrs recording retry {}` tries now.",
                meta.id,
                meta.error.as_deref().unwrap_or("no network"),
                whisrs::recording::NETWORK_RETRY_EVERY.as_secs() / 60,
                meta.id
            )
        }
        state => format!(
            "recording {} has no transcript yet (status: {}).",
            meta.id,
            recording_state_name(state)
        ),
    }
}

/// One-line recording status, e.g.
/// `Recording since 14:05 (Weekly sync)  id 2026-10-02_1405_weekly-sync`.
fn format_recording_status(status: &RecordingStatus, use_color: bool) -> String {
    let (word, color) = match status.phase {
        RecordingPhase::Idle => ("Idle", ""),
        RecordingPhase::Recording => ("Recording", GREEN),
        RecordingPhase::Processing => ("Processing", YELLOW),
    };
    let mut line = if use_color {
        format!("{BOLD}{color}{word}{RESET}")
    } else {
        word.to_string()
    };
    if status.phase == RecordingPhase::Recording {
        if let Some(started) = status.started_at {
            line.push_str(&format!(" since {}", started.format("%H:%M")));
        }
    }
    if status.phase != RecordingPhase::Idle {
        if let Some(title) = status.title.as_deref().filter(|t| !t.is_empty()) {
            line.push_str(&format!(" ({title})"));
        }
        if let Some(id) = &status.id {
            line.push_str(&format!("  id {id}"));
        }
    }
    if !status.queued.is_empty() {
        line.push_str(&format!("  ({} queued)", status.queued.len()));
    }
    if !status.waiting.is_empty() {
        line.push_str(&format!(
            "  ({} waiting for the network: {})",
            status.waiting.len(),
            status.waiting.join(", ")
        ));
    }
    line
}

/// Display transcription history entries.
fn print_history(entries: &[HistoryEntry], use_color: bool) {
    let dim = if use_color { "\x1b[2m" } else { "" };

    for entry in entries {
        let ts = entry.timestamp.format("%Y-%m-%d %H:%M:%S");
        let duration = format!("{:.1}s", entry.duration_secs);

        if use_color {
            println!(
                "{dim}{ts}{RESET}  {dim}[{backend} | {lang} | {dur}]{RESET}",
                backend = entry.backend,
                lang = entry.language,
                dur = duration,
            );
        } else {
            println!(
                "{ts}  [{backend} | {lang} | {dur}]",
                backend = entry.backend,
                lang = entry.language,
                dur = duration,
            );
        }
        println!("  {}", entry.text);
        println!();
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    fn parse_recording(args: &[&str]) -> RecordingCmd {
        let mut argv = vec!["whisrs", "recording"];
        argv.extend_from_slice(args);
        match Cli::try_parse_from(argv).expect("parse").command {
            SubCmd::Recording(cmd) => cmd,
            _ => panic!("not a recording command"),
        }
    }

    #[test]
    fn recording_start_parses_title() {
        assert!(matches!(
            parse_recording(&["start"]),
            RecordingCmd::Start { title: None, languages, others: None }
                if languages == LanguageArgs::default()
        ));
        for flag in ["-t", "--title"] {
            match parse_recording(&["start", flag, "Weekly sync"]) {
                RecordingCmd::Start { title, .. } => {
                    assert_eq!(title.as_deref(), Some("Weekly sync"))
                }
                _ => panic!("expected start"),
            }
        }
    }

    #[test]
    fn recording_toggle_parses_title() {
        assert!(matches!(
            parse_recording(&["toggle"]),
            RecordingCmd::Toggle { title: None, languages, others: None }
                if languages == LanguageArgs::default()
        ));
        match parse_recording(&["toggle", "-t", "Weekly sync"]) {
            RecordingCmd::Toggle { title, .. } => {
                assert_eq!(title.as_deref(), Some("Weekly sync"))
            }
            _ => panic!("expected toggle"),
        }
    }

    /// The language practice example from the docs, on `start` and `toggle`.
    #[test]
    fn recording_start_and_toggle_parse_languages_and_others() {
        let want = RecordingOptions {
            language: None,
            mic_language: Some("de".into()),
            system_language: Some("auto".into()),
            others: Some("Video".into()),
        };
        for action in ["start", "toggle"] {
            let cmd = parse_recording(&[
                action,
                "-t",
                "Lesson 50",
                "--mic-language",
                "DE",
                "--system-language",
                "auto",
                "--others",
                " Video ",
            ]);
            let options = match cmd {
                RecordingCmd::Start {
                    languages, others, ..
                }
                | RecordingCmd::Toggle {
                    languages, others, ..
                } => languages.into_options(others),
                _ => panic!("expected {action}"),
            };
            assert_eq!(options, want, "{action}: normalized as the daemon would");
        }
        for flag in ["-l", "--language"] {
            match parse_recording(&["start", flag, "de"]) {
                RecordingCmd::Start { languages, .. } => {
                    assert_eq!(languages.language.as_deref(), Some("de"))
                }
                _ => panic!("expected start"),
            }
        }
        for bad in [
            &["start", "-l", "english"][..],
            &["toggle", "--mic-language", "1"],
            &["start", "--system-language", ""],
            &["start", "--others", "  "],
        ] {
            let mut argv = vec!["whisrs", "recording"];
            argv.extend_from_slice(bad);
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "{bad:?} must be refused"
            );
        }
    }

    /// `retry` takes the three languages, and no title or others label.
    #[test]
    fn recording_retry_parses_languages() {
        match parse_recording(&["retry", "last", "-l", "de", "--system-language", "auto"]) {
            RecordingCmd::Retry { id, languages } => {
                assert_eq!(id, "last");
                assert_eq!(
                    languages.into_options(None),
                    RecordingOptions {
                        language: Some("de".into()),
                        system_language: Some("auto".into()),
                        ..Default::default()
                    }
                );
            }
            _ => panic!("expected retry"),
        }
        for extra in ["--others", "--title"] {
            let argv = ["whisrs", "recording", "retry", "last", extra, "x"];
            assert!(Cli::try_parse_from(argv).is_err(), "retry takes no {extra}");
        }
    }

    #[test]
    fn recording_plain_actions_parse() {
        assert!(matches!(parse_recording(&["stop"]), RecordingCmd::Stop));
        assert!(matches!(parse_recording(&["cancel"]), RecordingCmd::Cancel));
        assert!(matches!(parse_recording(&["status"]), RecordingCmd::Status));
        assert!(matches!(
            parse_recording(&["list"]),
            RecordingCmd::List { json: false }
        ));
        assert!(matches!(
            parse_recording(&["list", "--json"]),
            RecordingCmd::List { json: true }
        ));
    }

    #[test]
    fn recording_id_actions_parse() {
        match parse_recording(&["path", "last"]) {
            RecordingCmd::Path { id } => assert_eq!(id, "last"),
            _ => panic!("expected path"),
        }
        match parse_recording(&["copy", "2026-10-02"]) {
            RecordingCmd::Copy { id } => assert_eq!(id, "2026-10-02"),
            _ => panic!("expected copy"),
        }
        match parse_recording(&["retry", "last"]) {
            RecordingCmd::Retry { id, languages } => {
                assert_eq!(id, "last");
                assert_eq!(languages, LanguageArgs::default());
            }
            _ => panic!("expected retry"),
        }
        for action in ["show", "path", "copy", "retry"] {
            let argv = ["whisrs", "recording", action];
            assert!(Cli::try_parse_from(argv).is_err(), "{action} needs an id");
        }
    }

    #[test]
    fn recording_show_format_defaults_to_md() {
        match parse_recording(&["show", "last"]) {
            RecordingCmd::Show { id, format } => {
                assert_eq!(id, "last");
                assert_eq!(format, ShowFormat::Md);
            }
            _ => panic!("expected show"),
        }
        for (arg, want) in [
            ("md", ShowFormat::Md),
            ("json", ShowFormat::Json),
            ("txt", ShowFormat::Txt),
        ] {
            match parse_recording(&["show", "last", "--format", arg]) {
                RecordingCmd::Show { format, .. } => assert_eq!(format, want),
                _ => panic!("expected show"),
            }
        }
        match parse_recording(&["show", "x", "-f", "json"]) {
            RecordingCmd::Show { format, .. } => assert_eq!(format, ShowFormat::Json),
            _ => panic!("expected show"),
        }
        assert!(
            Cli::try_parse_from(["whisrs", "recording", "show", "x", "--format", "pdf"]).is_err()
        );
    }

    #[test]
    fn recording_requires_an_action() {
        assert!(Cli::try_parse_from(["whisrs", "recording"]).is_err());
        assert!(Cli::try_parse_from(["whisrs", "recording", "bogus"]).is_err());
    }

    fn at(h: u32, m: u32) -> chrono::DateTime<chrono::Local> {
        use chrono::TimeZone;
        chrono::Local
            .with_ymd_and_hms(2026, 10, 2, h, m, 0)
            .single()
            .unwrap()
    }

    #[test]
    fn recording_status_idle() {
        assert_eq!(
            format_recording_status(&RecordingStatus::default(), false),
            "Idle"
        );
    }

    #[test]
    fn recording_status_in_progress() {
        let status = RecordingStatus {
            phase: RecordingPhase::Recording,
            id: Some("2026-10-02_1405_weekly-sync".into()),
            title: Some("Weekly sync".into()),
            started_at: Some(at(14, 5)),
            ..Default::default()
        };
        assert_eq!(
            format_recording_status(&status, false),
            "Recording since 14:05 (Weekly sync)  id 2026-10-02_1405_weekly-sync"
        );

        let untitled = RecordingStatus {
            title: None,
            id: Some("2026-10-02_1405".into()),
            ..status
        };
        assert_eq!(
            format_recording_status(&untitled, false),
            "Recording since 14:05  id 2026-10-02_1405"
        );
    }

    #[test]
    fn recording_status_processing_with_queue() {
        let status = RecordingStatus {
            phase: RecordingPhase::Processing,
            id: Some("a".into()),
            started_at: Some(at(9, 0)),
            queued: vec!["b".into(), "c".into()],
            ..Default::default()
        };
        assert_eq!(
            format_recording_status(&status, false),
            "Processing  id a  (2 queued)"
        );
    }

    #[test]
    fn recording_status_color_wraps_phase_only() {
        let status = RecordingStatus {
            phase: RecordingPhase::Recording,
            started_at: Some(at(8, 30)),
            ..Default::default()
        };
        assert_eq!(
            format_recording_status(&status, true),
            format!("{BOLD}{GREEN}Recording{RESET} since 08:30")
        );
    }

    #[test]
    fn duration_formatting() {
        assert_eq!(format_duration(0), "0:00");
        assert_eq!(format_duration(125), "2:05");
        assert_eq!(format_duration(3725), "1:02:05");
    }

    #[test]
    fn recording_row_and_json() {
        let mut meta = RecordingMeta::new(
            "2026-10-02_1405_sync".into(),
            Some("Sync".into()),
            at(14, 5),
        );
        assert_eq!(recording_duration_secs(&meta), None);
        assert!(format_recording_row(&meta, 20).contains("       -  recording   Sync"));

        meta.ended_at = Some(at(15, 7));
        meta.status = RecordingState::Done;
        assert_eq!(recording_duration_secs(&meta), Some(62 * 60));
        assert_eq!(
            format_recording_row(&meta, 20),
            "2026-10-02_1405_sync  2026-10-02 14:05   1:02:00  done        Sync"
        );

        let json = recording_list_json(std::path::Path::new("/m/x"), &meta);
        assert_eq!(json["id"], "2026-10-02_1405_sync");
        assert_eq!(json["status"], "done");
        assert_eq!(json["duration_secs"], 3720);
        assert_eq!(json["dir"], "/m/x");
        assert!(json.get("segments").is_none());
    }

    #[test]
    fn not_ready_message_names_status_and_retry() {
        let mut meta = RecordingMeta::new("m1".into(), None, at(10, 0));
        assert_eq!(
            not_ready_message(&meta),
            "recording m1 has no transcript yet (status: recording)."
        );
        meta.status = RecordingState::Failed;
        meta.error = Some("backend down".into());
        assert!(not_ready_message(&meta).contains("backend down"));
        assert!(not_ready_message(&meta).contains("whisrs recording retry m1"));

        meta.status = RecordingState::Waiting;
        meta.ended_at = Some(at(11, 0));
        meta.error = Some("could not reach the transcription service".into());
        // The window runs from the first failed try (11:01), not the stop.
        meta.network_wait = Some(whisrs::recording::NetworkWait {
            since: at(11, 1),
            attempts: 0,
            last_attempt_at: at(11, 1),
        });
        assert_eq!(
            not_ready_message(&meta),
            "recording m1 is waiting for the network: could not reach the transcription \
             service. whisrs tries again every 10 minutes until 2026-10-03 11:01; \
             `whisrs recording retry m1` tries now."
        );
        assert!(format_recording_row(&meta, 2).contains("  waiting     "));
        let json = recording_list_json(std::path::Path::new("/m/m1"), &meta);
        assert_eq!(json["status"], "waiting");
        assert_eq!(json["error"], "could not reach the transcription service");
    }

    #[test]
    fn recording_status_lists_the_ones_waiting_for_the_network() {
        let status = RecordingStatus {
            waiting: vec!["a".into(), "b".into()],
            ..Default::default()
        };
        assert_eq!(
            format_recording_status(&status, false),
            "Idle  (2 waiting for the network: a, b)"
        );
    }

    #[test]
    fn recording_dir_read_from_config_toml() {
        assert_eq!(
            recording_dir_from_toml("[recording]\ndir = \"~/notes/recordings\"\n").as_deref(),
            Some("~/notes/recordings")
        );
        assert_eq!(
            recording_dir_from_toml("[general]\nbackend = \"groq\"\n"),
            None
        );
        assert_eq!(recording_dir_from_toml("not toml ["), None);
    }

    /// Issue #84: `llm-command` and `llm-command-set` existed in the clap CLI
    /// (`SubCmd` above) but were never added to the man page's SUBCOMMANDS
    /// section, and nothing caught it. Rather than hand-maintain a mirror
    /// list of subcommand names (which is exactly what went stale), this
    /// walks the real `Cli` derive so a future subcommand that isn't
    /// documented fails automatically.
    #[test]
    fn man_page_documents_all_cli_subcommands() {
        let page_path = format!("{}/contrib/whisrs.1", env!("CARGO_MANIFEST_DIR"));
        let page = std::fs::read_to_string(&page_path)
            .unwrap_or_else(|e| panic!("failed to read {page_path}: {e}"));

        let section = subcommands_section(&page);
        let headers = entry_headers(section);

        let cli = Cli::command();
        for sub in cli.get_subcommands() {
            let name = sub.get_name();
            assert!(
                headers.iter().any(|h| h == name),
                "contrib/whisrs.1: SUBCOMMANDS section is missing an entry header for \
                 `{name}` (expected a `.TP` block headed by `.B {name}` or `\\fB{name}\\fR`); \
                 found headers: {headers:?}"
            );
        }
    }

    /// Slice out the `.SH SUBCOMMANDS` section body, up to (but not
    /// including) the next `.SH `. Man pages repeat subcommand names as
    /// prose elsewhere (DESCRIPTION references `.B setup`, the `speak` entry
    /// mentions `.B cancel`, EXAMPLES shows `whisrs setup`, ...), so a
    /// whole-page search can't tell a real entry from a passing mention.
    /// Scoping to this section is necessary but not sufficient on its own —
    /// see `entry_headers` for the rest.
    fn subcommands_section(page: &str) -> &str {
        const HEADER: &str = ".SH SUBCOMMANDS";
        let start = page
            .find(HEADER)
            .expect("contrib/whisrs.1: missing .SH SUBCOMMANDS section");
        let rest = &page[start + HEADER.len()..];
        let end = rest.find("\n.SH ").unwrap_or(rest.len());
        &rest[..end]
    }

    /// Collect the subcommand names introduced by real `.TP` entry headers
    /// in `section` — i.e. the line immediately following a `.TP` macro,
    /// when that line is itself a `.B`/`.BR` macro or an inline `\fB...\fR`
    /// bold run. This deliberately ignores `.B name` / `\fBname\fR` used in
    /// body text (not directly under `.TP`), which is what let `.B setup`
    /// (referenced from the `config` entry) and `.B cancel` (referenced from
    /// the `speak` entry) mask a deleted header in a plain substring search.
    ///
    /// Escaped groff hyphens (`\-`) are normalized to plain `-` first so
    /// hyphenated subcommand names (`llm-command`, `llm-command-set`) match
    /// regardless of whether the man page escapes them.
    fn entry_headers(section: &str) -> Vec<String> {
        let normalized = section.replace("\\-", "-");
        let lines: Vec<&str> = normalized.lines().collect();

        let mut headers = Vec::new();
        for i in 0..lines.len() {
            if lines[i].trim() != ".TP" {
                continue;
            }
            let Some(header) = lines.get(i + 1).map(|l| l.trim()) else {
                continue;
            };

            if let Some(rest) = header.strip_prefix(".BR ") {
                if let Some(name) = rest.split_whitespace().next() {
                    headers.push(name.to_string());
                }
            } else if let Some(rest) = header.strip_prefix(".B ") {
                headers.push(rest.trim().to_string());
            } else if let Some(rest) = header.strip_prefix("\\fB") {
                if let Some(end) = rest.find("\\fR") {
                    headers.push(rest[..end].to_string());
                }
            }
        }
        headers
    }
}
