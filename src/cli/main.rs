use std::process;

use clap::{Parser, Subcommand, ValueEnum};
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

use whisrs::history::HistoryEntry;
use whisrs::meeting::{
    store, MeetingAction, MeetingMeta, MeetingPhase, MeetingState, MeetingStatus,
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
    /// Toggle recording on/off (start dictation or stop and transcribe)
    Toggle {
        /// Override the transcription language for this session: an ISO 639-1
        /// code (e.g. `en`, `pl`), optionally with a region (`en-US`), or `auto`
        #[arg(short, long, value_parser = whisrs::validate_language_override)]
        language: Option<String>,
    },
    /// Cancel the current recording and discard audio
    Cancel,
    /// Query the daemon state (idle, recording, transcribing)
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
    /// Record a meeting (mic + system audio) and turn it into a transcript
    #[command(subcommand)]
    Meeting(MeetingCmd),
}

#[derive(Subcommand)]
enum MeetingCmd {
    /// Start recording a meeting
    Start {
        /// Meeting title, used in the transcript and the folder name
        #[arg(short, long)]
        title: Option<String>,
    },
    /// Stop recording and queue the meeting for transcription
    Stop,
    /// Start a meeting, or stop the one recording (bind this to a hotkey)
    Toggle {
        /// Meeting title when this starts a meeting
        #[arg(short, long)]
        title: Option<String>,
    },
    /// Stop recording and delete the meeting
    Cancel,
    /// Show whether a meeting is recording or processing
    Status,
    /// List recorded meetings, newest first
    List {
        /// Print a JSON array instead of a table
        #[arg(long)]
        json: bool,
    },
    /// Print a meeting transcript
    Show {
        /// Meeting id, a unique id prefix, or `last`
        id: String,
        /// Output format
        #[arg(short, long, value_enum, default_value_t = ShowFormat::Md)]
        format: ShowFormat,
    },
    /// Print a meeting's folder
    Path {
        /// Meeting id, a unique id prefix, or `last`
        id: String,
    },
    /// Copy a meeting transcript to the clipboard
    Copy {
        /// Meeting id, a unique id prefix, or `last`
        id: String,
    },
    /// Transcribe a meeting again (e.g. after a failure)
    Retry {
        /// Meeting id, a unique id prefix, or `last`
        id: String,
    },
}

/// Output format of `whisrs meeting show`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ShowFormat {
    /// The rendered Markdown transcript (`<id>.md`)
    Md,
    /// The raw `meeting.json`
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
        SubCmd::Meeting(cmd) => {
            cmd_meeting(cmd).await?;
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
        Response::Meeting { meeting } => {
            println!("{}", format_meeting_status(&meeting, use_color));
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

/// `whisrs meeting <action>`. Recording control goes through the daemon;
/// `list`, `show` and `path` only read the meetings folder, so they work
/// with the daemon stopped.
async fn cmd_meeting(cmd: MeetingCmd) -> anyhow::Result<()> {
    let use_color = is_tty();
    let (action, title, id) = match cmd {
        MeetingCmd::Start { title } => (MeetingAction::Start, title, None),
        MeetingCmd::Stop => (MeetingAction::Stop, None, None),
        MeetingCmd::Toggle { title } => (MeetingAction::Toggle, title, None),
        MeetingCmd::Cancel => (MeetingAction::Cancel, None, None),
        MeetingCmd::Status => (MeetingAction::Status, None, None),
        MeetingCmd::Copy { id } => (MeetingAction::Copy, None, Some(id)),
        MeetingCmd::Retry { id } => (MeetingAction::Retry, None, Some(id)),
        MeetingCmd::List { json } => return meeting_list(json),
        MeetingCmd::Show { id, format } => return meeting_show(&id, format),
        MeetingCmd::Path { id } => {
            let dir = resolve_or_exit(&id);
            println!("{}", dir.display());
            return Ok(());
        }
    };

    let response = request(Command::Meeting { action, title, id }).await?;
    match response {
        Response::Meeting { meeting } => {
            match action {
                MeetingAction::Copy => println!("Transcript copied to the clipboard."),
                MeetingAction::Retry => println!("Queued for transcription."),
                _ => {}
            }
            println!("{}", format_meeting_status(&meeting, use_color));
        }
        Response::Error { message } => exit_with_error(&message),
        other => exit_with_error(&format!("unexpected daemon response: {other:?}")),
    }
    Ok(())
}

/// The meetings root: `[meeting] dir` from config.toml, else the default.
///
/// Reads only that one key, so a config the daemon would reject for an
/// unrelated reason still lets you browse meetings.
fn meetings_root() -> std::path::PathBuf {
    let dir = std::fs::read_to_string(whisrs::config_path())
        .ok()
        .and_then(|text| meeting_dir_from_toml(&text));
    store::meetings_dir_from(dir.as_deref())
}

/// `[meeting] dir` from config.toml text, if set.
fn meeting_dir_from_toml(text: &str) -> Option<String> {
    let table: toml::Table = text.parse().ok()?;
    table
        .get("meeting")?
        .get("dir")?
        .as_str()
        .map(str::to_string)
}

/// Resolve `<id|last>` to a meeting folder, or print the error and exit 1.
fn resolve_or_exit(query: &str) -> std::path::PathBuf {
    match store::resolve(&meetings_root(), query) {
        Ok(dir) => dir,
        Err(e) => exit_with_error(&format!("{e:#}")),
    }
}

/// `whisrs meeting list`.
fn meeting_list(json: bool) -> anyhow::Result<()> {
    let root = meetings_root();
    let meetings = match store::list(&root) {
        Ok(m) => m,
        Err(e) => exit_with_error(&format!("{e:#}")),
    };

    if json {
        let items: Vec<serde_json::Value> = meetings
            .iter()
            .map(|(dir, meta)| meeting_list_json(dir, meta))
            .collect();
        println!("{}", serde_json::to_string_pretty(&items)?);
        return Ok(());
    }

    if meetings.is_empty() {
        println!("No meetings in {}.", root.display());
        return Ok(());
    }
    let id_width = meetings
        .iter()
        .map(|(_, m)| m.id.len())
        .max()
        .unwrap_or(0)
        .max("ID".len());
    println!(
        "{:<id_width$}  {:<16}  {:>8}  {:<10}  TITLE",
        "ID", "STARTED", "DURATION", "STATUS"
    );
    for (_, meta) in &meetings {
        println!("{}", format_meeting_row(meta, id_width));
    }
    Ok(())
}

/// One row of the `meeting list` table.
fn format_meeting_row(meta: &MeetingMeta, id_width: usize) -> String {
    let duration = meeting_duration_secs(meta)
        .map(format_duration)
        .unwrap_or_else(|| "-".to_string());
    format!(
        "{:<id_width$}  {:<16}  {:>8}  {:<10}  {}",
        meta.id,
        meta.started_at.format("%Y-%m-%d %H:%M"),
        duration,
        meeting_state_name(meta.status),
        meta.title.as_deref().unwrap_or(""),
    )
}

/// One element of `meeting list --json`: the metadata without segments.
fn meeting_list_json(dir: &std::path::Path, meta: &MeetingMeta) -> serde_json::Value {
    serde_json::json!({
        "id": meta.id,
        "title": meta.title,
        "status": meta.status,
        "started_at": meta.started_at,
        "ended_at": meta.ended_at,
        "duration_secs": meeting_duration_secs(meta),
        "dir": dir,
    })
}

/// Recorded length in whole seconds, or `None` while still recording.
fn meeting_duration_secs(meta: &MeetingMeta) -> Option<i64> {
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

fn meeting_state_name(state: MeetingState) -> &'static str {
    match state {
        MeetingState::Recording => "recording",
        MeetingState::Processing => "processing",
        MeetingState::Done => "done",
        MeetingState::Failed => "failed",
    }
}

/// `whisrs meeting show`.
fn meeting_show(query: &str, format: ShowFormat) -> anyhow::Result<()> {
    let dir = resolve_or_exit(query);
    match format {
        ShowFormat::Json => {
            let path = dir.join("meeting.json");
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
                if meta.status != MeetingState::Done {
                    exit_with_error(&not_ready_message(&meta));
                }
                print!("{}", whisrs::meeting::render::render_text(&meta));
            } else {
                let path = store::transcript_path(&dir, &meta.id);
                match std::fs::read_to_string(&path) {
                    Ok(text) => print!("{text}"),
                    Err(e)
                        if e.kind() == std::io::ErrorKind::NotFound
                            && meta.status != MeetingState::Done =>
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

/// Why `show` has no transcript to print for a meeting that is not done.
fn not_ready_message(meta: &MeetingMeta) -> String {
    match meta.status {
        MeetingState::Failed => format!(
            "meeting {} failed to process: {}. Run `whisrs meeting retry {}` to try again.",
            meta.id,
            meta.error.as_deref().unwrap_or("unknown error"),
            meta.id
        ),
        state => format!(
            "meeting {} has no transcript yet (status: {}).",
            meta.id,
            meeting_state_name(state)
        ),
    }
}

/// One-line meeting status, e.g.
/// `Recording since 14:05 (Weekly sync)  id 2026-10-02_1405_weekly-sync`.
fn format_meeting_status(status: &MeetingStatus, use_color: bool) -> String {
    let (word, color) = match status.phase {
        MeetingPhase::Idle => ("Idle", ""),
        MeetingPhase::Recording => ("Recording", GREEN),
        MeetingPhase::Processing => ("Processing", YELLOW),
    };
    let mut line = if use_color {
        format!("{BOLD}{color}{word}{RESET}")
    } else {
        word.to_string()
    };
    if status.phase == MeetingPhase::Recording {
        if let Some(started) = status.started_at {
            line.push_str(&format!(" since {}", started.format("%H:%M")));
        }
    }
    if status.phase != MeetingPhase::Idle {
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

    fn parse_meeting(args: &[&str]) -> MeetingCmd {
        let mut argv = vec!["whisrs", "meeting"];
        argv.extend_from_slice(args);
        match Cli::try_parse_from(argv).expect("parse").command {
            SubCmd::Meeting(cmd) => cmd,
            _ => panic!("not a meeting command"),
        }
    }

    #[test]
    fn meeting_start_parses_title() {
        assert!(matches!(
            parse_meeting(&["start"]),
            MeetingCmd::Start { title: None }
        ));
        for flag in ["-t", "--title"] {
            match parse_meeting(&["start", flag, "Weekly sync"]) {
                MeetingCmd::Start { title } => assert_eq!(title.as_deref(), Some("Weekly sync")),
                _ => panic!("expected start"),
            }
        }
    }

    #[test]
    fn meeting_toggle_parses_title() {
        assert!(matches!(
            parse_meeting(&["toggle"]),
            MeetingCmd::Toggle { title: None }
        ));
        match parse_meeting(&["toggle", "-t", "Weekly sync"]) {
            MeetingCmd::Toggle { title } => assert_eq!(title.as_deref(), Some("Weekly sync")),
            _ => panic!("expected toggle"),
        }
    }

    #[test]
    fn meeting_plain_actions_parse() {
        assert!(matches!(parse_meeting(&["stop"]), MeetingCmd::Stop));
        assert!(matches!(parse_meeting(&["cancel"]), MeetingCmd::Cancel));
        assert!(matches!(parse_meeting(&["status"]), MeetingCmd::Status));
        assert!(matches!(
            parse_meeting(&["list"]),
            MeetingCmd::List { json: false }
        ));
        assert!(matches!(
            parse_meeting(&["list", "--json"]),
            MeetingCmd::List { json: true }
        ));
    }

    #[test]
    fn meeting_id_actions_parse() {
        match parse_meeting(&["path", "last"]) {
            MeetingCmd::Path { id } => assert_eq!(id, "last"),
            _ => panic!("expected path"),
        }
        match parse_meeting(&["copy", "2026-10-02"]) {
            MeetingCmd::Copy { id } => assert_eq!(id, "2026-10-02"),
            _ => panic!("expected copy"),
        }
        match parse_meeting(&["retry", "last"]) {
            MeetingCmd::Retry { id } => assert_eq!(id, "last"),
            _ => panic!("expected retry"),
        }
        for action in ["show", "path", "copy", "retry"] {
            let argv = ["whisrs", "meeting", action];
            assert!(Cli::try_parse_from(argv).is_err(), "{action} needs an id");
        }
    }

    #[test]
    fn meeting_show_format_defaults_to_md() {
        match parse_meeting(&["show", "last"]) {
            MeetingCmd::Show { id, format } => {
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
            match parse_meeting(&["show", "last", "--format", arg]) {
                MeetingCmd::Show { format, .. } => assert_eq!(format, want),
                _ => panic!("expected show"),
            }
        }
        match parse_meeting(&["show", "x", "-f", "json"]) {
            MeetingCmd::Show { format, .. } => assert_eq!(format, ShowFormat::Json),
            _ => panic!("expected show"),
        }
        assert!(
            Cli::try_parse_from(["whisrs", "meeting", "show", "x", "--format", "pdf"]).is_err()
        );
    }

    #[test]
    fn meeting_requires_an_action() {
        assert!(Cli::try_parse_from(["whisrs", "meeting"]).is_err());
        assert!(Cli::try_parse_from(["whisrs", "meeting", "bogus"]).is_err());
    }

    fn at(h: u32, m: u32) -> chrono::DateTime<chrono::Local> {
        use chrono::TimeZone;
        chrono::Local
            .with_ymd_and_hms(2026, 10, 2, h, m, 0)
            .single()
            .unwrap()
    }

    #[test]
    fn meeting_status_idle() {
        assert_eq!(
            format_meeting_status(&MeetingStatus::default(), false),
            "Idle"
        );
    }

    #[test]
    fn meeting_status_recording() {
        let status = MeetingStatus {
            phase: MeetingPhase::Recording,
            id: Some("2026-10-02_1405_weekly-sync".into()),
            title: Some("Weekly sync".into()),
            started_at: Some(at(14, 5)),
            ..Default::default()
        };
        assert_eq!(
            format_meeting_status(&status, false),
            "Recording since 14:05 (Weekly sync)  id 2026-10-02_1405_weekly-sync"
        );

        let untitled = MeetingStatus {
            title: None,
            id: Some("2026-10-02_1405".into()),
            ..status
        };
        assert_eq!(
            format_meeting_status(&untitled, false),
            "Recording since 14:05  id 2026-10-02_1405"
        );
    }

    #[test]
    fn meeting_status_processing_with_queue() {
        let status = MeetingStatus {
            phase: MeetingPhase::Processing,
            id: Some("a".into()),
            started_at: Some(at(9, 0)),
            queued: vec!["b".into(), "c".into()],
            ..Default::default()
        };
        assert_eq!(
            format_meeting_status(&status, false),
            "Processing  id a  (2 queued)"
        );
    }

    #[test]
    fn meeting_status_color_wraps_phase_only() {
        let status = MeetingStatus {
            phase: MeetingPhase::Recording,
            started_at: Some(at(8, 30)),
            ..Default::default()
        };
        assert_eq!(
            format_meeting_status(&status, true),
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
    fn meeting_row_and_json() {
        let mut meta = MeetingMeta::new(
            "2026-10-02_1405_sync".into(),
            Some("Sync".into()),
            at(14, 5),
        );
        assert_eq!(meeting_duration_secs(&meta), None);
        assert!(format_meeting_row(&meta, 20).contains("       -  recording   Sync"));

        meta.ended_at = Some(at(15, 7));
        meta.status = MeetingState::Done;
        assert_eq!(meeting_duration_secs(&meta), Some(62 * 60));
        assert_eq!(
            format_meeting_row(&meta, 20),
            "2026-10-02_1405_sync  2026-10-02 14:05   1:02:00  done        Sync"
        );

        let json = meeting_list_json(std::path::Path::new("/m/x"), &meta);
        assert_eq!(json["id"], "2026-10-02_1405_sync");
        assert_eq!(json["status"], "done");
        assert_eq!(json["duration_secs"], 3720);
        assert_eq!(json["dir"], "/m/x");
        assert!(json.get("segments").is_none());
    }

    #[test]
    fn not_ready_message_names_status_and_retry() {
        let mut meta = MeetingMeta::new("m1".into(), None, at(10, 0));
        assert_eq!(
            not_ready_message(&meta),
            "meeting m1 has no transcript yet (status: recording)."
        );
        meta.status = MeetingState::Failed;
        meta.error = Some("backend down".into());
        assert!(not_ready_message(&meta).contains("backend down"));
        assert!(not_ready_message(&meta).contains("whisrs meeting retry m1"));
    }

    #[test]
    fn meeting_dir_read_from_config_toml() {
        assert_eq!(
            meeting_dir_from_toml("[meeting]\ndir = \"~/notes/meetings\"\n").as_deref(),
            Some("~/notes/meetings")
        );
        assert_eq!(
            meeting_dir_from_toml("[general]\nbackend = \"groq\"\n"),
            None
        );
        assert_eq!(meeting_dir_from_toml("not toml ["), None);
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
