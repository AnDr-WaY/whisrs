//! System tray implementation using ksni (StatusNotifierItem).

use ksni::menu::StandardItem;
use ksni::{Icon, MenuItem, ToolTip, TrayMethods};
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};

use super::NotifyFn;
use crate::meeting::{MeetingAction, MeetingPhase, MeetingStatus};
use crate::service::{RestartOutcome, ServiceManager};
use crate::{Command, State};
use icons::IconSet;

/// Speech-bubble tray icons, one set per dictation state plus two for a
/// meeting, at several sizes so HiDPI panels get a sharp render instead of an
/// upscaled 16x16. Dictation states fill the bubble; a meeting outlines it in
/// the matching color (red while recording, amber while transcribing), so the
/// two read apart at a glance.
///
/// The PNGs in `src/tray/icons/` are rendered from the SVGs next to them:
/// `rsvg-convert -w N -h N <name>.svg -o <name>-N.png` for each size.
mod icons {
    use std::sync::OnceLock;

    use ksni::Icon;

    use crate::State;

    macro_rules! set_pngs {
        ($name:literal) => {
            [
                include_bytes!(concat!("icons/", $name, "-16.png")).as_slice(),
                include_bytes!(concat!("icons/", $name, "-22.png")).as_slice(),
                include_bytes!(concat!("icons/", $name, "-32.png")).as_slice(),
                include_bytes!(concat!("icons/", $name, "-48.png")).as_slice(),
            ]
        };
    }

    /// One icon design, in all its sizes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum IconSet {
        Idle,
        Recording,
        Transcribing,
        Synthesizing,
        Speaking,
        /// A meeting is recording while dictation is idle.
        MeetingRecording,
        /// A meeting is being transcribed while dictation is idle.
        MeetingProcessing,
    }

    impl IconSet {
        /// Every set, in declaration order (the cache index is `set as usize`).
        pub const ALL: [IconSet; 7] = [
            IconSet::Idle,
            IconSet::Recording,
            IconSet::Transcribing,
            IconSet::Synthesizing,
            IconSet::Speaking,
            IconSet::MeetingRecording,
            IconSet::MeetingProcessing,
        ];

        /// The set for a dictation state.
        pub fn for_state(state: State) -> Self {
            match state {
                State::Idle => IconSet::Idle,
                State::Recording => IconSet::Recording,
                State::Transcribing => IconSet::Transcribing,
                State::Synthesizing => IconSet::Synthesizing,
                State::Speaking => IconSet::Speaking,
            }
        }

        fn pngs(self) -> [&'static [u8]; 4] {
            match self {
                IconSet::Idle => set_pngs!("idle"),
                IconSet::Recording => set_pngs!("recording"),
                IconSet::Transcribing => set_pngs!("transcribing"),
                IconSet::Synthesizing => set_pngs!("synthesizing"),
                IconSet::Speaking => set_pngs!("speaking"),
                IconSet::MeetingRecording => set_pngs!("meeting-recording"),
                IconSet::MeetingProcessing => set_pngs!("meeting-processing"),
            }
        }
    }

    /// Decode an RGBA PNG into the ARGB (big-endian) pixmap SNI expects.
    fn decode(png_bytes: &[u8]) -> Icon {
        let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
        let mut reader = decoder
            .read_info()
            .expect("embedded tray icon is a valid PNG");
        let mut buf = vec![
            0;
            reader
                .output_buffer_size()
                .expect("tray icon size fits in memory")
        ];
        let info = reader
            .next_frame(&mut buf)
            .expect("embedded tray icon decodes");
        assert_eq!(
            (info.color_type, info.bit_depth),
            (png::ColorType::Rgba, png::BitDepth::Eight),
            "tray icons must be 8-bit RGBA (rsvg-convert's output)"
        );
        let data = buf[..info.buffer_size()]
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[r, g, b, a]| [a, r, g, b])
            .collect();
        Icon {
            width: info.width as i32,
            height: info.height as i32,
            data,
        }
    }

    /// All sizes of `set`, decoded once and cached for the daemon's lifetime.
    pub fn get(set: IconSet) -> Vec<Icon> {
        static CACHE: OnceLock<[Vec<Icon>; 7]> = OnceLock::new();
        let cache = CACHE
            .get_or_init(|| IconSet::ALL.map(|s| s.pngs().iter().map(|b| decode(b)).collect()));
        cache[set as usize].clone()
    }
}

/// Small mutable state owned by the tray service itself.
///
/// Keeping this directly on the tray object is important: `ksni::Handle::update`
/// expects the closure to mutate the tray instance so the host knows which
/// properties changed. When the state lives out-of-band, some tray hosts can
/// miss icon refreshes and leave the old color visible.
struct TrayState {
    current: State,
    meeting: MeetingStatus,
}

/// One-line description of the meeting, or `None` when no meeting is active.
fn meeting_line(meeting: &MeetingStatus) -> Option<String> {
    match meeting.phase {
        MeetingPhase::Idle => None,
        MeetingPhase::Recording => {
            let mut line = "Meeting recording".to_string();
            if let Some(started) = meeting.started_at {
                line.push_str(&format!(" since {}", started.format("%H:%M")));
            }
            if let Some(title) = meeting.title.as_deref().filter(|t| !t.is_empty()) {
                line.push_str(&format!(" ({title})"));
            }
            Some(line)
        }
        MeetingPhase::Processing => Some("Processing meeting...".to_string()),
    }
}

/// Human-readable word for a state, used in the title and menu status line.
fn state_word(state: State) -> &'static str {
    match state {
        State::Idle => "idle",
        State::Recording => "recording",
        State::Transcribing => "transcribing",
        State::Synthesizing => "synthesizing",
        State::Speaking => "speaking",
    }
}

/// The ksni tray implementation.
struct WhisrsTray {
    state: TrayState,
    /// Sender into the daemon's shared command dispatch loop — the same loop
    /// the hotkey listener feeds — so tray clicks drive `handle_command`
    /// exactly like hotkey presses do.
    cmd_tx: mpsc::Sender<Command>,
    /// Desktop-toast hook from the daemon (`None` when notifications are
    /// disabled), used to surface menu-callback failures the journal alone
    /// would hide — currently a failed "Restart Daemon" click.
    notify: Option<NotifyFn>,
}

impl WhisrsTray {
    /// Queue a command for the daemon without blocking.
    ///
    /// ksni invokes tray callbacks on the tray service task and warns against
    /// blocking there, so use `try_send`; if the queue is somehow full the
    /// click is dropped with a warning instead of freezing the tray.
    fn send(&self, cmd: Command) {
        if let Err(e) = self.cmd_tx.try_send(cmd) {
            warn!("tray: failed to queue command for daemon: {e}");
        }
    }
}

impl ksni::Tray for WhisrsTray {
    fn id(&self) -> String {
        "whisrs".to_string()
    }

    fn title(&self) -> String {
        // A meeting only takes over the title while dictation is idle: a
        // dictation started before the meeting, or while it is transcribed,
        // matters more in the moment.
        match (self.state.current, self.state.meeting.phase) {
            (State::Idle, MeetingPhase::Recording) => "whisrs: meeting recording".to_string(),
            (State::Idle, MeetingPhase::Processing) => "whisrs: processing meeting".to_string(),
            (state, _) => format!("whisrs: {}", state_word(state)),
        }
    }

    /// Left-click on the icon: toggle recording, same as `whisrs toggle`.
    fn activate(&mut self, _x: i32, _y: i32) {
        debug!("tray activated (left-click): toggle");
        self.send(Command::Toggle { language: None });
    }

    fn icon_pixmap(&self) -> Vec<Icon> {
        // A meeting only takes over the icon while dictation is idle: a
        // dictation started before the meeting, or while it is transcribed,
        // matters more in the moment.
        let set = match (self.state.current, self.state.meeting.phase) {
            (State::Idle, MeetingPhase::Recording) => IconSet::MeetingRecording,
            (State::Idle, MeetingPhase::Processing) => IconSet::MeetingProcessing,
            (state, _) => IconSet::for_state(state),
        };
        icons::get(set)
    }

    fn tool_tip(&self) -> ToolTip {
        let description = match self.state.current {
            State::Idle => "Idle — ready to record",
            State::Recording => "Recording...",
            State::Transcribing => "Transcribing...",
            State::Synthesizing => "Synthesizing…",
            State::Speaking => "Reading aloud…",
        };
        // The meeting line leads while dictation is idle, and is appended
        // otherwise so a running meeting is never hidden.
        let description = match (self.state.current, meeting_line(&self.state.meeting)) {
            (_, None) => description.to_string(),
            (State::Idle, Some(meeting)) => meeting,
            (_, Some(meeting)) => format!("{description}\n{meeting}"),
        };
        ToolTip {
            title: "whisrs".to_string(),
            description,
            icon_name: String::new(),
            icon_pixmap: Vec::new(),
        }
    }

    /// Right-click menu.
    ///
    /// Deliberately holds no dictation controls: whisrs is driven by hotkeys,
    /// and toggle/cancel each already have a hotkey, a CLI command, and (for
    /// toggle) the left-click above. A menu item for them would be a fourth
    /// way to do the same thing. Meetings are the exception: they run for an
    /// hour, often started without a hotkey bound, and the tray is where the
    /// user sees one is running, so the start/stop/cancel controls live here.
    fn menu(&self) -> Vec<MenuItem<Self>> {
        let state = self.state.current;
        let mut items = vec![
            // Non-interactive status line: version + current state.
            MenuItem::Standard(StandardItem {
                label: format!(
                    "whisrs v{} — {}",
                    env!("CARGO_PKG_VERSION"),
                    state_word(state)
                ),
                enabled: false,
                ..Default::default()
            }),
            MenuItem::Separator,
        ];
        items.extend(meeting_menu_items(&self.state.meeting));
        items.extend([
            MenuItem::Separator,
            MenuItem::Standard(StandardItem {
                label: "Restart Daemon".to_string(),
                activate: Box::new(|tray: &mut Self| {
                    restart_daemon(tray.notify);
                }),
                ..Default::default()
            }),
            MenuItem::Standard(StandardItem {
                label: "Quit".to_string(),
                activate: Box::new(|_tray: &mut Self| {
                    quit_daemon();
                }),
                ..Default::default()
            }),
        ]);
        items
    }
}

/// The meeting section of the menu: a status line while a meeting is active,
/// a start/stop toggle, and cancel while recording.
fn meeting_menu_items(meeting: &MeetingStatus) -> Vec<MenuItem<WhisrsTray>> {
    let mut items = Vec::new();
    if let Some(line) = meeting_line(meeting) {
        items.push(MenuItem::Standard(StandardItem {
            label: line,
            enabled: false,
            ..Default::default()
        }));
    }
    let recording = meeting.phase == MeetingPhase::Recording;
    items.push(MenuItem::Standard(StandardItem {
        label: if recording {
            "Stop meeting"
        } else {
            "Start meeting"
        }
        .to_string(),
        activate: Box::new(|tray: &mut WhisrsTray| {
            tray.send(meeting_command(MeetingAction::Toggle));
        }),
        ..Default::default()
    }));
    if recording {
        items.push(MenuItem::Standard(StandardItem {
            label: "Cancel meeting".to_string(),
            activate: Box::new(|tray: &mut WhisrsTray| {
                tray.send(meeting_command(MeetingAction::Cancel));
            }),
            ..Default::default()
        }));
    }
    items
}

fn meeting_command(action: MeetingAction) -> Command {
    Command::Meeting {
        action,
        title: None,
        id: None,
    }
}

/// Restart the daemon through the detected service manager, from the tray menu.
///
/// This cannot go through the daemon's own command loop: a successful restart
/// kills this very process before it could reply. Runs on its own thread
/// because ksni menu callbacks must not block and the restart is a subprocess
/// round-trip.
///
/// Failure raises a desktop toast (when `notify` is set): without one, a click
/// on a setup with no installed service would silently do nothing, since log
/// warnings are invisible from the tray.
fn restart_daemon(notify: Option<NotifyFn>) {
    info!("tray: restart requested");
    std::thread::spawn(move || {
        let manager = ServiceManager::detect();
        match manager.restart() {
            // When this daemon runs under the service, the manager kills it
            // mid-restart, so this line is normally never reached.
            RestartOutcome::Restarted => {
                info!("tray: daemon restarted via {}", manager.name())
            }
            RestartOutcome::NoService => {
                warn!("tray: no whisrs user service installed — restart the daemon manually");
                if let Some(notify) = notify {
                    notify(
                        "whisrs",
                        "Restart from the tray needs an installed whisrs user service. \
                         Restart the daemon manually.",
                    );
                }
            }
            RestartOutcome::Failed => {
                let hint = manager.restart_hint().unwrap_or("the restart command");
                warn!("tray: `{hint}` failed");
                if let Some(notify) = notify {
                    notify(
                        "whisrs",
                        &format!("Daemon restart failed: `{hint}` returned an error."),
                    );
                }
            }
        }
    });
}

/// Quit from the tray menu by sending the daemon SIGTERM, so it takes the
/// same shutdown path as `systemctl stop`: a recording meeting is finalized
/// and the IPC socket removed before the process exits.
fn quit_daemon() {
    info!("tray: quit requested, shutting down");
    // SAFETY: kill(2) on our own pid has no memory-safety preconditions.
    let rc = unsafe { libc::kill(libc::getpid(), libc::SIGTERM) };
    if rc != 0 {
        // No handler could run; fall back to the old immediate exit.
        let _ = std::fs::remove_file(crate::socket_path());
        std::process::exit(0);
    }
}

/// Maximum number of attempts to connect to the SNI tray host.
const TRAY_MAX_RETRIES: u32 = 10;

/// Initial retry delay (doubles each attempt, capped at 10 s).
const TRAY_INITIAL_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

/// Spawn the system tray indicator.
///
/// Runs in the background and updates the icon whenever the daemon state changes.
/// Retries with exponential backoff if the SNI host isn't available yet (common
/// on boot when the daemon starts before the desktop environment is fully ready).
pub async fn spawn_tray(
    mut state_rx: watch::Receiver<State>,
    mut meeting_rx: watch::Receiver<MeetingStatus>,
    cmd_tx: mpsc::Sender<Command>,
    notify: Option<NotifyFn>,
) {
    // Retry spawning the tray with exponential backoff.
    let mut delay = TRAY_INITIAL_DELAY;
    let mut handle = None;

    for attempt in 1..=TRAY_MAX_RETRIES {
        let tray = WhisrsTray {
            state: TrayState {
                current: *state_rx.borrow(),
                meeting: meeting_rx.borrow().clone(),
            },
            cmd_tx: cmd_tx.clone(),
            notify,
        };

        match tray.spawn().await {
            Ok(h) => {
                info!("system tray started (attempt {attempt})");
                handle = Some(h);
                break;
            }
            Err(e) => {
                if attempt == TRAY_MAX_RETRIES {
                    warn!(
                        "failed to start system tray after {TRAY_MAX_RETRIES} attempts: {e} — continuing without tray"
                    );
                    return;
                }
                info!(
                    "tray host not available (attempt {attempt}/{TRAY_MAX_RETRIES}): {e} — retrying in {delay:?}"
                );
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(std::time::Duration::from_secs(10));
            }
        }
    }

    let handle = handle.expect("handle must be set after successful spawn");

    // Watch for dictation and meeting changes and update the tray. The
    // dictation channel ends the loop when it closes (the daemon is going
    // away); a closed meeting channel only stops meeting updates.
    tokio::spawn(async move {
        let mut meeting_open = true;
        loop {
            tokio::select! {
                changed = state_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let new_state = *state_rx.borrow();
                    debug!("tray state update: {new_state:?}");
                    // Mutate the tray object itself so ksni emits the
                    // corresponding D-Bus property changes for title,
                    // tooltip, and icon pixmap.
                    handle
                        .update(|tray| {
                            tray.state.current = new_state;
                        })
                        .await;
                }
                changed = meeting_rx.changed(), if meeting_open => {
                    if changed.is_err() {
                        meeting_open = false;
                        continue;
                    }
                    let meeting = meeting_rx.borrow().clone();
                    debug!("tray meeting update: {:?}", meeting.phase);
                    handle
                        .update(|tray| {
                            tray.state.meeting = meeting;
                        })
                        .await;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    // Brings the `activate` trait method into scope for method-call syntax
    // below; `impl ksni::Tray for WhisrsTray` above only qualifies the path,
    // it doesn't import the trait.
    use ksni::Tray as _;

    fn tray_with(
        cmd_tx: mpsc::Sender<Command>,
        current: State,
        meeting: MeetingStatus,
    ) -> WhisrsTray {
        WhisrsTray {
            state: TrayState { current, meeting },
            cmd_tx,
            notify: None,
        }
    }

    fn recording_meeting(title: Option<&str>) -> MeetingStatus {
        use chrono::TimeZone;
        MeetingStatus {
            phase: MeetingPhase::Recording,
            id: Some("2026-10-02_0930".to_string()),
            title: title.map(str::to_string),
            started_at: Some(
                chrono::Local
                    .with_ymd_and_hms(2026, 10, 2, 9, 30, 0)
                    .unwrap(),
            ),
            ..Default::default()
        }
    }

    fn labels(items: &[MenuItem<WhisrsTray>]) -> Vec<(String, bool)> {
        items
            .iter()
            .filter_map(|i| match i {
                MenuItem::Standard(s) => Some((s.label.clone(), s.enabled)),
                _ => None,
            })
            .collect()
    }

    /// Activate the standard menu item with this label.
    fn click(tray: &mut WhisrsTray, label: &str) {
        let mut items = meeting_menu_items(&tray.state.meeting);
        let item = items
            .iter_mut()
            .find_map(|i| match i {
                MenuItem::Standard(s) if s.label == label => Some(s),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no menu item `{label}`"));
        let activate = std::mem::replace(&mut item.activate, Box::new(|_: &mut WhisrsTray| {}));
        activate(tray);
    }

    #[test]
    fn meeting_line_describes_each_phase() {
        assert_eq!(meeting_line(&MeetingStatus::default()), None);
        assert_eq!(
            meeting_line(&recording_meeting(Some("standup"))).as_deref(),
            Some("Meeting recording since 09:30 (standup)")
        );
        assert_eq!(
            meeting_line(&recording_meeting(None)).as_deref(),
            Some("Meeting recording since 09:30")
        );
        let processing = MeetingStatus {
            phase: MeetingPhase::Processing,
            ..Default::default()
        };
        assert_eq!(
            meeting_line(&processing).as_deref(),
            Some("Processing meeting...")
        );
    }

    #[test]
    fn idle_meeting_menu_offers_only_start() {
        assert_eq!(
            labels(&meeting_menu_items(&MeetingStatus::default())),
            vec![("Start meeting".to_string(), true)]
        );
    }

    #[test]
    fn recording_meeting_menu_offers_status_stop_and_cancel() {
        assert_eq!(
            labels(&meeting_menu_items(&recording_meeting(None))),
            vec![
                ("Meeting recording since 09:30".to_string(), false),
                ("Stop meeting".to_string(), true),
                ("Cancel meeting".to_string(), true),
            ]
        );
    }

    #[test]
    fn meeting_menu_items_send_toggle_and_cancel() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel(4);
        let mut tray = tray_with(cmd_tx, State::Idle, recording_meeting(None));

        click(&mut tray, "Stop meeting");
        click(&mut tray, "Cancel meeting");

        for expected in [MeetingAction::Toggle, MeetingAction::Cancel] {
            let cmd = cmd_rx.try_recv().expect("a meeting command is queued");
            assert!(
                matches!(cmd, Command::Meeting { action, title: None, id: None } if action == expected),
                "expected meeting {expected:?}, got {cmd:?}"
            );
        }
    }

    /// Every size's pixmap, for comparing whole icon sets.
    fn pixmaps(icons: &[Icon]) -> Vec<&[u8]> {
        icons.iter().map(|i| i.data.as_slice()).collect()
    }

    /// The meeting shows in the icon and tooltip only while dictation is
    /// idle; a dictation in progress keeps its own icon, and the meeting is
    /// still named in the tooltip.
    #[test]
    fn dictation_state_wins_the_icon_during_a_meeting() {
        let (cmd_tx, _rx) = mpsc::channel(1);
        let idle = tray_with(cmd_tx.clone(), State::Idle, recording_meeting(Some("x")));
        assert_eq!(
            pixmaps(&idle.icon_pixmap()),
            pixmaps(&icons::get(IconSet::MeetingRecording))
        );
        assert_eq!(
            idle.tool_tip().description,
            "Meeting recording since 09:30 (x)"
        );

        let processing = MeetingStatus {
            phase: MeetingPhase::Processing,
            ..Default::default()
        };
        let idle_processing = tray_with(cmd_tx.clone(), State::Idle, processing.clone());
        assert_eq!(
            pixmaps(&idle_processing.icon_pixmap()),
            pixmaps(&icons::get(IconSet::MeetingProcessing))
        );

        let dictating = tray_with(
            cmd_tx.clone(),
            State::Recording,
            recording_meeting(Some("x")),
        );
        assert_eq!(
            pixmaps(&dictating.icon_pixmap()),
            pixmaps(&icons::get(IconSet::Recording))
        );
        assert!(dictating
            .tool_tip()
            .description
            .ends_with("Meeting recording since 09:30 (x)"));

        let transcribing = tray_with(cmd_tx, State::Transcribing, processing);
        assert_eq!(
            pixmaps(&transcribing.icon_pixmap()),
            pixmaps(&icons::get(IconSet::Transcribing))
        );
    }

    /// No two icon sets share a pixmap at any size, so a meeting never looks
    /// like a dictation state (or the other meeting phase) on the panel.
    #[test]
    fn every_icon_set_is_distinct_at_every_size() {
        let sets: Vec<(IconSet, Vec<Icon>)> =
            IconSet::ALL.iter().map(|&s| (s, icons::get(s))).collect();
        for (i, (a, a_icons)) in sets.iter().enumerate() {
            for (b, b_icons) in &sets[i + 1..] {
                for (x, y) in a_icons.iter().zip(b_icons) {
                    assert_ne!(x.data, y.data, "{a:?} and {b:?} match at {}px", x.width);
                }
            }
        }
    }

    /// `IconSet::ALL` lists every set in declaration order, which is what
    /// `icons::get` indexes its cache by.
    #[test]
    fn icon_set_cache_index_is_the_declaration_order() {
        for (i, set) in IconSet::ALL.iter().enumerate() {
            assert_eq!(*set as usize, i, "{set:?}");
        }
    }

    /// `activate()` (left-click on the tray icon) is the only tray-driven
    /// command, and nothing checks that it still sends `Toggle` after a
    /// refactor — a typo'd variant here would ship silently. Pin it.
    #[test]
    fn activate_queues_toggle_command() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel(1);
        let mut tray = tray_with(cmd_tx, State::Idle, MeetingStatus::default());

        tray.activate(0, 0);

        let cmd = cmd_rx
            .try_recv()
            .expect("activate() should queue a command");
        assert!(
            matches!(cmd, Command::Toggle { language: None }),
            "expected Command::Toggle {{ language: None }}, got {cmd:?}"
        );
    }

    /// Every icon set (each dictation state and both meeting phases) ships
    /// every size, and each decodes to a full ARGB pixmap that is not blank.
    /// Catches a missing or mis-rendered PNG at test time rather than as a
    /// panic in the running tray.
    #[test]
    fn every_state_has_all_icon_sizes() {
        for state in [
            State::Idle,
            State::Recording,
            State::Transcribing,
            State::Synthesizing,
            State::Speaking,
        ] {
            assert!(IconSet::ALL.contains(&IconSet::for_state(state)));
        }
        for set in IconSet::ALL {
            let icons = icons::get(set);
            let sizes: Vec<i32> = icons.iter().map(|i| i.width).collect();
            assert_eq!(sizes, [16, 22, 32, 48], "{set:?}");
            for icon in &icons {
                assert_eq!(icon.width, icon.height, "{set:?} icon is square");
                assert_eq!(
                    icon.data.len(),
                    (icon.width * icon.height * 4) as usize,
                    "{set:?} {}px pixmap length",
                    icon.width
                );
                assert!(
                    icon.data.as_chunks::<4>().0.iter().any(|p| p[0] > 0),
                    "{set:?} {}px icon is fully transparent",
                    icon.width
                );
            }
        }
    }
}
