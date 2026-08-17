//! Hooks fired when a recording session starts/stops. Driven by the state
//! broadcast channel (see [`crate::daemon::hooks::hook_dispatch_loop`]).

use crate::mpris::PlayerState;
use crate::State;
use tracing::{debug, warn};

/// Which hook set fires for a state change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    /// A recording session began.
    RecordStart,
    /// A recording session ended.
    RecordStop,
}

/// Map a state change to the hook that fires; `None` for no-op transitions.
///
/// `RecordStop` fires at the first non-`Recording` state. For dictation that
/// is `Transcribing` (MPRIS resumes while transcription runs); for command
/// mode it is `Idle` (finalize broadcasts after the full pipeline completes).
pub fn hook_event_for(prev: State, new: State) -> Option<HookEvent> {
    match (prev, new) {
        (prev, State::Recording) if prev != State::Recording => Some(HookEvent::RecordStart),
        (State::Recording, new) if new != State::Recording => Some(HookEvent::RecordStop),
        _ => None,
    }
}

/// The MPRIS work one hook event implies. Either list may be empty; an empty
/// list means "issue no D-Bus calls", not "sweep the bus".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MediaPlan {
    /// Bus names to pause, in the order they were seen.
    pub pause: Vec<String>,
    /// Bus names to resume. Never contains a player this daemon did not
    /// pause itself.
    pub resume: Vec<String>,
}

/// Remembers which MPRIS players *this daemon* paused, so a recording stop
/// resumes exactly those. A tab the user paused before dictating is never in
/// the set, so it is never started for them.
///
/// Pure bookkeeping — the D-Bus calls live in [`crate::mpris`] and the daemon
/// feeds their results back in via [`Self::confirm_paused`].
#[derive(Debug, Default)]
pub struct MediaPauseTracker {
    paused: Vec<String>,
}

impl MediaPauseTracker {
    /// A tracker holding nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Plan the MPRIS work for `event`.
    ///
    /// `players` is the session-bus snapshot taken at a recording start; it is
    /// unused for [`HookEvent::RecordStop`], where the plan is exactly the set
    /// this tracker is still holding. Planning a pause does *not* record it —
    /// the caller reports back what actually paused via [`Self::confirm_paused`].
    pub fn plan(&mut self, event: HookEvent, players: &[PlayerState]) -> MediaPlan {
        match event {
            // A start with no intervening stop is reachable: the state watch
            // channel coalesces, so Recording → Idle → Recording can surface as
            // a bare second start. Keep holding the earlier batch instead of
            // forgetting it — those players stay paused until a stop arrives.
            HookEvent::RecordStart => MediaPlan {
                pause: players
                    .iter()
                    .filter(|p| p.playing)
                    .map(|p| p.name.clone())
                    .collect(),
                resume: Vec::new(),
            },
            HookEvent::RecordStop => MediaPlan {
                pause: Vec::new(),
                resume: std::mem::take(&mut self.paused),
            },
        }
    }

    /// Record the players that actually paused, adding to anything still held
    /// from an earlier start. A player that failed to pause is simply absent,
    /// so it is never resumed.
    pub fn confirm_paused(&mut self, paused: &[String]) {
        for name in paused {
            if !self.paused.iter().any(|held| held == name) {
                self.paused.push(name.clone());
            }
        }
    }

    /// The players currently held paused, in the order they were paused.
    pub fn held(&self) -> &[String] {
        &self.paused
    }
}

/// Run a shell command fire-and-forget via `sh -c`. Never blocks recording.
/// Empty commands are skipped. Hangs are killed after 30 seconds.
pub fn run_hook(cmd: &str) {
    if cmd.trim().is_empty() {
        return;
    }
    match tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .kill_on_drop(true)
        .spawn()
    {
        Ok(mut child) => {
            debug!("running hook: {cmd}");
            let cmd = cmd.to_string();
            tokio::spawn(async move {
                match tokio::time::timeout(std::time::Duration::from_secs(30), child.wait()).await {
                    Ok(Ok(status)) => {
                        if !status.success() {
                            warn!("hook `{cmd}` exited with {status}");
                        }
                    }
                    Ok(Err(e)) => warn!("hook `{cmd}` wait error: {e}"),
                    Err(_) => {
                        warn!("hook `{cmd}` timed out after 30s, killing");
                        let _ = child.kill().await;
                    }
                }
            });
        }
        Err(e) => warn!("failed to run record hook `{cmd}`: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::State;

    #[test]
    fn idle_to_recording_fires_start() {
        assert_eq!(
            hook_event_for(State::Idle, State::Recording),
            Some(HookEvent::RecordStart)
        );
    }

    #[test]
    fn transcribing_to_recording_fires_start() {
        // Never occurs today, but entering Recording from any non-Recording
        // state is a start.
        assert_eq!(
            hook_event_for(State::Transcribing, State::Recording),
            Some(HookEvent::RecordStart)
        );
    }

    #[test]
    fn recording_to_transcribing_fires_stop() {
        assert_eq!(
            hook_event_for(State::Recording, State::Transcribing),
            Some(HookEvent::RecordStop)
        );
    }

    #[test]
    fn recording_to_idle_fires_stop() {
        assert_eq!(
            hook_event_for(State::Recording, State::Idle),
            Some(HookEvent::RecordStop)
        );
    }

    #[test]
    fn duplicate_broadcasts_are_ignored() {
        assert_eq!(hook_event_for(State::Recording, State::Recording), None);
        assert_eq!(hook_event_for(State::Idle, State::Idle), None);
        assert_eq!(
            hook_event_for(State::Transcribing, State::Transcribing),
            None
        );
    }

    #[test]
    fn read_aloud_transitions_are_ignored() {
        assert_eq!(hook_event_for(State::Idle, State::Synthesizing), None);
        assert_eq!(hook_event_for(State::Synthesizing, State::Speaking), None);
        assert_eq!(hook_event_for(State::Speaking, State::Idle), None);
    }

    #[test]
    fn transcribing_to_idle_is_ignored() {
        assert_eq!(hook_event_for(State::Transcribing, State::Idle), None);
    }

    #[tokio::test]
    async fn empty_hook_is_noop() {
        // Should not panic or spawn any child.
        run_hook("");
        run_hook("   ");
    }

    // -- MediaPauseTracker ---------------------------------------------------

    fn playing(name: &str) -> PlayerState {
        PlayerState {
            name: name.to_string(),
            playing: true,
        }
    }

    fn not_playing(name: &str) -> PlayerState {
        PlayerState {
            name: name.to_string(),
            playing: false,
        }
    }

    #[test]
    fn stop_resumes_exactly_what_start_paused() {
        let mut tracker = MediaPauseTracker::new();
        let bus = [playing("spotify"), playing("vlc")];

        let start = tracker.plan(HookEvent::RecordStart, &bus);
        assert_eq!(start.pause, ["spotify", "vlc"]);
        assert!(start.resume.is_empty(), "a start never resumes");
        tracker.confirm_paused(&start.pause);

        let stop = tracker.plan(HookEvent::RecordStop, &[]);
        assert_eq!(stop.resume, ["spotify", "vlc"]);
        assert!(stop.pause.is_empty(), "a stop never pauses");
    }

    /// The bug this tracker exists for: a tab the user paused themselves must
    /// not start playing when dictation ends.
    #[test]
    fn a_player_already_paused_is_never_touched() {
        let mut tracker = MediaPauseTracker::new();
        let bus = [
            playing("spotify"),
            not_playing("firefox"),
            not_playing("vlc"),
        ];

        let start = tracker.plan(HookEvent::RecordStart, &bus);
        assert_eq!(
            start.pause,
            ["spotify"],
            "only the playing player may be paused"
        );
        tracker.confirm_paused(&start.pause);

        let stop = tracker.plan(HookEvent::RecordStop, &[]);
        assert_eq!(
            stop.resume,
            ["spotify"],
            "firefox and vlc were the user's to keep paused"
        );
    }

    #[test]
    fn second_start_without_a_stop_keeps_the_first_batch() {
        let mut tracker = MediaPauseTracker::new();

        let first = tracker.plan(HookEvent::RecordStart, &[playing("spotify")]);
        tracker.confirm_paused(&first.pause);

        // The watch channel coalesced a stop away; spotify is still paused by
        // us, and a new player started in the meantime.
        let second = tracker.plan(
            HookEvent::RecordStart,
            &[not_playing("spotify"), playing("vlc")],
        );
        assert_eq!(second.pause, ["vlc"]);
        tracker.confirm_paused(&second.pause);
        assert_eq!(tracker.held(), ["spotify", "vlc"]);

        let stop = tracker.plan(HookEvent::RecordStop, &[]);
        assert_eq!(
            stop.resume,
            ["spotify", "vlc"],
            "the first batch must not be stranded"
        );
    }

    #[test]
    fn a_player_still_playing_at_a_second_start_is_paused_once() {
        let mut tracker = MediaPauseTracker::new();

        let first = tracker.plan(HookEvent::RecordStart, &[playing("spotify")]);
        tracker.confirm_paused(&first.pause);

        // The user hit play again mid-recording: pause it, but do not record
        // it twice or the resume list grows duplicates.
        let second = tracker.plan(HookEvent::RecordStart, &[playing("spotify")]);
        assert_eq!(second.pause, ["spotify"]);
        tracker.confirm_paused(&second.pause);

        assert_eq!(tracker.plan(HookEvent::RecordStop, &[]).resume, ["spotify"]);
    }

    #[test]
    fn stop_with_nothing_paused_resumes_nothing() {
        let mut tracker = MediaPauseTracker::new();
        let stop = tracker.plan(HookEvent::RecordStop, &[]);
        assert!(
            stop.resume.is_empty(),
            "a stop must never sweep the bus for players to resume"
        );
        assert!(stop.pause.is_empty());
    }

    #[test]
    fn a_stop_forgets_its_batch() {
        let mut tracker = MediaPauseTracker::new();
        tracker.confirm_paused(&["spotify".to_string()]);

        assert_eq!(tracker.plan(HookEvent::RecordStop, &[]).resume, ["spotify"]);
        assert!(
            tracker.plan(HookEvent::RecordStop, &[]).resume.is_empty(),
            "a second stop must not resume the same batch again"
        );
        assert!(tracker.held().is_empty());
    }

    #[test]
    fn a_pause_that_failed_is_never_resumed() {
        let mut tracker = MediaPauseTracker::new();
        let start = tracker.plan(
            HookEvent::RecordStart,
            &[playing("spotify"), playing("vlc")],
        );
        assert_eq!(start.pause, ["spotify", "vlc"]);

        // vlc refused the Pause call, so the daemon reports only spotify.
        tracker.confirm_paused(&["spotify".to_string()]);

        assert_eq!(tracker.plan(HookEvent::RecordStop, &[]).resume, ["spotify"]);
    }
}
