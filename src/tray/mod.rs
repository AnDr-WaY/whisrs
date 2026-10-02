//! System tray indicator via StatusNotifierItem (KDE/freedesktop SNI protocol).
//!
//! Shows the daemon state (idle/recording/transcribing) as a tray icon, plus
//! meeting recording status and start/stop/cancel meeting menu items.
//! Works with any SNI-compatible tray host: waybar, swaybar, KDE Plasma,
//! GNOME (with AppIndicator extension), etc.

#[cfg(feature = "tray")]
mod service;

#[cfg(feature = "tray")]
pub use service::spawn_tray;

/// Desktop-toast hook the daemon hands to the tray so failures inside menu
/// callbacks (currently a failed "Restart Daemon" click) surface to the user
/// instead of only reaching the journal. A bare `fn` pointer keeps the tray
/// decoupled from the daemon's notification module; `None` means the user has
/// notifications disabled.
pub type NotifyFn = fn(&str, &str);

#[cfg(not(feature = "tray"))]
pub async fn spawn_tray(
    _state_rx: tokio::sync::watch::Receiver<crate::State>,
    _meeting_rx: tokio::sync::watch::Receiver<crate::meeting::MeetingStatus>,
    _cmd_tx: tokio::sync::mpsc::Sender<crate::Command>,
    _notify: Option<NotifyFn>,
) {
    // Tray feature not enabled — no-op.
}
