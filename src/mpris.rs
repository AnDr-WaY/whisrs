//! Native MPRIS v2 pause/resume over the session D-Bus. Every player
//! (browsers, Spotify, VLC, MPV, KDE Connect) registers as
//! `org.mpris.MediaPlayer2.*`, so the session bus *is* the integration.
//!
//! Only players reporting `PlaybackStatus = "Playing"` are paused, and only
//! those are resumed: media the user paused before dictating stays paused.
//! Failures degrade gracefully — a missing bus, an unreadable property or a
//! bus name that vanished mid-session never blocks dictation.

use std::time::Duration;

use tracing::{debug, warn};

const PLAYER_PATH: &str = "/org/mpris/MediaPlayer2";
const PLAYER_IFACE: &str = "org.mpris.MediaPlayer2.Player";

/// Budget for one sweep of the bus. Reading `PlaybackStatus` costs a
/// round-trip per player, so a wedged player must not stall recording.
const SWEEP_DEADLINE: Duration = Duration::from_secs(5);

/// Per-call ceiling. The sweep deadline is only checked *between* players, so
/// without this a peer that owns its bus name but never replies would block
/// the hook loop forever: the paused media would never resume and no later
/// hook would fire for the life of the daemon.
const CALL_TIMEOUT: Duration = Duration::from_secs(2);

/// A session-bus connection that cannot hang on an unresponsive peer.
async fn session_connection() -> zbus::Result<zbus::Connection> {
    zbus::connection::Builder::session()?
        .method_timeout(CALL_TIMEOUT)
        .build()
        .await
}

/// One MPRIS player as seen on the session bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerState {
    /// The `org.mpris.MediaPlayer2.*` bus name.
    pub name: String,
    /// Whether `PlaybackStatus` read back as `Playing`.
    pub playing: bool,
}

/// Does this bus name identify a controllable MPRIS player?
pub fn is_mpris_player(name: &str) -> bool {
    name.starts_with("org.mpris.MediaPlayer2.")
}

/// A one-shot `org.mpris.MediaPlayer2.Player` proxy for `name`.
///
/// Property caching is off: the default (`Lazily`) turns the first property
/// read into a `GetAll` plus a `PropertiesChanged` match rule per player, and
/// every proxy here is used for a single call.
async fn player_proxy(conn: &zbus::Connection, name: &str) -> zbus::Result<zbus::Proxy<'static>> {
    zbus::proxy::Builder::new(conn)
        .destination(name.to_string())?
        .path(PLAYER_PATH)?
        .interface(PLAYER_IFACE)?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
}

/// Snapshot every MPRIS player on the session bus with its playback status.
///
/// A player whose `PlaybackStatus` cannot be read is skipped entirely, so it
/// is never paused and therefore never resumed — an unreadable player is not
/// an excuse to start someone's music.
pub async fn players() -> Vec<PlayerState> {
    let Ok(conn) = session_connection().await else {
        debug!("no session bus — MPRIS media pause unavailable");
        return Vec::new();
    };
    let Ok(dbus) = zbus::fdo::DBusProxy::new(&conn).await else {
        debug!("failed to create D-Bus proxy — MPRIS media pause unavailable");
        return Vec::new();
    };
    let Ok(names) = dbus.list_names().await else {
        debug!("failed to list D-Bus names — MPRIS media pause unavailable");
        return Vec::new();
    };

    // Pre-filter to MPRIS names so the deadline only counts real players.
    let mpris_names: Vec<String> = names
        .into_iter()
        .filter_map(|n| {
            let s = n.to_string();
            is_mpris_player(&s).then_some(s)
        })
        .collect();

    let deadline = tokio::time::Instant::now() + SWEEP_DEADLINE;
    let mut seen = Vec::with_capacity(mpris_names.len());
    for name in mpris_names {
        // Checked before the PlaybackStatus round-trip, not after it.
        if tokio::time::Instant::now() >= deadline {
            warn!(
                "MPRIS scan timed out after inspecting {} player(s)",
                seen.len()
            );
            break;
        }

        let Ok(proxy) = player_proxy(&conn, &name).await else {
            debug!("skipped {name}: failed to create MPRIS proxy");
            continue;
        };
        let playing = match proxy.get_property::<String>("PlaybackStatus").await {
            Ok(status) => status == "Playing",
            Err(e) => {
                debug!("skipped {name}: failed to read PlaybackStatus: {e}");
                continue;
            }
        };
        debug!("MPRIS player {name}: playing={playing}");
        seen.push(PlayerState { name, playing });
    }
    seen
}

/// Pause `names`, returning exactly the ones that actually paused so a later
/// [`resume`] never touches a player this daemon did not stop.
pub async fn pause(names: &[String]) -> Vec<String> {
    if names.is_empty() {
        return Vec::new();
    }
    let Ok(conn) = session_connection().await else {
        return Vec::new();
    };

    let deadline = tokio::time::Instant::now() + SWEEP_DEADLINE;
    let mut paused = Vec::with_capacity(names.len());
    for name in names {
        if tokio::time::Instant::now() >= deadline {
            warn!(
                "MPRIS pause timed out after pausing {} player(s)",
                paused.len()
            );
            break;
        }

        let Ok(proxy) = player_proxy(&conn, name).await else {
            debug!("skipped {name}: failed to create MPRIS proxy");
            continue;
        };
        match proxy.call_method("Pause", &()).await {
            Ok(_) => {
                debug!("paused MPRIS player {name}");
                paused.push(name.clone());
            }
            Err(e) => warn!("failed to pause MPRIS player {name}: {e}"),
        }
    }
    paused
}

/// Resume exactly `names` — the players this daemon paused, nothing else.
///
/// A name that vanished mid-session (the player exited, or a browser renamed
/// its media session) fails silently. Missing a resume beats resuming media
/// the user never had playing, so there is deliberately no bus-wide fallback.
pub async fn resume(names: &[String]) {
    if names.is_empty() {
        return;
    }
    let Ok(conn) = session_connection().await else {
        return;
    };
    let deadline = tokio::time::Instant::now() + SWEEP_DEADLINE;
    for name in names {
        if tokio::time::Instant::now() >= deadline {
            warn!("MPRIS resume timed out; some players stay paused");
            break;
        }
        let Ok(proxy) = player_proxy(&conn, name).await else {
            continue;
        };
        match proxy.call_method("Play", &()).await {
            Ok(_) => debug!("resumed MPRIS player {name}"),
            Err(e) => debug!("failed to resume MPRIS player {name}: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpris_player_detection() {
        assert!(is_mpris_player("org.mpris.MediaPlayer2.spotify"));
        assert!(is_mpris_player(
            "org.mpris.MediaPlayer2.plasma-browser-integration"
        ));
        assert!(is_mpris_player(
            "org.mpris.MediaPlayer2.kdeconnect.qOo4rGkeX7g"
        ));
        assert!(!is_mpris_player("org.freedesktop.DBus"));
        assert!(!is_mpris_player("org.mpris.MediaPlayer2"));
        assert!(!is_mpris_player("com.spotify.Client"));
    }

    /// No session bus is touched for an empty list — a recording stop with
    /// nothing held must issue no D-Bus traffic at all.
    #[tokio::test]
    async fn empty_lists_are_noops() {
        assert!(pause(&[]).await.is_empty());
        resume(&[]).await;
    }
}
