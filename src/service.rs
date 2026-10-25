//! Init-system abstraction for controlling the whisrs user service.
//!
//! whisrs is a per-session daemon, so it is managed by whatever runs the
//! user's *session* services. That was originally assumed to be systemd, which
//! left users on OpenRC distros (Gentoo, Artix, Alpine, Devuan) being told to
//! run `systemctl` commands that do not exist on their machine.
//!
//! Everything init-specific lives here: detection, the control commands, and
//! the command strings shown in help text. Callers ask the detected
//! [`ServiceManager`] what to do rather than hardcoding a command.

use std::path::{Path, PathBuf};
use std::process::Command;

/// systemd unit name, as installed under `~/.config/systemd/user`.
pub const SYSTEMD_UNIT: &str = "whisrs.service";

/// OpenRC service name, as installed under `~/.config/rc/init.d`.
pub const OPENRC_SERVICE: &str = "whisrs";

/// Which init system manages the user's session services.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceManager {
    /// systemd user manager, driven by `systemctl --user`.
    Systemd,
    /// OpenRC in user mode, driven by `rc-service --user` (OpenRC >= 0.55).
    OpenRc,
    /// Neither detected — the daemon has to be started by hand.
    None,
}

/// Outcome of an attempt to restart the daemon through the service manager.
///
/// The CLI and `whisrs config` both nudge the daemon after writing a new
/// `config.toml`. They share this detection logic but format output
/// differently (e.g. ANSI colors only when stdout is a TTY), so this is a
/// structured outcome rather than something printed directly.
#[derive(Debug, PartialEq, Eq)]
pub enum RestartOutcome {
    /// The restart command succeeded.
    Restarted,
    /// No whisrs service is installed — the caller should show fallback hints.
    NoService,
    /// A service manager is present but the restart command failed.
    Failed,
}

impl ServiceManager {
    /// Detect the init system managing this session.
    ///
    /// Both systemd and OpenRC advertise themselves with a runtime directory,
    /// which is more reliable than looking for the client binaries on `$PATH`:
    /// `systemctl` is often installed on non-systemd machines as a dependency
    /// of some other package, and would otherwise produce a false positive.
    pub fn detect() -> Self {
        Self::detect_in(Path::new("/run"))
    }

    /// [`Self::detect`] against an arbitrary `/run`, so tests can fake it.
    fn detect_in(run_dir: &Path) -> Self {
        // sd_booted(3): this directory exists if and only if the system was
        // booted with systemd.
        if run_dir.join("systemd/system").is_dir() {
            Self::Systemd
        } else if run_dir.join("openrc").is_dir() {
            Self::OpenRc
        } else {
            Self::None
        }
    }

    /// Human-readable name, for log lines and prompts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Systemd => "systemd",
            Self::OpenRc => "OpenRC",
            Self::None => "no service manager",
        }
    }

    /// Whether a whisrs service is installed for this manager.
    pub fn service_installed(self) -> bool {
        match self {
            // `is-enabled` exits 0 for enabled/static/linked units and
            // non-zero when the unit isn't loaded. Fall back to
            // `list-unit-files` for distros where `is-enabled` exits non-zero
            // on static or linked units.
            Self::Systemd => {
                if run_ok("systemctl", &["--user", "is-enabled", SYSTEMD_UNIT]) {
                    return true;
                }
                match capture_ok("systemctl", &["--user", "list-unit-files", SYSTEMD_UNIT]) {
                    Some(out) => out.contains(SYSTEMD_UNIT),
                    None => false,
                }
            }
            // `rc-service --user -e <svc>` exits 0 when the init script
            // exists, 1 when it does not.
            Self::OpenRc => run_ok("rc-service", &["--user", "-e", OPENRC_SERVICE]),
            Self::None => false,
        }
    }

    /// Whether the daemon is currently running under this manager.
    ///
    /// Only distinguishes active from not — callers use this to decide
    /// whether a restart is meaningful, not to report failure detail.
    pub fn is_active(self) -> bool {
        match self {
            Self::Systemd => {
                matches!(
                    capture("systemctl", &["--user", "is-active", SYSTEMD_UNIT]).as_deref(),
                    Some(out) if out.trim() == "active"
                )
            }
            // `status` exits 0 when started and 3 when stopped.
            Self::OpenRc => run_ok("rc-service", &["--user", OPENRC_SERVICE, "status"]),
            Self::None => false,
        }
    }

    /// Restart the daemon, if a service is installed for this manager.
    ///
    /// Returns [`RestartOutcome::NoService`] without running anything when no
    /// service is installed, so callers can print manual instructions instead.
    pub fn restart(self) -> RestartOutcome {
        if !self.service_installed() {
            return RestartOutcome::NoService;
        }
        let ok = match self {
            Self::Systemd => run_ok("systemctl", &["--user", "restart", SYSTEMD_UNIT]),
            Self::OpenRc => run_ok("rc-service", &["--user", OPENRC_SERVICE, "restart"]),
            Self::None => return RestartOutcome::NoService,
        };
        if ok {
            RestartOutcome::Restarted
        } else {
            RestartOutcome::Failed
        }
    }

    /// Command that enables the service to start at login, and starts it now.
    ///
    /// `None` when no service manager was detected, in which case callers
    /// should fall back to [`Self::manual_start_hint`].
    pub fn enable_hint(self) -> Option<&'static str> {
        match self {
            Self::Systemd => Some("systemctl --user enable --now whisrs.service"),
            Self::OpenRc => {
                Some("rc-update --user add whisrs default && rc-service --user whisrs start")
            }
            Self::None => None,
        }
    }

    /// Command that restarts the running daemon.
    pub fn restart_hint(self) -> Option<&'static str> {
        match self {
            Self::Systemd => Some("systemctl --user restart whisrs.service"),
            Self::OpenRc => Some("rc-service --user whisrs restart"),
            Self::None => None,
        }
    }

    /// Command that reports service status.
    pub fn status_hint(self) -> Option<&'static str> {
        match self {
            Self::Systemd => Some("systemctl --user status whisrs.service"),
            Self::OpenRc => Some("rc-service --user whisrs status"),
            Self::None => None,
        }
    }

    /// Fallback for when there is no service manager to delegate to.
    pub fn manual_start_hint(self) -> &'static str {
        "whisrsd &"
    }

    /// Directory the service definition is installed into.
    pub fn unit_dir(self) -> Option<PathBuf> {
        let config = dirs::config_dir()?;
        match self {
            Self::Systemd => Some(config.join("systemd/user")),
            Self::OpenRc => Some(config.join("rc/init.d")),
            Self::None => None,
        }
    }
}

/// Run a command, reporting whether it exited successfully.
///
/// A missing binary is indistinguishable from a failed run here, which is what
/// callers want: both mean "this service manager cannot do the thing".
fn run_ok(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run a command and capture stdout, or `None` if it could not run.
fn capture(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// [`capture`], but `None` unless the command also exited successfully.
///
/// Used where stdout alone is not evidence: `systemctl list-unit-files` still
/// prints a header when it finds nothing, so a substring match on failed output
/// could report a unit that does not exist.
fn capture_ok(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_systemd_from_runtime_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("systemd/system")).unwrap();
        assert_eq!(
            ServiceManager::detect_in(tmp.path()),
            ServiceManager::Systemd
        );
    }

    #[test]
    fn detects_openrc_from_runtime_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("openrc")).unwrap();
        assert_eq!(
            ServiceManager::detect_in(tmp.path()),
            ServiceManager::OpenRc
        );
    }

    #[test]
    fn detects_none_when_no_marker_present() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(ServiceManager::detect_in(tmp.path()), ServiceManager::None);
    }

    #[test]
    fn prefers_systemd_when_both_markers_exist() {
        // Some systemd machines ship OpenRC's runtime dir via a compat
        // package; systemd is the one actually running the session.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("systemd/system")).unwrap();
        std::fs::create_dir_all(tmp.path().join("openrc")).unwrap();
        assert_eq!(
            ServiceManager::detect_in(tmp.path()),
            ServiceManager::Systemd
        );
    }

    #[test]
    fn no_manager_reports_nothing_installed_and_no_hints() {
        let none = ServiceManager::None;
        assert!(!none.service_installed());
        assert!(!none.is_active());
        assert_eq!(none.restart(), RestartOutcome::NoService);
        assert!(none.enable_hint().is_none());
        assert!(none.restart_hint().is_none());
        assert!(none.status_hint().is_none());
    }

    #[test]
    fn hints_name_the_right_tool_for_each_manager() {
        for (mgr, tool) in [
            (ServiceManager::Systemd, "systemctl"),
            (ServiceManager::OpenRc, "rc-service"),
        ] {
            assert!(mgr.restart_hint().unwrap().contains(tool));
            assert!(mgr.status_hint().unwrap().contains(tool));
        }
        // Enabling under OpenRC goes through rc-update, not rc-service.
        assert!(ServiceManager::OpenRc
            .enable_hint()
            .unwrap()
            .contains("rc-update"));
        assert!(ServiceManager::Systemd
            .enable_hint()
            .unwrap()
            .contains("systemctl"));
    }
}
