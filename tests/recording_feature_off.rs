//! `whisrs recording` in a build without the `recording` feature: the error
//! on stderr and exit code 1, before any daemon is asked. Empty in a build
//! with the feature.
#![cfg(not(feature = "recording"))]

use std::process::Command;

const EXPECTED: &str = "error: this whisrs was built without the recording feature, so \
    `whisrs recording` is not available. Use a release build, or rebuild with `cargo install \
    whisrs --features recording`.";

#[test]
fn recording_exits_1_and_says_how_to_get_it() {
    // A socket folder with no daemon in it, so a regression that sends the
    // command fails here ("whisrsd is not running") instead of reaching a
    // daemon on this machine.
    let runtime = tempfile::tempdir().expect("temp dir");
    for args in [
        &["recording"][..],
        &[
            "recording",
            "start",
            "-t",
            "Weekly sync",
            "--with",
            "Sarah, Omar",
        ],
        &["recording", "toggle"],
        &["recording", "list", "--json"],
        &["recording", "show", "last"],
        &["recording", "summarize", "last"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_whisrs"))
            .args(args)
            .env("XDG_RUNTIME_DIR", runtime.path())
            .output()
            .expect("run whisrs");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{args:?}: {stderr}");
        assert_eq!(stderr.trim_end(), EXPECTED, "{args:?}");
        assert!(out.stdout.is_empty(), "{args:?}: nothing on stdout");
    }
}
