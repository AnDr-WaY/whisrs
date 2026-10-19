//! On-disk recording store: one directory per recording under the recordings root.
//!
//! Layout: `<root>/<id>/recording.json`, `<root>/<id>/audio/{mic,system}.wav`
//! and `<root>/<id>/<id>.md`. Directories are 0700 and `recording.json` is
//! 0600, since both hold private conversation content.

use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use chrono::{DateTime, Local};
use tracing::{debug, warn};

use super::{RecordingMeta, RecordingState, TrackKind};

const META_FILE: &str = "recording.json";
const META_TMP: &str = ".recording.json.tmp";
const AUDIO_DIR: &str = "audio";

/// Per-chunk results of finished requests, under the recording folder, so a
/// retry or a recovery after a crash resumes instead of starting over.
/// Dropped once the transcript is written, except the voices a diarization
/// sidecar found (`diarization-*.json`), which a later retry reuses; a failed
/// recording keeps all of it until its audio is deleted, since a retry needs
/// both.
pub const CACHE_DIR: &str = ".cache";
const SLUG_MAX: usize = 40;

/// The default recordings root, ignoring any `[recording] dir` override.
pub fn recordings_dir() -> PathBuf {
    recordings_dir_from(None)
}

/// The recordings root: `[recording] dir` (with a leading `~/` expanded) or
/// `$XDG_DATA_HOME/whisrs/recordings`.
pub fn recordings_dir_from(cfg_dir: Option<&str>) -> PathBuf {
    if let Some(dir) = cfg_dir.map(str::trim).filter(|d| !d.is_empty()) {
        return expand_tilde(dir);
    }
    dirs::data_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
        .unwrap_or_else(std::env::temp_dir)
        .join("whisrs")
        .join("recordings")
}

fn expand_tilde(dir: &str) -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        if dir == "~" {
            return home;
        }
        if let Some(rest) = dir.strip_prefix("~/") {
            return home.join(rest);
        }
    }
    PathBuf::from(dir)
}

/// Turn a title into an id suffix: lowercase ASCII alphanumerics joined by
/// single dashes, at most [`SLUG_MAX`] chars. Empty if nothing survives.
fn slugify(title: &str) -> String {
    let mut slug = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.truncate(SLUG_MAX);
    slug.trim_end_matches('-').to_string()
}

fn base_id(started: DateTime<Local>, title: Option<&str>) -> String {
    let stamp = started.format("%Y-%m-%d_%H%M").to_string();
    match title.map(slugify).filter(|s| !s.is_empty()) {
        Some(slug) => format!("{stamp}_{slug}"),
        None => stamp,
    }
}

fn private_dir_builder() -> fs::DirBuilder {
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    builder
}

/// Create a new recording directory (with `audio/`) and return its id and path.
///
/// The id is `YYYY-MM-DD_HHMM`, plus `_<slug>` when a title is given; a
/// collision appends `-2`, `-3`, ... The directory is claimed with
/// `create_dir`, so two concurrent callers never get the same one. The root
/// is created 0700 if missing; an existing root keeps its permissions.
pub fn create(
    root: &Path,
    started: DateTime<Local>,
    title: Option<&str>,
) -> anyhow::Result<(String, PathBuf)> {
    if !root.exists() {
        private_dir_builder()
            .recursive(true)
            .create(root)
            .with_context(|| format!("failed to create recordings dir {}", root.display()))?;
        // `mode` is masked by the umask; set it explicitly.
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
    }

    let base = base_id(started, title);
    for n in 1u32.. {
        let id = if n == 1 {
            base.clone()
        } else {
            format!("{base}-{n}")
        };
        let dir = root.join(&id);
        match private_dir_builder().create(&dir) {
            Ok(()) => {
                fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
                let audio = dir.join(AUDIO_DIR);
                private_dir_builder()
                    .create(&audio)
                    .with_context(|| format!("failed to create {}", audio.display()))?;
                fs::set_permissions(&audio, fs::Permissions::from_mode(0o700))?;
                return Ok((id, dir));
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).with_context(|| format!("failed to create {}", dir.display())),
        }
    }
    unreachable!("u32 range exhausted while picking a recording id")
}

/// Read `recording.json` from a recording directory.
pub fn read_meta(dir: &Path) -> anyhow::Result<RecordingMeta> {
    let path = dir.join(META_FILE);
    let data = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("failed to parse {}", path.display()))
}

/// Write `recording.json` atomically (temp file in the same dir, fsync,
/// rename), mode 0600.
pub fn write_meta(dir: &Path, meta: &RecordingMeta) -> anyhow::Result<()> {
    let tmp = dir.join(META_TMP);
    let path = dir.join(META_FILE);
    let json = serde_json::to_vec_pretty(meta)?;
    {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .with_context(|| format!("failed to open {}", tmp.display()))?;
        // A stale tmp from a crash may have other permissions.
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
        file.write_all(&json)?;
        file.sync_all()?;
    }
    fs::rename(&tmp, &path)
        .with_context(|| format!("failed to rename {} into place", tmp.display()))?;
    // Persist the rename itself; failure here is not worth failing the write.
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// All recordings under `root`, newest first by `started_at`. Directories
/// without a readable `recording.json` are skipped. A missing root is empty.
pub fn list(root: &Path) -> anyhow::Result<Vec<(PathBuf, RecordingMeta)>> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("failed to read {}", root.display())),
    };
    let mut recordings = Vec::new();
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let dir = entry.path();
        match read_meta(&dir) {
            Ok(meta) => recordings.push((dir, meta)),
            Err(e) => debug!("skipping {}: {e:#}", dir.display()),
        }
    }
    recordings.sort_by(|(_, a), (_, b)| {
        b.started_at
            .cmp(&a.started_at)
            .then_with(|| b.id.cmp(&a.id))
    });
    Ok(recordings)
}

/// Find a recording by `"last"` (the newest), an exact id, or a unique id prefix.
pub fn resolve(root: &Path, query: &str) -> anyhow::Result<PathBuf> {
    let query = query.trim();
    let recordings = list(root)?;
    if recordings.is_empty() {
        bail!("no recordings found in {}", root.display());
    }
    if query == "last" {
        return Ok(recordings[0].0.clone());
    }
    if let Some((dir, _)) = recordings.iter().find(|(_, m)| m.id == query) {
        return Ok(dir.clone());
    }
    let matches: Vec<&(PathBuf, RecordingMeta)> = if query.is_empty() {
        Vec::new()
    } else {
        recordings
            .iter()
            .filter(|(_, m)| m.id.starts_with(query))
            .collect()
    };
    match matches.as_slice() {
        [] => bail!("no recording matches '{query}'"),
        [(dir, _)] => Ok(dir.clone()),
        many => {
            let ids: Vec<&str> = many.iter().map(|(_, m)| m.id.as_str()).collect();
            bail!(
                "'{query}' matches {} recordings: {}",
                ids.len(),
                ids.join(", ")
            )
        }
    }
}

/// Path of the rendered transcript: `<dir>/<id>.md`.
pub fn transcript_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.md"))
}

/// Path of a track's WAV file: `<dir>/audio/{mic,system}.wav`.
pub fn audio_path(dir: &Path, kind: TrackKind) -> PathBuf {
    dir.join(AUDIO_DIR)
        .join(format!("{}.wav", kind.file_stem()))
}

/// Bytes available to an unprivileged user on the filesystem holding
/// `path`, or its nearest existing ancestor (the recordings root may not
/// exist yet).
pub fn free_space(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let existing = path
        .ancestors()
        .find(|p| p.exists())
        .unwrap_or_else(|| Path::new("/"));
    let c_path = std::ffi::CString::new(existing.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(ErrorKind::InvalidInput, e))?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` a valid
    // out-pointer for the duration of the call.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Both fields are 64-bit on Linux but 32-bit on some 32-bit targets.
    #[allow(clippy::useless_conversion)]
    let (blocks, size) = (u64::from(stat.f_bavail), u64::from(stat.f_frsize));
    Ok(blocks.saturating_mul(size))
}

/// Recordings left mid-capture or mid-processing (e.g. by a crash).
pub fn interrupted(root: &Path) -> Vec<PathBuf> {
    match list(root) {
        Ok(recordings) => recordings
            .into_iter()
            .filter(|(_, m)| {
                matches!(
                    m.status,
                    RecordingState::Recording | RecordingState::Processing
                )
            })
            .map(|(dir, _)| dir)
            .collect(),
        Err(e) => {
            warn!("failed to scan recordings for interrupted ones: {e:#}");
            Vec::new()
        }
    }
}

/// Recordings waiting for the network, oldest first.
pub fn waiting(root: &Path) -> Vec<(PathBuf, RecordingMeta)> {
    match list(root) {
        Ok(recordings) => recordings
            .into_iter()
            .rev()
            .filter(|(_, m)| m.status == RecordingState::Waiting)
            .collect(),
        Err(e) => {
            warn!("failed to scan recordings for ones waiting for the network: {e:#}");
            Vec::new()
        }
    }
}

/// Delete the `audio/` dir of `Done` recordings that ended at least `days`
/// days before `now` (by `ended_at`, else `started_at`), mark them
/// `audio_deleted`, and return their dirs. `0` sweeps every `Done` recording;
/// a negative value keeps audio forever. `Failed`, `Waiting`, `Recording` and
/// `Processing` recordings are never touched, so their audio stays retryable.
pub fn sweep_retention(root: &Path, days: i64, now: DateTime<Local>) -> Vec<PathBuf> {
    sweep_retention_guarded(root, days, now, |_| Some(()))
}

/// [`sweep_retention`], taking `guard(id)` before a recording is touched and
/// holding it until its `recording.json` is written. A recording `guard`
/// refuses (another writer has it, a rename) is skipped until the next
/// sweep, and the meta is read again under the guard, so what that writer
/// wrote is kept.
pub fn sweep_retention_guarded<G>(
    root: &Path,
    days: i64,
    now: DateTime<Local>,
    mut guard: impl FnMut(&str) -> Option<G>,
) -> Vec<PathBuf> {
    if days < 0 {
        return Vec::new();
    }
    let recordings = match list(root) {
        Ok(recordings) => recordings,
        Err(e) => {
            warn!("failed to scan recordings for audio retention: {e:#}");
            return Vec::new();
        }
    };
    let max_age = chrono::Duration::days(days);
    let mut swept = Vec::new();
    for (dir, listed) in recordings {
        let due = |meta: &RecordingMeta| {
            let reference = meta.ended_at.unwrap_or(meta.started_at);
            meta.status == RecordingState::Done
                && !meta.audio_deleted
                && now.signed_duration_since(reference) >= max_age
        };
        if !due(&listed) {
            continue;
        }
        let Some(_held) = guard(&listed.id) else {
            debug!("recording {} is busy; its audio is swept later", listed.id);
            continue;
        };
        let mut meta = match read_meta(&dir) {
            Ok(meta) if due(&meta) => meta,
            Ok(_) => continue,
            Err(e) => {
                warn!(
                    "failed to read {} for audio retention: {e:#}",
                    dir.display()
                );
                continue;
            }
        };
        let audio = dir.join(AUDIO_DIR);
        match fs::remove_dir_all(&audio) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => {
                warn!("failed to delete {}: {e}", audio.display());
                continue;
            }
        }
        // What a done recording keeps there: a diarization sidecar's voices,
        // which go with the audio.
        let cache = dir.join(CACHE_DIR);
        match fs::remove_dir_all(&cache) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => warn!("failed to delete {}: {e}", cache.display()),
        }
        meta.audio_deleted = true;
        if let Err(e) = write_meta(&dir, &meta) {
            warn!("failed to mark audio deleted in {}: {e:#}", dir.display());
            continue;
        }
        swept.push(dir);
    }
    swept
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Local> {
        Local.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }

    fn mode(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    /// Create a recording with a meta of the given status and times.
    fn seed(
        root: &Path,
        started: DateTime<Local>,
        title: Option<&str>,
        status: RecordingState,
        ended: Option<DateTime<Local>>,
    ) -> PathBuf {
        let (id, dir) = create(root, started, title).unwrap();
        let mut meta = RecordingMeta::new(id, title.map(String::from), started);
        meta.status = status;
        meta.ended_at = ended;
        write_meta(&dir, &meta).unwrap();
        fs::write(audio_path(&dir, TrackKind::Mic), b"wav").unwrap();
        dir
    }

    #[test]
    fn slug_rules() {
        assert_eq!(
            slugify("Weekly Sync: Q4 / Planning!"),
            "weekly-sync-q4-planning"
        );
        assert_eq!(slugify("  --  "), "");
        assert_eq!(slugify("Café déjà"), "caf-d-j");
        let long = slugify(&"ab ".repeat(40));
        assert!(long.len() <= SLUG_MAX && !long.ends_with('-'), "{long}");
    }

    #[test]
    fn create_ids_and_collisions() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("recordings");
        let t = at(2026, 10, 2, 14, 5);

        let (id1, dir1) = create(&root, t, None).unwrap();
        assert_eq!(id1, "2026-10-02_1405");
        let (id2, _) = create(&root, t, None).unwrap();
        assert_eq!(id2, "2026-10-02_1405-2");
        let (id3, _) = create(&root, t, None).unwrap();
        assert_eq!(id3, "2026-10-02_1405-3");
        let (id4, _) = create(&root, t, Some("Team Standup")).unwrap();
        assert_eq!(id4, "2026-10-02_1405_team-standup");
        let (id5, _) = create(&root, t, Some("team standup")).unwrap();
        assert_eq!(id5, "2026-10-02_1405_team-standup-2");
        let (id6, _) = create(&root, t, Some("!!!")).unwrap();
        assert_eq!(id6, "2026-10-02_1405-4");

        assert!(dir1.join("audio").is_dir());
        assert_eq!(mode(&root), 0o700);
        assert_eq!(mode(&dir1), 0o700);
        assert_eq!(mode(&dir1.join("audio")), 0o700);
    }

    #[test]
    fn meta_write_read_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let (id, dir) = create(tmp.path(), at(2026, 1, 1, 9, 0), Some("x")).unwrap();
        let mut meta = RecordingMeta::new(id, Some("x".into()), at(2026, 1, 1, 9, 0));
        write_meta(&dir, &meta).unwrap();
        assert_eq!(read_meta(&dir).unwrap(), meta);
        assert_eq!(mode(&dir.join(META_FILE)), 0o600);
        assert!(!dir.join(META_TMP).exists());

        meta.status = RecordingState::Done;
        meta.errors.push("system audio unavailable".into());
        write_meta(&dir, &meta).unwrap();
        assert_eq!(read_meta(&dir).unwrap(), meta);
    }

    #[test]
    fn list_is_newest_first_and_skips_junk() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let old = seed(root, at(2026, 1, 1, 9, 0), None, RecordingState::Done, None);
        let new = seed(root, at(2026, 3, 1, 9, 0), None, RecordingState::Done, None);
        let mid = seed(root, at(2026, 2, 1, 9, 0), None, RecordingState::Done, None);
        fs::create_dir(root.join("not-a-recording")).unwrap();
        fs::write(root.join("stray.txt"), b"x").unwrap();

        let dirs: Vec<PathBuf> = list(root).unwrap().into_iter().map(|(d, _)| d).collect();
        assert_eq!(dirs, vec![new, mid, old]);
        assert!(list(&root.join("missing")).unwrap().is_empty());
    }

    #[test]
    fn resolve_queries() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert!(resolve(root, "last").is_err());

        let a = seed(
            root,
            at(2026, 1, 1, 9, 0),
            Some("alpha"),
            RecordingState::Done,
            None,
        );
        let b = seed(
            root,
            at(2026, 1, 1, 10, 0),
            Some("beta"),
            RecordingState::Done,
            None,
        );
        let c = seed(
            root,
            at(2026, 2, 5, 11, 0),
            None,
            RecordingState::Done,
            None,
        );

        assert_eq!(resolve(root, "last").unwrap(), c);
        assert_eq!(resolve(root, "2026-01-01_0900_alpha").unwrap(), a);
        assert_eq!(resolve(root, "2026-01-01_10").unwrap(), b);
        assert_eq!(resolve(root, "2026-02").unwrap(), c);

        let ambiguous = resolve(root, "2026-01-01").unwrap_err().to_string();
        assert!(ambiguous.contains("2026-01-01_0900_alpha"), "{ambiguous}");
        assert!(ambiguous.contains("2026-01-01_1000_beta"), "{ambiguous}");

        let none = resolve(root, "2025").unwrap_err().to_string();
        assert!(none.contains("no recording matches"), "{none}");
        assert!(resolve(root, "").is_err());
    }

    #[test]
    fn resolve_prefers_exact_over_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let t = at(2026, 1, 1, 9, 0);
        let first = seed(root, t, None, RecordingState::Done, None);
        let _second = seed(root, t, None, RecordingState::Done, None);
        assert_eq!(resolve(root, "2026-01-01_0900").unwrap(), first);
    }

    #[test]
    fn interrupted_finds_recording_and_processing() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let rec = seed(
            root,
            at(2026, 1, 1, 9, 0),
            None,
            RecordingState::Recording,
            None,
        );
        let proc_ = seed(
            root,
            at(2026, 1, 2, 9, 0),
            None,
            RecordingState::Processing,
            None,
        );
        seed(root, at(2026, 1, 3, 9, 0), None, RecordingState::Done, None);
        seed(
            root,
            at(2026, 1, 4, 9, 0),
            None,
            RecordingState::Failed,
            None,
        );
        let later = seed(
            root,
            at(2026, 1, 6, 9, 0),
            None,
            RecordingState::Waiting,
            None,
        );
        let earlier = seed(
            root,
            at(2026, 1, 5, 9, 0),
            None,
            RecordingState::Waiting,
            None,
        );

        // Waiting for the network is not interrupted: recovery leaves it to
        // the automatic retries.
        assert_eq!(interrupted(root), vec![proc_, rec]);
        let waiting: Vec<PathBuf> = waiting(root).into_iter().map(|(dir, _)| dir).collect();
        assert_eq!(waiting, vec![earlier, later], "oldest first");
    }

    /// The sweep skips a recording its guard refuses, and reads the meta
    /// again under the guard: what another writer wrote between the
    /// listing and the guard is kept.
    #[test]
    fn the_sweep_skips_what_is_busy_and_keeps_what_was_written() {
        let tmp = tempfile::tempdir().unwrap();
        let old = at(2026, 1, 1, 9, 0);
        let busy = seed(
            tmp.path(),
            old,
            Some("busy"),
            RecordingState::Done,
            Some(old),
        );
        let renamed = seed(
            tmp.path(),
            old,
            Some("renamed"),
            RecordingState::Done,
            Some(old),
        );
        let busy_id = read_meta(&busy).unwrap().id;
        let swept = sweep_retention_guarded(tmp.path(), 0, at(2026, 2, 1, 9, 0), |id| {
            if id == busy_id {
                return None;
            }
            // A rename lands just before the guard is taken.
            let mut meta = read_meta(&renamed).unwrap();
            meta.speaker_names
                .insert("S1".into(), crate::recording::SpeakerName::plain("Sarah"));
            write_meta(&renamed, &meta).unwrap();
            Some(())
        });
        assert_eq!(swept, vec![renamed.clone()]);
        assert!(audio_path(&busy, TrackKind::Mic).exists(), "skipped");
        assert!(!read_meta(&busy).unwrap().audio_deleted);
        let meta = read_meta(&renamed).unwrap();
        assert!(meta.audio_deleted);
        assert_eq!(meta.speaker_names["S1"].name, "Sarah", "the rename is kept");
    }

    #[test]
    fn sweep_respects_age_and_status() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let now = at(2026, 10, 10, 12, 0);
        // Ended 8 days ago, started long before: swept at 7 days.
        let old = seed(
            root,
            at(2026, 9, 1, 9, 0),
            Some("old"),
            RecordingState::Done,
            Some(at(2026, 10, 2, 12, 0)),
        );
        // Ended 3 days ago: kept at 7 days.
        let recent = seed(
            root,
            at(2026, 10, 7, 11, 0),
            Some("recent"),
            RecordingState::Done,
            Some(at(2026, 10, 7, 12, 0)),
        );
        // No ended_at: falls back to started_at (20 days ago).
        let no_end = seed(
            root,
            at(2026, 9, 20, 9, 0),
            Some("noend"),
            RecordingState::Done,
            None,
        );
        let failed = seed(
            root,
            at(2026, 1, 1, 9, 0),
            Some("f"),
            RecordingState::Failed,
            None,
        );
        let rec = seed(
            root,
            at(2026, 1, 2, 9, 0),
            Some("r"),
            RecordingState::Recording,
            None,
        );
        let proc_ = seed(
            root,
            at(2026, 1, 3, 9, 0),
            Some("p"),
            RecordingState::Processing,
            None,
        );
        let waiting = seed(
            root,
            at(2026, 1, 4, 9, 0),
            Some("w"),
            RecordingState::Waiting,
            Some(at(2026, 1, 4, 10, 0)),
        );

        assert!(sweep_retention(root, -1, now).is_empty());
        assert!(old.join("audio").exists());

        let mut swept = sweep_retention(root, 7, now);
        swept.sort();
        let mut expected = vec![old.clone(), no_end.clone()];
        expected.sort();
        assert_eq!(swept, expected);
        assert!(!old.join("audio").exists());
        assert!(read_meta(&old).unwrap().audio_deleted);
        assert!(recent.join("audio").exists());
        assert!(!read_meta(&recent).unwrap().audio_deleted);

        // Already swept recordings are not reported again.
        assert_eq!(sweep_retention(root, 0, now), vec![recent.clone()]);
        assert!(!recent.join("audio").exists());

        for dir in [&failed, &rec, &proc_, &waiting] {
            assert!(dir.join("audio").exists(), "{}", dir.display());
            assert!(!read_meta(dir).unwrap().audio_deleted);
        }
    }

    #[test]
    fn free_space_reads_the_filesystem() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(free_space(tmp.path()).unwrap() > 0);
        // A root that does not exist yet is measured at its parent.
        let missing = tmp.path().join("not/yet/created");
        assert!(free_space(&missing).unwrap() > 0);
    }

    #[test]
    fn recordings_dir_override_and_tilde() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(recordings_dir_from(Some("~/m")), home.join("m"));
        assert_eq!(recordings_dir_from(Some("/srv/m")), PathBuf::from("/srv/m"));
        assert!(recordings_dir_from(None).ends_with("whisrs/recordings"));
        assert_eq!(recordings_dir_from(Some("  ")), recordings_dir());
    }

    #[test]
    fn transcript_and_audio_paths() {
        let dir = Path::new("/m/2026-01-01_0900");
        assert_eq!(
            transcript_path(dir, "2026-01-01_0900"),
            dir.join("2026-01-01_0900.md")
        );
        assert_eq!(audio_path(dir, TrackKind::Mic), dir.join("audio/mic.wav"));
        assert_eq!(
            audio_path(dir, TrackKind::System),
            dir.join("audio/system.wav")
        );
    }
}
