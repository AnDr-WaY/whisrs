//! Track writer for the recording feature: writes captured audio chunks to
//! WAV files.
//!
//! One writer per track (mic, system). The header is rewritten every
//! `flush_every`, so a crash leaves a file that [`super::wav_repair`] can fix
//! with at most that much audio missing from the header's length.

use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Context;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::debug;

use crate::audio::capture::SAMPLE_RATE;
use crate::audio::AudioChunk;

/// What a track writer recorded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackStats {
    /// Samples written (16 kHz mono).
    pub samples: u64,
    /// When the first non-empty chunk arrived, to align the tracks.
    pub first_chunk_at: Option<Instant>,
}

/// Whether a track writer failed because the disk is full (or the user's
/// quota is): `ENOSPC` or `EDQUOT`, as the plain I/O error or inside
/// hound's, which does not expose it as a `source`.
pub fn is_disk_full(err: &anyhow::Error) -> bool {
    let full = |io: &std::io::Error| {
        matches!(io.raw_os_error(), Some(libc::ENOSPC) | Some(libc::EDQUOT))
            || io.kind() == std::io::ErrorKind::StorageFull
    };
    err.chain().any(|cause| {
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            return full(io);
        }
        matches!(cause.downcast_ref::<hound::Error>(), Some(hound::Error::IoError(io)) if full(io))
    })
}

/// The format of every recording track: 16 kHz mono 16-bit PCM.
fn track_spec() -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

/// Write the chunks from `rx` to a WAV file at `path` (created 0600) until
/// the channel closes, then finalize it. Runs on the blocking pool.
pub fn spawn_track_writer(
    rx: mpsc::UnboundedReceiver<AudioChunk>,
    path: PathBuf,
    flush_every: Duration,
) -> JoinHandle<anyhow::Result<TrackStats>> {
    tokio::task::spawn_blocking(move || write_track(rx, &path, flush_every))
}

fn create_track_file(path: &Path) -> anyhow::Result<File> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    // `mode` only applies to a new file.
    use std::os::unix::fs::PermissionsExt;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("failed to set permissions on {}", path.display()))?;
    Ok(file)
}

fn write_track(
    mut rx: mpsc::UnboundedReceiver<AudioChunk>,
    path: &Path,
    flush_every: Duration,
) -> anyhow::Result<TrackStats> {
    let file = create_track_file(path)?;
    let mut writer = hound::WavWriter::new(BufWriter::new(file), track_spec())
        .with_context(|| format!("failed to start WAV {}", path.display()))?;
    let mut stats = TrackStats::default();
    let mut last_flush = Instant::now();

    while let Some(chunk) = rx.blocking_recv() {
        if chunk.is_empty() {
            continue;
        }
        stats.first_chunk_at.get_or_insert_with(Instant::now);
        for &sample in &chunk {
            writer
                .write_sample(sample)
                .with_context(|| format!("failed to write {}", path.display()))?;
        }
        stats.samples += chunk.len() as u64;
        if last_flush.elapsed() >= flush_every {
            // Rewrites the header lengths, so the file is valid up to here.
            writer
                .flush()
                .with_context(|| format!("failed to flush {}", path.display()))?;
            last_flush = Instant::now();
        }
    }

    writer
        .finalize()
        .with_context(|| format!("failed to finalize {}", path.display()))?;
    debug!("wrote {} samples to {}", stats.samples, path.display());
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn read_samples(path: &Path) -> Vec<i16> {
        let mut reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec(), track_spec());
        reader.samples::<i16>().map(|s| s.unwrap()).collect()
    }

    #[tokio::test]
    async fn writes_all_chunks_and_finalizes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        let (tx, rx) = mpsc::unbounded_channel();
        let task = spawn_track_writer(rx, path.clone(), Duration::from_secs(2));

        let mut expected = Vec::new();
        for i in 0..10i16 {
            let chunk: Vec<i16> = (0..160).map(|j| i * 1000 + j).collect();
            expected.extend_from_slice(&chunk);
            tx.send(chunk).unwrap();
        }
        // An empty chunk neither counts nor sets the first-chunk time.
        tx.send(Vec::new()).unwrap();
        drop(tx);

        let stats = task.await.unwrap().unwrap();
        assert_eq!(stats.samples, 1600);
        assert!(stats.first_chunk_at.is_some());
        assert_eq!(read_samples(&path), expected);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[tokio::test]
    async fn empty_track_is_a_valid_empty_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("system.wav");
        let (tx, rx) = mpsc::unbounded_channel::<AudioChunk>();
        tx.send(Vec::new()).unwrap();
        drop(tx);
        let stats = spawn_track_writer(rx, path.clone(), Duration::from_secs(2))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stats, TrackStats::default());
        assert!(read_samples(&path).is_empty());
    }

    #[tokio::test]
    async fn flush_leaves_a_valid_header_mid_stream() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        let (tx, rx) = mpsc::unbounded_channel();
        // Flush after every chunk.
        let task = spawn_track_writer(rx, path.clone(), Duration::ZERO);

        tx.send(vec![7i16; 1600]).unwrap();
        // Wait until the flushed header reports the chunk, without closing
        // the channel: this is what a crash mid-recording leaves on disk.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(reader) = hound::WavReader::open(&path) {
                if reader.len() == 1600 {
                    break;
                }
            }
            assert!(Instant::now() < deadline, "header never flushed");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(read_samples(&path), vec![7i16; 1600]);
        let repaired = crate::recording::wav_repair::repair_wav(&path).unwrap();
        assert_eq!(repaired.samples, 1600);
        assert!(!repaired.changed);

        tx.send(vec![1i16; 400]).unwrap();
        drop(tx);
        let stats = task.await.unwrap().unwrap();
        assert_eq!(stats.samples, 2000);
        assert_eq!(read_samples(&path).len(), 2000);
    }

    #[tokio::test]
    async fn unflushed_tail_is_repairable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        let (tx, rx) = mpsc::unbounded_channel();
        // Never flush on cadence: only the header from creation is on disk
        // until the BufWriter spills.
        let task = spawn_track_writer(rx, path.clone(), Duration::from_secs(3600));
        // Enough to overflow the 8 KiB BufWriter several times.
        tx.send(vec![3i16; 16_000]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) < 16_000 {
            assert!(Instant::now() < deadline, "data never reached the file");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        // Simulate a crash: copy the file as it is now, before finalize.
        let crashed = dir.path().join("crashed.wav");
        std::fs::copy(&path, &crashed).unwrap();
        let repaired = crate::recording::wav_repair::repair_wav(&crashed).unwrap();
        assert!(repaired.changed);
        assert!(repaired.samples > 0 && repaired.samples <= 16_000);
        assert_eq!(read_samples(&crashed).len() as u64, repaired.samples);

        drop(tx);
        assert_eq!(task.await.unwrap().unwrap().samples, 16_000);
    }

    #[test]
    fn a_full_disk_is_recognized() {
        let enospc = || std::io::Error::from_raw_os_error(libc::ENOSPC);
        let write =
            anyhow::Error::from(hound::Error::IoError(enospc())).context("failed to write mic.wav");
        assert!(is_disk_full(&write));
        let finalize = anyhow::Error::from(enospc()).context("failed to finalize mic.wav");
        assert!(is_disk_full(&finalize));
        let quota = anyhow::Error::from(std::io::Error::from_raw_os_error(libc::EDQUOT));
        assert!(is_disk_full(&quota));
        let other = anyhow::Error::from(hound::Error::IoError(std::io::Error::from_raw_os_error(
            libc::EIO,
        )));
        assert!(!is_disk_full(&other));
        assert!(!is_disk_full(&anyhow::anyhow!("no space left")));
    }

    #[tokio::test]
    async fn unwritable_path_is_an_error() {
        let (_tx, rx) = mpsc::unbounded_channel::<AudioChunk>();
        let err = spawn_track_writer(
            rx,
            PathBuf::from("/nonexistent-whisrs-dir/mic.wav"),
            Duration::from_secs(2),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(err.to_string().contains("failed to create"), "{err:#}");
    }
}
