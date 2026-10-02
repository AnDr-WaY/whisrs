//! Repair a WAV file whose writer never finalized it (crash, kill, power loss).
//!
//! A WAV header records the RIFF and `data` chunk sizes. If the writer dies
//! between flushes, the file holds more sample bytes than the header claims
//! (or a trailing partial frame). [`repair_wav`] rewrites both sizes from the
//! actual file length and trims any partial frame, without reading the
//! samples, so it stays cheap on multi-hour recordings.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{bail, Context};

/// Result of [`repair_wav`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepairOutcome {
    /// Frames (samples per channel) in the repaired file.
    pub samples: u64,
    /// Whether any byte of the file was modified.
    pub changed: bool,
}

fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) -> anyhow::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)?;
    Ok(())
}

fn write_u32_at(file: &mut File, offset: u64, value: u32) -> anyhow::Result<bool> {
    let mut cur = [0u8; 4];
    read_exact_at(file, offset, &mut cur)?;
    if u32::from_le_bytes(cur) == value {
        return Ok(false);
    }
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&value.to_le_bytes())?;
    Ok(true)
}

/// Fix the RIFF and `data` sizes of `path` to match its real length.
///
/// The `data` chunk is assumed to be the last chunk (true for files written
/// by `hound` and by the meeting recorder): everything after its header up
/// to the last whole frame is counted as samples, and a trailing partial
/// frame is truncated. A file whose header is complete but has no sample
/// bytes repairs to `samples: 0`. A file cut inside the header, or that is
/// not a PCM RIFF/WAVE file, is an error.
pub fn repair_wav(path: &Path) -> anyhow::Result<RepairOutcome> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let file_len = file.metadata()?.len();

    let mut riff = [0u8; 12];
    read_exact_at(&mut file, 0, &mut riff)
        .with_context(|| format!("{}: too short for a WAV header", path.display()))?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        bail!("{}: not a RIFF/WAVE file", path.display());
    }

    // Walk chunks until `data`, remembering `fmt `'s block_align.
    let mut pos: u64 = 12;
    let mut block_align: Option<u16> = None;
    let data_offset = loop {
        let mut header = [0u8; 8];
        read_exact_at(&mut file, pos, &mut header).with_context(|| {
            format!("{}: truncated before the data chunk header", path.display())
        })?;
        let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as u64;
        match &header[0..4] {
            b"data" => break pos,
            b"fmt " => {
                let mut fmt = [0u8; 16];
                read_exact_at(&mut file, pos + 8, &mut fmt)
                    .with_context(|| format!("{}: truncated fmt chunk", path.display()))?;
                block_align = Some(u16::from_le_bytes([fmt[12], fmt[13]]));
            }
            _ => {}
        }
        // Chunks are padded to an even size.
        pos += 8 + size + (size & 1);
    };

    let block_align = match block_align {
        Some(0) => bail!("{}: fmt chunk has block_align 0", path.display()),
        Some(b) => b as u64,
        None => bail!("{}: no fmt chunk before data", path.display()),
    };

    let data_start = data_offset + 8;
    // RIFF size is data_offset + len and must fit in a u32.
    let max_len = (u32::MAX as u64).saturating_sub(data_offset);
    let available = file_len.saturating_sub(data_start).min(max_len);
    let len = available / block_align * block_align;

    let mut changed = write_u32_at(&mut file, data_offset + 4, len as u32)?;
    changed |= write_u32_at(&mut file, 4, (data_offset + len) as u32)?;
    let new_len = data_start + len;
    if file_len != new_len {
        file.set_len(new_len)?;
        changed = true;
    }
    file.sync_all()?;
    drop(file);

    let samples = len / block_align;
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("{}: still unreadable after repair", path.display()))?;
    let read_back = reader.duration() as u64;
    if read_back != samples {
        bail!(
            "{}: repaired to {samples} frames but hound reads {read_back}",
            path.display()
        );
    }
    Ok(RepairOutcome { samples, changed })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufWriter;

    fn spec() -> hound::WavSpec {
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        }
    }

    /// Write `flushed` samples through hound and flush (header now claims
    /// them), then append `extra` raw sample bytes as a crashed writer
    /// would, without finalizing.
    fn crashed_wav(path: &Path, flushed: usize, extra: &[u8]) {
        let mut writer = hound::WavWriter::create(path, spec()).unwrap();
        for i in 0..flushed {
            writer.write_sample(i as i16).unwrap();
        }
        writer.flush().unwrap();
        // Simulate the abort: never run finalize/Drop.
        std::mem::forget(writer);
        let mut f = OpenOptions::new().append(true).open(path).unwrap();
        f.write_all(extra).unwrap();
    }

    fn raw_samples(n: usize) -> Vec<u8> {
        (0..n).flat_map(|i| (i as i16).to_le_bytes()).collect()
    }

    fn read_all(path: &Path) -> Vec<i16> {
        hound::WavReader::open(path)
            .unwrap()
            .into_samples::<i16>()
            .map(Result::unwrap)
            .collect()
    }

    #[test]
    fn recovers_samples_written_after_last_flush() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mic.wav");
        crashed_wav(&path, 1000, &raw_samples(500));

        let out = repair_wav(&path).unwrap();
        assert_eq!(
            out,
            RepairOutcome {
                samples: 1500,
                changed: true
            }
        );
        let samples = read_all(&path);
        assert_eq!(samples.len(), 1500);
        assert_eq!(samples[1200], 200);

        // Idempotent.
        assert!(!repair_wav(&path).unwrap().changed);
    }

    #[test]
    fn truncates_odd_trailing_byte() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mic.wav");
        let mut extra = raw_samples(10);
        extra.push(0x7f);
        crashed_wav(&path, 100, &extra);
        let before = std::fs::metadata(&path).unwrap().len();

        let out = repair_wav(&path).unwrap();
        assert_eq!(out.samples, 110);
        assert!(out.changed);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), before - 1);
        assert_eq!(read_all(&path).len(), 110);
    }

    #[test]
    fn header_only_file_has_zero_samples() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mic.wav");
        crashed_wav(&path, 0, &[]);
        let out = repair_wav(&path).unwrap();
        assert_eq!(out.samples, 0);
        assert!(read_all(&path).is_empty());
    }

    #[test]
    fn valid_file_is_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ok.wav");
        let mut writer = hound::WavWriter::create(&path, spec()).unwrap();
        for i in 0..321 {
            writer.write_sample(i as i16).unwrap();
        }
        writer.finalize().unwrap();
        let bytes = std::fs::read(&path).unwrap();

        let out = repair_wav(&path).unwrap();
        assert_eq!(
            out,
            RepairOutcome {
                samples: 321,
                changed: false
            }
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn rejects_non_wav_and_truncated_header() {
        let tmp = tempfile::tempdir().unwrap();
        let junk = tmp.path().join("junk.wav");
        std::fs::write(&junk, b"definitely not a wav file at all").unwrap();
        assert!(repair_wav(&junk).is_err());

        let short = tmp.path().join("short.wav");
        let mut writer =
            hound::WavWriter::new(BufWriter::new(File::create(&short).unwrap()), spec()).unwrap();
        writer.write_sample(1i16).unwrap();
        writer.finalize().unwrap();
        // Cut inside the fmt chunk.
        let f = OpenOptions::new().write(true).open(&short).unwrap();
        f.set_len(20).unwrap();
        assert!(repair_wav(&short).is_err());

        assert!(repair_wav(&tmp.path().join("missing.wav")).is_err());
    }
}
