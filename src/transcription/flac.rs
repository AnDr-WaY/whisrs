//! FLAC uploads for meeting chunks.
//!
//! Meeting chunks reach the backends as 16 kHz mono i16 WAV. FLAC is
//! lossless, so a cloud service decodes the exact samples the WAV holds (same
//! transcript, same billed duration) from less than half the bytes. Only the
//! native-timestamp `transcribe_segments` of Deepgram, Groq and OpenAI send
//! it: dictation, the phrase fallback, asr-sidecar and local-whisper keep WAV.

use std::borrow::Cow;

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, Stream};
use flacenc::error::Verify;
use flacenc::source::{Context, Fill, FrameBuf};
use tracing::{debug, warn};

use super::{decode_wav_samples, SEGMENT_SAMPLE_RATE};

/// Channels of a meeting chunk.
const CHANNELS: usize = 1;

/// Bits per sample of a meeting chunk.
const BITS_PER_SAMPLE: usize = 16;

/// How an upload is labelled on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UploadFormat {
    /// `Content-Type` of a raw body (Deepgram) or of the multipart file part.
    pub mime: &'static str,
    /// File name of the multipart file part.
    pub file_name: &'static str,
}

impl UploadFormat {
    pub(crate) const WAV: Self = Self {
        mime: "audio/wav",
        file_name: "audio.wav",
    };
    pub(crate) const FLAC: Self = Self {
        mime: "audio/flac",
        file_name: "audio.flac",
    };
}

/// Audio ready to send, with the format it is in.
pub(crate) struct Upload<'a> {
    pub bytes: Cow<'a, [u8]>,
    pub format: UploadFormat,
}

/// A meeting chunk as it goes on the wire: FLAC, or the WAV unchanged when
/// encoding fails for any reason. A meeting never fails over compression.
///
/// The encoder runs on the blocking pool, so a one-hour Deepgram chunk does
/// not stall the daemon's runtime. Empty audio is passed through for the
/// backend's own check to reject.
pub(crate) async fn meeting_upload(wav: &[u8]) -> Upload<'_> {
    let unchanged = Upload {
        bytes: Cow::Borrowed(wav),
        format: UploadFormat::WAV,
    };
    if wav.is_empty() {
        return unchanged;
    }

    let owned = wav.to_vec();
    let started = std::time::Instant::now();
    let encoded = tokio::task::spawn_blocking(move || {
        let samples = decode_wav_samples(&owned)?;
        drop(owned);
        encode_flac(&samples)
    })
    .await;

    match encoded {
        Ok(Ok(flac)) => {
            debug!(
                "encoded {} bytes of WAV as {} bytes of FLAC in {:.2?}",
                wav.len(),
                flac.len(),
                started.elapsed()
            );
            Upload {
                bytes: Cow::Owned(flac),
                format: UploadFormat::FLAC,
            }
        }
        Ok(Err(e)) => {
            warn!("FLAC encoding failed, uploading WAV instead: {e:#}");
            unchanged
        }
        Err(e) => {
            warn!("FLAC encoder did not finish, uploading WAV instead: {e}");
            unchanged
        }
    }
}

/// Encode 16 kHz mono samples as a FLAC stream with flacenc's defaults.
///
/// Frame by frame, each written out as soon as it is encoded. flacenc's
/// one-call `encode_with_fixed_block_size` keeps every frame's residuals
/// (8 bytes per sample) until the end, which is over 400 MB for an hour of
/// audio. The frames are byte-identical to its output; only the STREAMINFO
/// minimum block size differs (see below).
pub(crate) fn encode_flac(samples: &[i16]) -> anyhow::Result<Vec<u8>> {
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, e)| flac_error(e))?;
    let block_size = config.block_size;

    let mut stream =
        Stream::new(SEGMENT_SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE).map_err(flac_error)?;
    stream
        .stream_info_mut()
        .set_block_sizes(block_size, block_size)
        .map_err(flac_error)?;

    let mut frame_buf = FrameBuf::with_size(CHANNELS, block_size).map_err(flac_error)?;
    let mut context = Context::new(BITS_PER_SAMPLE, CHANNELS);
    // Room for 8 bits a sample (a recorded meeting took 6.8), so this rarely
    // regrows.
    let mut frames = ByteSink::with_capacity(samples.len() * 8);
    let mut block = Vec::with_capacity(block_size);
    for chunk in samples.chunks(block_size) {
        block.clear();
        block.extend(chunk.iter().map(|&s| i32::from(s)));
        frame_buf.fill_interleaved(&block).map_err(flac_error)?;
        context.fill_interleaved(&block).map_err(flac_error)?;
        let frame_number = context
            .current_frame_number()
            .ok_or_else(|| anyhow::anyhow!("FLAC encoder: no frame was read"))?;
        let frame = flacenc::encode_fixed_size_frame(
            &config,
            &frame_buf,
            frame_number,
            stream.stream_info(),
        )
        .map_err(flac_error)?;
        stream.stream_info_mut().update_frame_info(&frame);
        frame.write(&mut frames).map_err(flac_error)?;
    }

    let info = stream.stream_info_mut();
    info.set_md5_digest(&context.md5_digest());
    info.set_total_samples(context.total_samples());
    // flacenc lowers the minimum to the short last block, which libFLAC
    // reads as a variable block size (and warns) and claxon rejects below
    // 16. The reference encoder writes the fixed size for both.
    info.set_block_sizes(block_size, block_size)
        .map_err(flac_error)?;

    // The stream holds no frames, so this writes only `fLaC` and STREAMINFO.
    let mut header = ByteSink::new();
    stream.write(&mut header).map_err(flac_error)?;
    let mut out = header.into_inner();
    out.extend_from_slice(frames.as_slice());
    Ok(out)
}

/// flacenc's errors are not `Send` (some hold an `Rc`), so keep the message.
fn flac_error(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("FLAC encoder: {e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a FLAC stream with claxon, checking it is 16 kHz mono 16-bit.
    fn decode(flac: &[u8]) -> Vec<i16> {
        let mut reader = claxon::FlacReader::new(flac).expect("valid FLAC");
        let info = reader.streaminfo();
        assert_eq!(info.sample_rate, 16_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);
        let samples: Vec<i16> = reader
            .samples()
            .map(|s| i16::try_from(s.unwrap()).unwrap())
            .collect();
        // Zero total samples means "unknown" in STREAMINFO; claxon says None.
        assert_eq!(info.samples.unwrap_or(0), samples.len() as u64);
        samples
    }

    /// Runs of 3000 samples: silence, a tone, alternating full-scale
    /// extremes, white noise.
    fn mixed(len: usize) -> Vec<i16> {
        let mut state = 0x2545_f491_u32;
        (0..len)
            .map(|i| match (i / 3000) % 4 {
                0 => 0,
                1 => ((i as f64 * 0.17).sin() * 9000.0) as i16,
                2 => {
                    if i % 2 == 0 {
                        i16::MAX
                    } else {
                        i16::MIN
                    }
                }
                _ => {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as i16
                }
            })
            .collect()
    }

    #[test]
    fn flac_round_trips_exactly() {
        // Lengths around flacenc's 4096-sample block: odd tails, one short
        // block, a single sample.
        for len in [4096 * 3 + 1234, 4096, 4097, 4095, 777, 16, 1] {
            let samples = mixed(len);
            let flac = encode_flac(&samples).unwrap();
            assert!(flac.starts_with(b"fLaC"), "len {len}");
            assert_eq!(decode(&flac), samples, "len {len}");
        }
    }

    #[test]
    fn flac_round_trips_silence_and_extremes() {
        for samples in [
            vec![0i16; 16_000],
            vec![i16::MAX; 5000],
            vec![i16::MIN; 5000],
            (0..10_000)
                .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
                .collect(),
        ] {
            assert_eq!(decode(&encode_flac(&samples).unwrap()), samples);
        }
    }

    #[test]
    fn empty_audio_encodes_to_an_empty_stream() {
        let flac = encode_flac(&[]).unwrap();
        assert!(flac.starts_with(b"fLaC"));
        assert!(decode(&flac).is_empty());
    }

    #[test]
    fn frame_by_frame_matches_the_one_call_encoder() {
        let samples = mixed(4096 * 5 + 321);
        let config = flacenc::config::Encoder::default().into_verified().unwrap();
        let wide: Vec<i32> = samples.iter().map(|&s| i32::from(s)).collect();
        let source = flacenc::source::MemSource::from_samples(
            &wide,
            CHANNELS,
            BITS_PER_SAMPLE,
            SEGMENT_SAMPLE_RATE,
        );
        let mut stream =
            flacenc::encode_with_fixed_block_size(&config, source, config.block_size).unwrap();
        // The one deliberate difference: the fixed block size as the minimum.
        assert_eq!(stream.stream_info().min_block_size(), 321);
        stream
            .stream_info_mut()
            .set_block_sizes(config.block_size, config.block_size)
            .unwrap();
        let mut sink = ByteSink::new();
        stream.write(&mut sink).unwrap();

        assert_eq!(encode_flac(&samples).unwrap(), sink.into_inner());
    }

    #[test]
    fn speech_band_audio_compresses() {
        let samples: Vec<i16> = (0..160_000)
            .map(|i| ((i as f64 * 0.05).sin() * 3000.0) as i16)
            .collect();
        let wav = crate::audio::wav::encode_wav(&samples).unwrap();
        let flac = encode_flac(&samples).unwrap();
        assert!(
            flac.len() * 2 < wav.len(),
            "{} vs {}",
            flac.len(),
            wav.len()
        );
    }

    #[tokio::test]
    async fn meeting_upload_is_flac() {
        let samples = mixed(20_000);
        let wav = crate::audio::wav::encode_wav(&samples).unwrap();
        let upload = meeting_upload(&wav).await;
        assert_eq!(upload.format, UploadFormat::FLAC);
        assert_eq!(decode(&upload.bytes), samples);
    }

    #[tokio::test]
    async fn meeting_upload_falls_back_to_the_wav() {
        let not_wav = b"RIFF but not really a WAV".to_vec();
        let upload = meeting_upload(&not_wav).await;
        assert_eq!(upload.format, UploadFormat::WAV);
        assert_eq!(&*upload.bytes, &not_wav[..]);

        let upload = meeting_upload(&[]).await;
        assert_eq!(upload.format, UploadFormat::WAV);
        assert!(upload.bytes.is_empty());
    }

    /// Encode a real recording and report time, size and peak memory:
    ///
    /// ```text
    /// WHISRS_FLAC_BENCH_WAV=/path/to/copy.wav cargo test --release --lib \
    ///     transcription::flac::tests::real_recording -- --ignored --nocapture
    /// ```
    ///
    /// Point it at a copy, never at a file inside the meetings folder.
    #[tokio::test]
    #[ignore = "needs WHISRS_FLAC_BENCH_WAV"]
    async fn real_recording() {
        let path = std::env::var("WHISRS_FLAC_BENCH_WAV").expect("WHISRS_FLAC_BENCH_WAV");
        let wav = std::fs::read(&path).unwrap();
        let started = std::time::Instant::now();
        let upload = meeting_upload(&wav).await;
        let elapsed = started.elapsed();
        let peak = std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find(|l| l.starts_with("VmHWM:"))
            .map(str::to_string)
            .unwrap_or_default();
        assert_eq!(upload.format, UploadFormat::FLAC);
        println!(
            "{} bytes of WAV -> {} bytes of FLAC ({:.1}%) in {:.2?}; {peak}",
            wav.len(),
            upload.bytes.len(),
            upload.bytes.len() as f64 * 100.0 / wav.len() as f64,
            elapsed
        );
        assert_eq!(decode(&upload.bytes), decode_wav_samples(&wav).unwrap());
    }
}
