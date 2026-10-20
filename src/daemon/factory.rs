//! Dictation backend construction: `[general] backend` through the shared
//! factory in [`whisrs::transcription::factory`], which recordings also use.

use std::sync::Arc;

use whisrs::transcription::factory::{create_backend_named, model_for_backend};
use whisrs::transcription::TranscriptionBackend;
use whisrs::Config;

pub(crate) use whisrs::transcription::factory::resolve_tts_api_key;

/// The dictation backend: the one `[general] backend` names.
pub(crate) fn create_backend(config: &Config) -> Arc<dyn TranscriptionBackend> {
    create_backend_named(&config.general.backend, config)
}

/// The model dictation requests carry: `[general] backend`'s.
pub(crate) fn get_model_for_backend(config: &Config) -> String {
    model_for_backend(&config.general.backend, config)
}
