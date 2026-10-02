//! Waiting out rate limits (HTTP 429/503) for long jobs such as meetings.
//!
//! One retry layer per request: [`retry_rate_limited`] retries an operation
//! that fails with a [`RetryableHttpError`], and when its budget runs out it
//! returns a [`RateLimitExhausted`], which is not retryable. So a caller that
//! wraps an operation already retried inside (a phrase backend retries each
//! phrase in [`super::segments_by_phrase`]) gives up at once instead of
//! multiplying the wait.

use std::future::Future;
use std::time::Duration;

use tracing::info;

use super::RetryableHttpError;

/// Longest one request waits out rate limits before giving up.
pub const RATE_LIMIT_BUDGET: Duration = Duration::from_secs(300);

/// First backoff without a `Retry-After`, doubled per attempt up to
/// [`BACKOFF_MAX`].
pub const BACKOFF_BASE: Duration = Duration::from_secs(2);
pub const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// A request that was still rate limited when its budget ran out. Not a
/// [`RetryableHttpError`], so an outer retry layer does not retry it again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitExhausted {
    pub status: u16,
    pub message: String,
}

impl std::fmt::Display for RateLimitExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RateLimitExhausted {}

/// How long to wait before retrying a rate-limited (429) or overloaded
/// (503) request, or `None` to give up. Honors the server's `Retry-After`;
/// otherwise backs off exponentially from [`BACKOFF_BASE`] to
/// [`BACKOFF_MAX`]. Gives up once the wait would push the total past
/// `budget`.
pub fn backoff_delay(
    attempt: u32,
    retry_after: Option<Duration>,
    waited: Duration,
    budget: Duration,
) -> Option<Duration> {
    let exponential = BACKOFF_BASE
        .saturating_mul(1u32 << attempt.min(16))
        .min(BACKOFF_MAX);
    // A zero `Retry-After` still gets a short pause, so a confused server
    // cannot make this spin.
    let delay = retry_after.map_or(exponential, |d| d.max(Duration::from_millis(500)));
    (waited.saturating_add(delay) <= budget).then_some(delay)
}

/// Why a retry gave up, in words.
fn give_up_message(retryable: &RetryableHttpError, waited: Duration, budget: Duration) -> String {
    let status = retryable.status;
    match retryable.retry_after {
        Some(asked) if waited.saturating_add(asked) > budget => {
            let mut message = format!(
                "rate limited (HTTP {status}); the server asked to wait {} s, more than the \
                 {} s budget",
                asked.as_secs_f64().ceil() as u64,
                budget.as_secs()
            );
            if !waited.is_zero() {
                message.push_str(&format!(" after {} s of retries", waited.as_secs()));
            }
            format!("{message}: {}", retryable.message)
        }
        _ => format!(
            "the backend still answered HTTP {status} after {} s of retries: {}",
            waited.as_secs(),
            retryable.message
        ),
    }
}

/// Run `op`, waiting out rate limits within `budget`.
///
/// Any error that is not a [`RetryableHttpError`] is returned as is. When
/// the budget runs out the result is a [`RateLimitExhausted`].
pub async fn retry_rate_limited<T, F, Fut>(budget: Duration, mut op: F) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    let mut attempt = 0u32;
    let mut waited = Duration::ZERO;
    loop {
        let err = match op().await {
            Ok(value) => return Ok(value),
            Err(e) => e,
        };
        let Some(retryable) = err
            .chain()
            .find_map(|c| c.downcast_ref::<RetryableHttpError>())
            .cloned()
        else {
            return Err(err);
        };
        match backoff_delay(attempt, retryable.retry_after, waited, budget) {
            Some(delay) => {
                info!(
                    "transcription got HTTP {}; retrying in {:.1} s",
                    retryable.status,
                    delay.as_secs_f64()
                );
                tokio::time::sleep(delay).await;
                waited += delay;
                attempt += 1;
            }
            None => {
                return Err(RateLimitExhausted {
                    status: retryable.status,
                    message: give_up_message(&retryable, waited, budget),
                }
                .into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn limited(retry_after: Option<u64>) -> anyhow::Error {
        RetryableHttpError {
            status: 429,
            retry_after: retry_after.map(Duration::from_secs),
            message: "slow down".into(),
        }
        .into()
    }

    #[test]
    fn backoff_honors_retry_after_and_the_budget() {
        let budget = Duration::from_secs(300);
        // Exponential without a Retry-After, capped.
        assert_eq!(
            backoff_delay(0, None, Duration::ZERO, budget),
            Some(Duration::from_secs(2))
        );
        assert_eq!(
            backoff_delay(3, None, Duration::ZERO, budget),
            Some(Duration::from_secs(16))
        );
        assert_eq!(
            backoff_delay(30, None, Duration::ZERO, budget),
            Some(BACKOFF_MAX)
        );
        // The server's wait wins, even when longer than the backoff.
        assert_eq!(
            backoff_delay(0, Some(Duration::from_secs(90)), Duration::ZERO, budget),
            Some(Duration::from_secs(90))
        );
        // A zero Retry-After still pauses.
        assert_eq!(
            backoff_delay(0, Some(Duration::ZERO), Duration::ZERO, budget),
            Some(Duration::from_millis(500))
        );
        // Past the budget: give up.
        assert_eq!(
            backoff_delay(
                0,
                Some(Duration::from_secs(61)),
                Duration::from_secs(240),
                budget
            ),
            None
        );
        assert_eq!(
            backoff_delay(0, None, Duration::from_secs(299), budget),
            None
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_retry_after_beyond_the_budget_says_so() {
        let calls = AtomicUsize::new(0);
        let err = retry_rate_limited(RATE_LIMIT_BUDGET, || {
            calls.fetch_add(1, Ordering::Relaxed);
            async { Err::<(), _>(limited(Some(900))) }
        })
        .await
        .unwrap_err();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            err.to_string(),
            "rate limited (HTTP 429); the server asked to wait 900 s, more than the 300 s \
             budget: slow down"
        );
        // Exhausted is final: an outer layer must not retry it.
        assert!(err.downcast_ref::<RetryableHttpError>().is_none());
        assert!(err.downcast_ref::<RateLimitExhausted>().is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn endless_backoff_reports_the_time_waited() {
        let started = tokio::time::Instant::now();
        let err = retry_rate_limited(RATE_LIMIT_BUDGET, || async { Err::<(), _>(limited(None)) })
            .await
            .unwrap_err();
        assert!(started.elapsed() <= RATE_LIMIT_BUDGET);
        let text = err.to_string();
        assert!(
            text.starts_with("the backend still answered HTTP 429 after ")
                && text.ends_with(" s of retries: slow down"),
            "{text}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn other_errors_are_not_retried() {
        let calls = AtomicUsize::new(0);
        let err = retry_rate_limited(RATE_LIMIT_BUDGET, || {
            calls.fetch_add(1, Ordering::Relaxed);
            async { Err::<(), _>(anyhow::anyhow!("401 unauthorized")) }
        })
        .await
        .unwrap_err();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(err.to_string(), "401 unauthorized");
    }
}
