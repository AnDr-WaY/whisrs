//! One-shot HTTP server for backend tests: what a request put on the wire.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A request as the server read it.
pub(crate) struct Request {
    /// Request line and headers, through the blank line.
    pub head: String,
    pub body: Vec<u8>,
}

impl Request {
    /// The value of header `name`, matched case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    /// The headers of the multipart part named `file` and the bytes from
    /// the start of its data to the end of the body.
    pub fn file_part(&self) -> (String, &[u8]) {
        let marker = b"name=\"file\"";
        let start = self
            .body
            .windows(marker.len())
            .position(|w| w == marker)
            .expect("multipart body has no `file` part");
        let data = start
            + self.body[start..]
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .expect("file part has no header end")
            + 4;
        let headers = String::from_utf8_lossy(&self.body[start..data]).into_owned();
        (headers, &self.body[data..])
    }
}

/// Accept a single connection, read the whole request, answer `200 OK` with
/// `body`, and hand back the request so a test can assert on what was
/// actually put on the wire.
pub(crate) async fn serve_once(listener: TcpListener, body: &'static str) -> Request {
    serve_one(&listener, 200, body).await
}

/// Answer one connection per `(status, body)`, in order, and hand back the
/// requests: a server that fails before it succeeds, or keeps failing.
pub(crate) async fn serve_responses(
    listener: TcpListener,
    responses: Vec<(u16, &'static str)>,
) -> Vec<Request> {
    let mut requests = Vec::new();
    for (status, body) in responses {
        requests.push(serve_one(&listener, status, body).await);
    }
    requests
}

/// Read one request and answer it with `status` and `body`, closing the
/// connection so the client opens a new one for its next request.
async fn serve_one(listener: &TcpListener, status: u16, body: &str) -> Request {
    let (mut stream, _) = listener.accept().await.unwrap();

    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "client closed before sending a request head");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();

    // Drain the body first: replying while the client is still writing
    // surfaces as a connection reset instead of a parsed response.
    let content_length: usize = head
        .to_ascii_lowercase()
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse().ok())
        .expect("reqwest must send content-length for a body of known length");
    while buf.len() < head_end + content_length {
        let n = stream.read(&mut chunk).await.unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }

    let reason = if status == 200 { "OK" } else { "Error" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();
    Request {
        head,
        body: buf.split_off(head_end),
    }
}
