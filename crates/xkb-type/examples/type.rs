//! Manual test harness for the keyboard-injection backends.
//!
//! Builds a single backend in isolation — no daemon, audio, or transcription
//! needed — and types the supplied text at the focused window's cursor. Use
//! it to eyeball that the Wayland virtual-keyboard backend types
//! layout-independently (e.g. mixed Latin/Arabic), and to compare against the
//! layout-dependent uinput backend.
//!
//! Build (the `wayland-vk` backend is feature-gated):
//!
//! ```sh
//! cargo run -p xkb-type --features wayland-vk --example type -- \
//!     --backend wayland-vk --delay-ms 4 "ammeter اميتر ω д 1+2=3"
//! ```
//!
//! Usage:
//!
//! ```text
//! type [--backend auto|uinput|wayland-vk|xtest] [--delay-ms N] <text...>
//! ```
//!
//! `xtest` (X11 only) needs `--features xtest`. It refuses to start when the
//! X server's XTEST keyboard has a different layout than the detected one.
//!
//! `auto` mirrors the daemon: it uses the Wayland virtual keyboard when
//! `WAYLAND_DISPLAY` is set and the compositor supports
//! `zwp_virtual_keyboard_v1`, otherwise it falls back to uinput. It never
//! picks `xtest`, which is opt-in.

use std::ops::ControlFlow;
use std::time::Duration;

use xkb_type::KeyInjector;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Auto,
    Uinput,
    #[cfg(feature = "wayland-vk")]
    WaylandVk,
    #[cfg(feature = "xtest")]
    Xtest,
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "usage: type [--backend auto|uinput|wayland-vk|xtest] [--delay-ms N] <text...>\n\
         \n\
           --backend   which injection backend to use (default: auto)\n\
           --delay-ms  inter-key-event delay in milliseconds (default: 4)\n\
         \n\
         The 'wayland-vk' backend requires building with --features wayland-vk,\n\
         and 'xtest' with --features xtest."
    );
    std::process::exit(code);
}

fn parse_backend(s: &str) -> Backend {
    match s {
        "auto" => Backend::Auto,
        "uinput" => Backend::Uinput,
        "wayland-vk" => {
            #[cfg(feature = "wayland-vk")]
            {
                Backend::WaylandVk
            }
            #[cfg(not(feature = "wayland-vk"))]
            {
                eprintln!(
                    "error: the 'wayland-vk' backend requires building with --features wayland-vk"
                );
                std::process::exit(2);
            }
        }
        "xtest" => {
            #[cfg(feature = "xtest")]
            {
                Backend::Xtest
            }
            #[cfg(not(feature = "xtest"))]
            {
                eprintln!("error: the 'xtest' backend requires building with --features xtest");
                std::process::exit(2);
            }
        }
        other => {
            eprintln!("error: unknown backend '{other}'");
            print_usage_and_exit(2);
        }
    }
}

fn build_backend(
    backend: Backend,
    key_delay: Duration,
) -> anyhow::Result<(Box<dyn KeyInjector>, &'static str)> {
    match backend {
        Backend::Uinput => Ok((Box::new(xkb_type::Keyboard::new(key_delay)?), "uinput")),
        #[cfg(feature = "wayland-vk")]
        Backend::WaylandVk => Ok((
            Box::new(xkb_type::wayland_vk::WaylandVkKeyboard::new(key_delay)?),
            "wayland-vk",
        )),
        #[cfg(feature = "xtest")]
        Backend::Xtest => Ok((Box::new(xkb_type::Keyboard::new_xtest(key_delay)?), "xtest")),
        Backend::Auto => {
            // Mirror the daemon's Auto logic: prefer the Wayland virtual
            // keyboard when a Wayland session is detected, else uinput.
            #[cfg(feature = "wayland-vk")]
            if std::env::var_os("WAYLAND_DISPLAY").is_some() {
                match xkb_type::wayland_vk::WaylandVkKeyboard::new(key_delay) {
                    Ok(kb) => return Ok((Box::new(kb), "wayland-vk (auto)")),
                    Err(e) => {
                        eprintln!("wayland-vk unavailable, falling back to uinput: {e:#}");
                    }
                }
            }
            Ok((
                Box::new(xkb_type::Keyboard::new(key_delay)?),
                "uinput (auto)",
            ))
        }
    }
}

fn main() -> anyhow::Result<()> {
    let mut backend = Backend::Auto;
    let mut delay_ms: u64 = 4;
    let mut text_parts: Vec<String> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                let val = args.next().unwrap_or_else(|| {
                    eprintln!("error: --backend requires a value");
                    print_usage_and_exit(2);
                });
                backend = parse_backend(&val);
            }
            "--delay-ms" => {
                let val = args.next().unwrap_or_else(|| {
                    eprintln!("error: --delay-ms requires a value");
                    print_usage_and_exit(2);
                });
                delay_ms = val.parse().unwrap_or_else(|_| {
                    eprintln!("error: --delay-ms must be an integer");
                    print_usage_and_exit(2);
                });
            }
            "-h" | "--help" => print_usage_and_exit(0),
            _ => text_parts.push(arg),
        }
    }

    if text_parts.is_empty() {
        eprintln!("error: no text to type");
        print_usage_and_exit(2);
    }
    let text = text_parts.join(" ");

    let key_delay = Duration::from_millis(delay_ms);
    let (mut injector, chosen) = build_backend(backend, key_delay)?;
    eprintln!("backend: {chosen}");
    eprintln!("text:    {text:?}");

    // Give the operator time to focus the target window.
    for n in (1..=2).rev() {
        eprintln!("typing in {n}...");
        std::thread::sleep(Duration::from_secs(1));
    }
    eprintln!("typing now");

    // Type everything: the gate never stops it.
    let _ = injector.type_text(&text, &mut |_| ControlFlow::Continue(()))?;
    Ok(())
}
