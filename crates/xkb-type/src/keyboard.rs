//! Virtual keyboard via evdev/uinput, or via XTEST on X11.
//!
//! Creates a virtual keyboard device via `/dev/uinput` (or, with the `xtest`
//! feature, an X server connection) and injects key events using the reverse
//! XKB keymap to produce the correct characters regardless of the user's
//! keyboard layout.

use std::thread;
use std::time::Duration;

use anyhow::Context;
use evdev::{uinput::VirtualDevice, AttributeSet, EventType, InputEvent, Key};
// Logging via the `log` crate (only active with `logging` feature).
#[cfg(feature = "logging")]
use log::{debug, warn};
#[cfg(not(feature = "logging"))]
macro_rules! debug {
    ($($arg:tt)*) => {};
}
#[cfg(not(feature = "logging"))]
macro_rules! warn {
    ($($arg:tt)*) => {};
}

use crate::keymap::{KeyboardLayout, XkbKeymap};
use crate::{default_clipboard, ClipboardBackend, KeyTap};

/// Delay after creating the virtual device on the runtime path (e.g. when
/// the daemon re-creates the keyboard after an error). Short, because this
/// runs on the user's typing path — every error-recovery would otherwise
/// stall typing.
const DEVICE_SETTLE_DELAY: Duration = Duration::from_millis(200);

/// Longer delay used at daemon startup (prewarm), so X11 has time to
/// process MappingNotify and attach the new device's keymap before the
/// first character is typed. Without this, the very first typed character
/// after boot can race the X11 keymap refresh and drop accented keys
/// (notably `<LVL3>` characters like `é` on `us:qwerty-fr`).
const INITIAL_WARMUP_DELAY: Duration = Duration::from_secs(1);

/// See [`Sink::min_delay`].
const UINPUT_MIN_DELAY: Duration = Duration::from_millis(1);

/// Where a [`Keyboard`]'s key events go.
enum Sink {
    /// A uinput device: each event is written to the kernel immediately.
    Uinput(VirtualDevice),
    /// The X server via XTEST: events are queued on the X connection.
    #[cfg(feature = "xtest")]
    Xtest(Box<crate::xtest::XtestSink>),
}

impl Sink {
    /// Press (`pressed`) or release the key with evdev code `code`.
    fn key(&mut self, code: u16, pressed: bool) -> anyhow::Result<()> {
        match self {
            Sink::Uinput(device) => {
                device.emit(&[InputEvent::new(EventType::KEY, code, i32::from(pressed))])?;
            }
            #[cfg(feature = "xtest")]
            Sink::Xtest(xtest) => xtest.key(code, pressed)?,
        }
        Ok(())
    }

    /// Hand queued events to the X server. uinput writes are immediate.
    fn flush(&mut self) -> anyhow::Result<()> {
        match self {
            Sink::Uinput(_) => Ok(()),
            #[cfg(feature = "xtest")]
            Sink::Xtest(xtest) => xtest.flush(),
        }
    }

    /// The shortest wait between events. uinput events pass through the
    /// kernel's evdev queue (about 64 events) to the X server or compositor;
    /// with no wait a burst overflows it and the reader drops the rest of the
    /// text, so a zero `key_delay` is raised to 1 ms there.
    fn min_delay(&self) -> Duration {
        match self {
            Sink::Uinput(_) => UINPUT_MIN_DELAY,
            #[cfg(feature = "xtest")]
            Sink::Xtest(_) => Duration::ZERO,
        }
    }

    /// Return once every event sent so far has been processed (XTEST only).
    fn sync(&mut self) -> anyhow::Result<()> {
        match self {
            Sink::Uinput(_) => Ok(()),
            #[cfg(feature = "xtest")]
            Sink::Xtest(xtest) => xtest.sync(),
        }
    }
}

/// Virtual keyboard that injects keystrokes via uinput, or via XTEST on X11
/// (see [`Keyboard::new_xtest`]).
pub struct Keyboard {
    sink: Sink,
    keymap: XkbKeymap,
    clipboard: Box<dyn ClipboardBackend>,
    key_delay: Duration,
}

impl Keyboard {
    /// Create a new virtual keyboard with auto-detected layout and clipboard.
    ///
    /// This is the most ergonomic constructor: it picks up the active
    /// XKB layout from the compositor (Hyprland / Sway / env vars), picks
    /// the right clipboard backend for the current display server, and
    /// builds the uinput device.
    ///
    /// `key_delay` is the inter-event delay. Raise it for TUIs that drop
    /// characters in raw mode (e.g. Node/Ink-based apps like Claude Code).
    ///
    /// Requires write access to `/dev/uinput`.
    ///
    /// For full control over keymap/clipboard, see
    /// [`Keyboard::with_components`]; to force a specific XKB layout, see
    /// [`Keyboard::with_layout`].
    pub fn new(key_delay: Duration) -> anyhow::Result<Self> {
        let layout = KeyboardLayout::detect();
        let keymap = XkbKeymap::from_layout(&layout)?;
        Self::with_components(keymap, default_clipboard(), key_delay)
    }

    /// Like [`Keyboard::new`], but uses a longer post-creation settle
    /// delay suited for the **first** virtual keyboard created in a
    /// process. Call this from a startup prewarm so X11 has time to
    /// process MappingNotify and attach the new device's keymap before
    /// any character is typed.
    ///
    /// Subsequent recreates (e.g. on error recovery) should keep using
    /// [`Keyboard::new`], which uses a shorter settle so the user's
    /// typing path is not stalled by 1s on every recovery.
    pub fn new_prewarm(key_delay: Duration) -> anyhow::Result<Self> {
        let layout = KeyboardLayout::detect();
        let keymap = XkbKeymap::from_layout(&layout)?;
        Self::with_components_settle(keymap, default_clipboard(), key_delay, INITIAL_WARMUP_DELAY)
    }

    /// Create a virtual keyboard for a specific XKB layout (and optional
    /// variant), with the auto-detected clipboard backend.
    ///
    /// Useful when the application already knows the user's layout and
    /// wants to skip the compositor probe (e.g. it's stored in a config
    /// file).
    pub fn with_layout(
        layout: &str,
        variant: Option<&str>,
        key_delay: Duration,
    ) -> anyhow::Result<Self> {
        let detected = KeyboardLayout {
            layout: layout.to_string(),
            variant: variant.unwrap_or("").to_string(),
        };
        let keymap = XkbKeymap::from_layout(&detected)?;
        Self::with_components(keymap, default_clipboard(), key_delay)
    }

    /// Create a virtual keyboard from explicit components (keymap +
    /// clipboard backend).
    ///
    /// Use this when you need to inject a custom [`XkbKeymap`] (e.g. for
    /// tests) or a non-default clipboard backend
    /// ([`crate::NoopClipboard`] / [`crate::WaylandClipboard`] /
    /// [`crate::X11Clipboard`]).
    ///
    /// Requires write access to `/dev/uinput` (user must be in the `input`
    /// group or have the appropriate udev rule installed).
    ///
    /// `key_delay` is the inter-event delay. Raise it for TUIs that drop
    /// characters in raw mode (e.g. Node/Ink-based apps like Claude Code).
    pub fn with_components(
        keymap: XkbKeymap,
        clipboard: Box<dyn ClipboardBackend>,
        key_delay: Duration,
    ) -> anyhow::Result<Self> {
        Self::with_components_settle(keymap, clipboard, key_delay, DEVICE_SETTLE_DELAY)
    }

    /// Same as [`Keyboard::with_components`], but with an explicit
    /// post-creation settle delay. Kept private — callers should pick
    /// between [`Keyboard::new`] (short settle, runtime path) and
    /// [`Keyboard::new_prewarm`] (long settle, startup path).
    fn with_components_settle(
        keymap: XkbKeymap,
        clipboard: Box<dyn ClipboardBackend>,
        key_delay: Duration,
        settle: Duration,
    ) -> anyhow::Result<Self> {
        // Register all key codes we might need (1..=247 covers standard
        // keys; KEY_MAX on Linux is ~767 but codes beyond 247 are rare).
        let mut keys = AttributeSet::<Key>::new();
        for code in 1..=247 {
            keys.insert(Key::new(code));
        }

        let device = evdev::uinput::VirtualDeviceBuilder::new()
            .context("failed to create VirtualDeviceBuilder")?
            .name("whisrs virtual keyboard")
            .with_keys(&keys)
            .context("failed to register key events")?
            .build()
            .context("failed to build uinput virtual device — check /dev/uinput permissions")?;

        // Give the kernel (and on X11, the server) time to register the
        // new device and refresh its keymap before any keys are sent.
        thread::sleep(settle);

        debug!("uinput virtual keyboard created");

        Ok(Self {
            sink: Sink::Uinput(device),
            keymap,
            clipboard,
            key_delay,
        })
    }

    /// Create a keyboard that injects through the X server's XTEST extension
    /// instead of uinput, with auto-detected layout and clipboard.
    ///
    /// XTEST events go straight to the X server, which processes them in
    /// order, so nothing is dropped however fast they are sent: `key_delay`
    /// can be zero. Needs no `/dev/uinput` access and no settle delay.
    ///
    /// Fails on Xwayland (keys would only reach X11 clients) and when the
    /// server's XTEST keyboard has a different layout than the detected one.
    #[cfg(feature = "xtest")]
    pub fn new_xtest(key_delay: Duration) -> anyhow::Result<Self> {
        let layout = KeyboardLayout::detect();
        let keymap = XkbKeymap::from_layout(&layout)?;
        let sink = Sink::Xtest(Box::new(crate::xtest::XtestSink::connect(&keymap)?));
        debug!("XTEST keyboard connected");
        Ok(Self {
            sink,
            keymap,
            clipboard: default_clipboard(),
            key_delay,
        })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    pub fn set_key_delay(&mut self, key_delay: Duration) {
        self.key_delay = key_delay;
    }

    /// Type `text` by injecting keystrokes through the virtual device.
    ///
    /// Characters found in the keymap are typed via direct key events
    /// (with Shift/AltGr modifiers as needed, including dead-key sequences).
    /// Characters *not* found in the keymap are accumulated and pasted as
    /// a batch via the clipboard (Ctrl+V), with clipboard save/restore
    /// so the user's clipboard is not permanently overwritten.
    pub fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
        self.release_all_modifiers()?;

        let mut paste_buf = String::new();

        for ch in text.chars() {
            // Copy the lookup result to release the immutable borrow on
            // self.keymap before calling mutable methods below.
            let mapping = self.keymap.lookup(ch).copied();
            if let Some(mapping) = mapping {
                // Flush any pending paste buffer first.
                if !paste_buf.is_empty() {
                    self.paste_text(&paste_buf)?;
                    paste_buf.clear();
                }
                self.tap_key(&mapping.main)?;
                if let Some(follow) = mapping.follow.as_ref() {
                    self.tap_key(follow)?;
                }
            } else {
                // Character not in keymap — accumulate for clipboard paste.
                paste_buf.push(ch);
            }
        }

        // Flush remaining paste buffer.
        if !paste_buf.is_empty() {
            self.paste_text(&paste_buf)?;
        }

        self.sink.sync()
    }

    /// Emit Backspace `count` times.
    pub fn backspace(&mut self, count: usize) -> anyhow::Result<()> {
        self.release_all_modifiers()?;

        for _ in 0..count {
            self.tap_key(&KeyTap {
                keycode: Key::KEY_BACKSPACE.code(),
                shift: false,
                altgr: false,
            })?;
        }

        self.sink.sync()
    }

    /// Press all keys in `keys`, then release them in reverse order.
    ///
    /// Each press and release is separated by `self.key_delay`.
    pub fn send_combo(&mut self, keys: &[Key]) -> anyhow::Result<()> {
        self.release_all_modifiers()?;

        // Press all keys in order.
        for key in keys {
            self.sink.key(key.code(), true)?;
            self.pause()?;
        }

        // Release in reverse order.
        for key in keys.iter().rev() {
            self.sink.key(key.code(), false)?;
            self.pause()?;
        }

        self.sink.sync()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Send queued events, then wait `key_delay` (at least
    /// [`Sink::min_delay`]).
    fn pause(&mut self) -> anyhow::Result<()> {
        self.sink.flush()?;
        thread::sleep(self.key_delay.max(self.sink.min_delay()));
        Ok(())
    }

    /// Press or release a modifier key.
    fn set_modifier(&mut self, modifier: Key, pressed: bool) -> anyhow::Result<()> {
        self.sink.key(modifier.code(), pressed)?;
        self.pause()
    }

    /// Press and release a single key, with Shift and/or AltGr held as needed.
    ///
    /// AltGr is the keycode XKB exposes for `ISO_Level3_Shift`. On some
    /// X11 layouts that is a dedicated `<LVL3>` keycode rather than
    /// `KEY_RIGHTALT`; using the XKB keycode keeps third-level characters
    /// like `é` working with uinput.
    fn tap_key(&mut self, tap: &KeyTap) -> anyhow::Result<()> {
        let level3_key = Key::new(self.keymap.level3_keycode());

        if tap.shift {
            self.set_modifier(Key::KEY_LEFTSHIFT, true)?;
        }
        if tap.altgr {
            self.set_modifier(level3_key, true)?;
        }

        self.sink.key(tap.keycode, true)?;
        self.pause()?;
        self.sink.key(tap.keycode, false)?;
        self.pause()?;

        if tap.altgr {
            self.set_modifier(level3_key, false)?;
        }
        if tap.shift {
            self.set_modifier(Key::KEY_LEFTSHIFT, false)?;
        }

        Ok(())
    }

    /// Release all modifier keys to prevent interference with injected text.
    fn release_all_modifiers(&mut self) -> anyhow::Result<()> {
        let level3_key = Key::new(self.keymap.level3_keycode());
        let modifiers = [
            Key::KEY_LEFTSHIFT,
            Key::KEY_RIGHTSHIFT,
            Key::KEY_LEFTCTRL,
            Key::KEY_RIGHTCTRL,
            Key::KEY_LEFTALT,
            Key::KEY_RIGHTALT,
            level3_key,
            Key::KEY_LEFTMETA,
            Key::KEY_RIGHTMETA,
        ];

        for modifier in &modifiers {
            self.sink.key(modifier.code(), false)?;
        }
        self.pause()
    }

    /// Inject Ctrl+V to paste from clipboard.
    fn inject_ctrl_v(&mut self) -> anyhow::Result<()> {
        self.set_modifier(Key::KEY_LEFTCTRL, true)?;
        self.sink.key(Key::KEY_V.code(), true)?;
        self.pause()?;
        self.sink.key(Key::KEY_V.code(), false)?;
        self.pause()?;
        self.set_modifier(Key::KEY_LEFTCTRL, false)?;
        // The paste must reach the server before the clipboard is restored.
        self.sink.sync()
    }

    /// Set clipboard to `text`, inject Ctrl+V, then restore previous content.
    fn paste_text(&mut self, text: &str) -> anyhow::Result<()> {
        // Save current clipboard.
        let saved = self.clipboard.get_text().ok();

        // Set new clipboard content.
        self.clipboard
            .set_text(text)
            .context("failed to set clipboard for paste")?;

        // Small delay to ensure clipboard is ready.
        thread::sleep(Duration::from_millis(10));

        // Inject Ctrl+V.
        self.release_all_modifiers()?;
        self.inject_ctrl_v()?;

        // Small delay before restoring clipboard.
        thread::sleep(Duration::from_millis(50));

        // Restore previous clipboard content.
        if let Some(previous) = saved {
            if let Err(_e) = self.clipboard.set_text(&previous) {
                warn!("failed to restore clipboard: {_e}");
            }
        }

        Ok(())
    }
}
