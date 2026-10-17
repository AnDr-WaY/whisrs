//! X11 key injection through the XTEST extension.
//!
//! XTEST requests travel over the client's X connection and the server
//! processes them in order. A uinput device instead goes through the
//! kernel's per-reader evdev queue (about 64 events): a burst written
//! faster than the X server reads it overflows that queue, the server's
//! libevdev discards whatever is still queued after `SYN_DROPPED`, and the
//! end of the text never arrives. XTEST has no such queue, so the per-key
//! delay is only pacing and can be zero.

use std::collections::{HashMap, HashSet};

use anyhow::Context;
use x11rb::connection::{Connection, RequestConnection};
use x11rb::protocol::xinput::{self, ConnectionExt as _};
use x11rb::protocol::xproto::{ConnectionExt as _, Window, KEY_PRESS_EVENT, KEY_RELEASE_EVENT};
use x11rb::protocol::xtest::{self, ConnectionExt as _};
use x11rb::rust_connection::RustConnection;

use crate::XkbKeymap;

/// Offset from an evdev keycode to the X11 keycode of the server's evdev
/// keymap.
const EVDEV_TO_X11_KEYCODE: u16 = 8;

/// Name of the slave keyboard the X server attaches XTEST key events to.
const XTEST_KEYBOARD_NAME: &[u8] = b"Virtual core XTEST keyboard";

/// An X server connection that sends fake key events via XTEST.
pub(crate) struct XtestSink {
    conn: RustConnection,
    root: Window,
    /// X keycodes this sink has pressed and not yet released.
    pressed: [bool; 256],
}

impl XtestSink {
    /// Connect to `$DISPLAY`, check that it is a real X server with XTEST
    /// whose XTEST keyboard reads keycodes the way `keymap` expects, and
    /// release any key a previous connection left down.
    pub(crate) fn connect(keymap: &XkbKeymap) -> anyhow::Result<Self> {
        let (conn, screen) = x11rb::connect(None).context("failed to connect to the X server")?;
        if conn
            .extension_information(xtest::X11_EXTENSION_NAME)
            .context("failed to query the XTEST extension")?
            .is_none()
        {
            anyhow::bail!("the X server does not support the XTEST extension");
        }
        // Under Xwayland, XTEST keys reach only X11 clients, not the
        // Wayland session.
        if conn
            .extension_information("XWAYLAND")
            .context("failed to query the XWAYLAND extension")?
            .is_some()
        {
            anyhow::bail!("the X server is Xwayland; XTEST keys would only reach X11 clients");
        }
        let root = conn.setup().roots[screen].root;
        let mut sink = Self {
            conn,
            root,
            pressed: [false; 256],
        };
        sink.check_keymap(keymap)?;
        sink.release_all_keys()?;
        Ok(sink)
    }

    /// Fail unless every unmodified key `keymap` would press for a character
    /// the XTEST keyboard has as a base character gives that character.
    ///
    /// The XTEST keyboard has its own keymap. Layouts set only through an
    /// xorg.conf `InputClass` (e.g. `localectl set-x11-keymap`) apply to
    /// hotplugged devices, not to it, so it can stay `us` while the
    /// detected layout is `de`: typing through it would swap `y` and `z`.
    fn check_keymap(&self, keymap: &XkbKeymap) -> anyhow::Result<()> {
        let device = self.xtest_keyboard_id()?;
        let setup = self.conn.setup();
        let (min, max) = (setup.min_keycode, setup.max_keycode);
        let reply = self
            .conn
            .xinput_get_device_key_mapping(device, min, max - min + 1)
            .context("failed to request the XTEST keyboard mapping")?
            .reply()
            .context("failed to read the XTEST keyboard mapping")?;
        // Base character (group 1, level 1) of each key, by evdev keycode.
        let per_key = usize::from(reply.keysyms_per_keycode);
        let server: HashMap<u16, char> = (min..=max)
            .filter_map(|x11| {
                let index = usize::from(x11 - min) * per_key;
                let keysym = *reply.keysyms.get(index)?;
                let ch = char::from_u32(xkbcommon::xkb::keysym_to_utf32(keysym.into()))?;
                Some((u16::from(x11).checked_sub(EVDEV_TO_X11_KEYCODE)?, ch))
            })
            .collect();
        let server_chars: HashSet<char> = server.values().copied().collect();

        // Only characters the XTEST keyboard has as a base character are
        // compared: a second layout group in the detected layout (`us,ru`)
        // adds characters it never types without a group switch anyway.
        let mismatched: Vec<char> = keymap
            .iter()
            .filter(|(ch, m)| {
                !m.main.shift
                    && !m.main.altgr
                    && m.follow.is_none()
                    && server_chars.contains(ch)
                    && server.get(&m.main.keycode) != Some(ch)
            })
            .map(|(ch, _)| ch)
            .collect();
        if !mismatched.is_empty() {
            let mut sample: Vec<char> =
                mismatched.into_iter().filter(|c| !c.is_control()).collect();
            sample.sort_unstable();
            sample.truncate(12);
            anyhow::bail!(
                "the X server's XTEST keyboard uses a different layout than the detected one \
                 (keys for {sample:?} differ); run setxkbmap with your layout to fix it"
            );
        }
        Ok(())
    }

    /// The XInput device id of the XTEST slave keyboard.
    fn xtest_keyboard_id(&self) -> anyhow::Result<u8> {
        self.conn
            .xinput_xi_query_version(2, 0)
            .context("failed to query XInput")?
            .reply()
            .context("the X server does not support XInput 2")?;
        let devices = self
            .conn
            .xinput_xi_query_device(xinput::Device::ALL)
            .context("failed to list input devices")?
            .reply()
            .context("failed to list input devices")?;
        let device = devices
            .infos
            .iter()
            .find(|d| {
                d.type_ == xinput::DeviceType::SLAVE_KEYBOARD && d.name == XTEST_KEYBOARD_NAME
            })
            .context("the X server has no XTEST keyboard")?;
        u8::try_from(device.deviceid).context("XTEST keyboard device id out of range")
    }

    /// Release every keycode, so nothing a previous connection left down
    /// (a crash mid-keystroke) stays held. Releasing a key that is not down
    /// is a no-op.
    fn release_all_keys(&mut self) -> anyhow::Result<()> {
        let setup = self.conn.setup();
        for detail in setup.min_keycode..=setup.max_keycode {
            self.fake_key(detail, false)?;
        }
        self.sync()
    }

    fn fake_key(&mut self, detail: u8, pressed: bool) -> anyhow::Result<()> {
        let kind = if pressed {
            KEY_PRESS_EVENT
        } else {
            KEY_RELEASE_EVENT
        };
        // time 0 = now; deviceid is ignored for core key events.
        self.conn
            .xtest_fake_input(kind, detail, 0, self.root, 0, 0, 0)
            .context("failed to send XTEST key event")?;
        self.pressed[usize::from(detail)] = pressed;
        Ok(())
    }

    /// Queue a press or release of the key with evdev code `code`.
    pub(crate) fn key(&mut self, code: u16, pressed: bool) -> anyhow::Result<()> {
        let detail = u8::try_from(code + EVDEV_TO_X11_KEYCODE)
            .with_context(|| format!("evdev keycode {code} has no X11 keycode"))?;
        self.fake_key(detail, pressed)
    }

    /// Write queued requests to the server.
    pub(crate) fn flush(&mut self) -> anyhow::Result<()> {
        self.conn.flush().context("failed to flush X connection")?;
        Ok(())
    }

    /// Return once the server has processed every event sent so far.
    pub(crate) fn sync(&mut self) -> anyhow::Result<()> {
        self.conn
            .get_input_focus()
            .context("failed to send X round-trip")?
            .reply()
            .context("X server round-trip failed")?;
        // Nothing reads events (MappingNotify and the like); drop them so
        // they do not pile up for the life of the connection.
        while self
            .conn
            .poll_for_event()
            .context("failed to read X events")?
            .is_some()
        {}
        Ok(())
    }
}

impl Drop for XtestSink {
    /// XTEST keys stay down when the client disconnects, unlike a uinput
    /// device whose keys go up when it is destroyed.
    fn drop(&mut self) {
        for detail in 0..=u8::MAX {
            if self.pressed[usize::from(detail)] {
                let _ = self.fake_key(detail, false);
            }
        }
        let _ = self.conn.flush();
    }
}
