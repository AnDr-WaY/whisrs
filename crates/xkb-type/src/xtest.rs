//! X11 key injection through the XTEST extension.
//!
//! XTEST requests travel over the client's X connection and the server
//! processes them in order. A uinput device instead goes through the
//! kernel's per-reader evdev queue (about 64 events): a burst written
//! faster than the X server reads it overflows that queue, the server's
//! libevdev discards whatever is still queued after `SYN_DROPPED`, and the
//! end of the text never arrives. XTEST has no such queue, so the per-key
//! delay is only pacing and can be zero.

use anyhow::Context;
use x11rb::connection::{Connection, RequestConnection};
use x11rb::protocol::xinput::{self, ConnectionExt as _};
use x11rb::protocol::xkb::{self, ConnectionExt as _};
use x11rb::protocol::xproto::{ConnectionExt as _, Window, KEY_PRESS_EVENT, KEY_RELEASE_EVENT};
use x11rb::protocol::xtest::{self, ConnectionExt as _};
use x11rb::rust_connection::RustConnection;
use xkbcommon::xkb::keysyms::KEY_ISO_Level3_Shift;

use crate::keymap::Group1Keysyms;
use crate::{KeyTap, XkbKeymap};

/// Offset from an evdev keycode to the X11 keycode of the server's evdev
/// keymap.
const EVDEV_TO_X11_KEYCODE: u16 = 8;

/// Name of the slave keyboard the X server attaches XTEST key events to.
const XTEST_KEYBOARD_NAME: &[u8] = b"Virtual core XTEST keyboard";

/// The group 1 keysyms of each key in `maps`, the XKB symbol maps of the
/// X keycodes from `first_keycode`.
///
/// The core-protocol mapping cannot stand in for this: it lists levels 3
/// and up of every group after the first two levels of both, so with a
/// two-level group 1 (`us,ru`) its fifth column is group 2 level 3.
fn group1_keysyms(first_keycode: u8, maps: &[xkb::KeySymMap]) -> Group1Keysyms {
    let levels = maps
        .iter()
        .zip(u16::from(first_keycode)..)
        .filter_map(|(map, x11_keycode)| {
            // `syms` holds `width` keysyms per group, group 1 first.
            let width = usize::from(map.width);
            let levels = [0, 1, 2, 3].map(|level| {
                if level < width {
                    map.syms.get(level).copied().unwrap_or(0)
                } else {
                    0
                }
            });
            Some((x11_keycode.checked_sub(EVDEV_TO_X11_KEYCODE)?, levels))
        })
        .collect();
    Group1Keysyms(levels)
}

/// Characters `keymap` would type differently on the `server` keyboard:
/// a key it presses for them (a dead key and the key after it included)
/// gives a different keysym at that level than in `keymap`'s group 1, or
/// the character needs AltGr and `keymap`'s level 3 key is not
/// `ISO_Level3_Shift` there.
///
/// Group 1 against group 1 keeps a second layout group (`us,ru`) from
/// failing a `us` XTEST keyboard: its characters are pressed with group 1
/// keys, so both keyboards give the same group 1 keysym for them.
/// Characters on a key `server` lacks are skipped: past the X keycode
/// range (`€` on `KEY_EURO`), no X keyboard has it, whatever its layout.
fn layout_mismatches(server: &Group1Keysyms, keymap: &XkbKeymap) -> Vec<char> {
    let level3 = KeyTap {
        keycode: keymap.level3_keycode(),
        shift: false,
        altgr: false,
    };
    let level3_ok = same_symbol(server.keysym(&level3), KEY_ISO_Level3_Shift);
    keymap
        .iter()
        .filter(|(_, m)| {
            let taps = || std::iter::once(&m.main).chain(&m.follow);
            taps().all(|tap| server.0.contains_key(&tap.keycode))
                && taps().any(|tap| {
                    (tap.altgr && !level3_ok)
                        || !same_symbol(server.keysym(tap), keymap.group1().keysym(tap))
                })
        })
        .map(|(ch, _)| ch)
        .collect()
}

/// Whether keysyms `a` and `b` type the same: equal, or the same character
/// (`Cyrillic_ef` and `U0444`).
fn same_symbol(a: u32, b: u32) -> bool {
    let ch = xkbcommon::xkb::keysym_to_utf32(a.into());
    a == b || (ch != 0 && ch == xkbcommon::xkb::keysym_to_utf32(b.into()))
}

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

    /// Fail unless every key `keymap` would press, at the level it would
    /// press it, gives the same keysym on the XTEST keyboard.
    ///
    /// The XTEST keyboard has its own keymap. Layouts set only through an
    /// xorg.conf `InputClass` (e.g. `localectl set-x11-keymap`) apply to
    /// hotplugged devices, not to it, so it can stay `us` while the
    /// detected layout is `de`: typing through it would swap `y` and `z`.
    fn check_keymap(&self, keymap: &XkbKeymap) -> anyhow::Result<()> {
        let device = self.xtest_keyboard_id()?;
        if !self
            .conn
            .xkb_use_extension(1, 0)
            .context("failed to query the XKB extension")?
            .reply()
            .context("failed to query the XKB extension")?
            .supported
        {
            anyhow::bail!("the X server does not support the XKB extension");
        }
        // The symbols of every key (`full`); the ranges below are only read
        // for `partial` parts, of which there are none.
        let reply = self
            .conn
            .xkb_get_map(
                device.into(),
                xkb::MapPart::KEY_SYMS,
                xkb::MapPart::default(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                xkb::VMod::default(),
                0,
                0,
                0,
                0,
                0,
                0,
            )
            .context("failed to request the XTEST keyboard mapping")?
            .reply()
            .context("failed to read the XTEST keyboard mapping")?;
        let server = group1_keysyms(
            reply.first_key_sym,
            reply.map.syms_rtrn.as_deref().unwrap_or_default(),
        );
        let mismatched = layout_mismatches(&server, keymap);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KeyboardLayout;

    fn detected(layout: &str, variant: &str) -> XkbKeymap {
        XkbKeymap::from_layout(&KeyboardLayout {
            layout: layout.to_string(),
            variant: variant.to_string(),
        })
        .unwrap()
    }

    /// The group 1 keysyms of an XTEST keyboard with this layout, compiled
    /// by xkbcommon so no X server is needed, without the keys past the X
    /// keycode range that a server cannot have.
    fn server(layout: &str, variant: &str) -> Group1Keysyms {
        let context = xkbcommon::xkb::Context::new(xkbcommon::xkb::CONTEXT_NO_FLAGS);
        let keymap = xkbcommon::xkb::Keymap::new_from_names(
            &context,
            "",
            "",
            layout,
            variant,
            None,
            xkbcommon::xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .unwrap();
        let mut keysyms = Group1Keysyms::from_keymap(&keymap);
        keysyms
            .0
            .retain(|&keycode, _| keycode + EVDEV_TO_X11_KEYCODE <= u16::from(u8::MAX));
        keysyms
    }

    fn mismatches(detected_layout: (&str, &str), server_layout: (&str, &str)) -> Vec<char> {
        let keymap = detected(detected_layout.0, detected_layout.1);
        let mut chars = layout_mismatches(&server(server_layout.0, server_layout.1), &keymap);
        chars.sort_unstable();
        chars
    }

    #[test]
    fn de_on_us_xtest_keyboard_fails() {
        let chars = mismatches(("de", ""), ("us", ""));
        assert!(chars.contains(&'y') && chars.contains(&'z'), "{chars:?}");
    }

    #[test]
    fn altgr_level_differences_fail() {
        // `é` is AltGr+e on us(altgr-intl); plain `us` has no third level.
        let chars = mismatches(("us", "altgr-intl"), ("us", ""));
        assert!(chars.contains(&'é'), "{chars:?}");
    }

    #[test]
    fn dead_key_layout_on_us_xtest_keyboard_fails() {
        // us(intl) turns `'` into dead_acute and moves `'` to AltGr.
        let chars = mismatches(("us", "intl"), ("us", ""));
        assert!(chars.contains(&'\''), "{chars:?}");
    }

    #[test]
    fn matching_layouts_pass() {
        for (layout, variant) in [("us", ""), ("us", "altgr-intl"), ("us", "intl"), ("de", "")] {
            let chars = mismatches((layout, variant), (layout, variant));
            assert!(chars.is_empty(), "{layout}({variant}): {chars:?}");
        }
    }

    #[test]
    fn second_group_passes_on_its_first_group() {
        // `ф` comes from group 2 and is typed with group 1's `a` key, the
        // same on a `us` XTEST keyboard as on a `us,ru` one.
        assert!(detected("us,ru", "").lookup('ф').is_some());
        assert_eq!(mismatches(("us,ru", ""), ("us", "")), []);
        assert_eq!(mismatches(("us,ru", ""), ("us,ru", "")), []);
    }

    #[test]
    fn altgr_characters_fail_without_level3_shift() {
        let keymap = detected("us", "altgr-intl");
        let mut server = server("us", "altgr-intl");
        server.0.get_mut(&keymap.level3_keycode()).unwrap()[0] = xkbcommon::xkb::keysyms::KEY_Alt_R;
        let chars = layout_mismatches(&server, &keymap);
        assert!(chars.contains(&'é'), "{chars:?}");
        assert!(!chars.contains(&'e'), "{chars:?}");
    }

    #[test]
    fn group1_keysyms_skips_later_groups() {
        // X keycode 17 (`8`) on `us,ru`: two groups of width 4, group 1
        // with only two levels. Group 2 level 3 is `₽`.
        let maps = [xkb::KeySymMap {
            kt_index: [0; 4],
            group_info: 2,
            width: 4,
            syms: vec![0x38, 0x2a, 0, 0, 0x38, 0x2a, 0x10020bd, 0],
        }];
        let server = group1_keysyms(17, &maps);
        assert_eq!(server.0.get(&9), Some(&[0x38, 0x2a, 0, 0]));
    }
}
