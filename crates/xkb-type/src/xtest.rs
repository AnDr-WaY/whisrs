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
use xkbcommon::xkb::keysyms::{KEY_Control_L, KEY_ISO_Level3_Shift, KEY_Shift_L};

use crate::keymap::Group1Keysyms;
use crate::{Key, KeyMapping, KeyTap, XkbKeymap};

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

/// The modifiers [`Keyboard`](crate::Keyboard) presses by keycode, with the
/// keysym each must give: Shift for shifted characters and the terminal
/// paste (Ctrl+Shift+V), Ctrl for the paste batch's Ctrl+V and
/// `send_combo` shortcuts. Each is checked on its own, so an option that
/// acts only on a combination (`grp:ctrl_shift_toggle` switching group on
/// Ctrl+Shift) is not caught; the physical keyboard behaves the same.
const MODIFIERS: [(Key, u32, &str); 2] = [
    (Key::KEY_LEFTSHIFT, KEY_Shift_L, "Shift_L"),
    (Key::KEY_LEFTCTRL, KEY_Control_L, "Control_L"),
];

/// What [`layout_mismatches`] found typing differently on the server.
#[derive(Debug, Default)]
struct Mismatches {
    /// Modifier keys the sink presses, by the keysym they should give.
    modifiers: Vec<&'static str>,
    /// Characters with a key that gives another keysym.
    chars: Vec<char>,
}

impl Mismatches {
    fn is_empty(&self) -> bool {
        self.modifiers.is_empty() && self.chars.is_empty()
    }

    /// Up to 12 of them, modifiers first: `[Control_L, 'y', 'z']`.
    fn sample(&self) -> String {
        let mut chars = self.chars.clone();
        chars.sort_unstable();
        let items: Vec<String> = self
            .modifiers
            .iter()
            .map(|name| name.to_string())
            .chain(chars.iter().map(|ch| format!("{ch:?}")))
            .take(12)
            .collect();
        format!("[{}]", items.join(", "))
    }
}

/// Every key `mapping` presses: the main one and the one after a dead key.
fn taps(mapping: &KeyMapping) -> impl Iterator<Item = &KeyTap> {
    std::iter::once(&mapping.main).chain(&mapping.follow)
}

/// What `keymap` would type differently on the `server` keyboard:
/// a modifier the sink presses is not that modifier there (`ctrl:swapcaps`
/// makes Ctrl+V a Caps Lock), or a key pressed for a character (a dead key
/// and the key after it included) gives a different keysym at that level
/// than in `keymap`'s group 1.
///
/// Control characters other than `\t`, `\n` and `\r` are not compared:
/// whisrs strips them from dictated text, and options like
/// `caps:swapescape` move Escape without changing what it types.
///
/// Group 1 against group 1 keeps a second layout group (`us,ru`) from
/// failing a `us` XTEST keyboard: its characters are pressed with group 1
/// keys, so both keyboards give the same group 1 keysym for them.
fn layout_mismatches(server: &Group1Keysyms, keymap: &XkbKeymap) -> Mismatches {
    let base = |keycode| KeyTap {
        keycode,
        shift: false,
        altgr: false,
    };
    let typed = || {
        keymap
            .iter()
            .filter(|(ch, _)| !ch.is_control() || matches!(ch, '\t' | '\n' | '\r'))
    };
    let mut modifiers: Vec<&'static str> = MODIFIERS
        .iter()
        .filter(|(key, keysym, _)| !same_symbol(server.keysym(&base(key.code())), *keysym))
        .map(|(_, _, name)| *name)
        .collect();
    if typed().any(|(_, m)| taps(m).any(|tap| tap.altgr))
        && !same_symbol(
            server.keysym(&base(keymap.level3_keycode())),
            KEY_ISO_Level3_Shift,
        )
    {
        modifiers.push("ISO_Level3_Shift");
    }
    let chars = typed()
        .filter(|(_, m)| {
            taps(m).any(|tap| !same_symbol(server.keysym(tap), keymap.group1().keysym(tap)))
        })
        .map(|(ch, _)| ch)
        .collect();
    Mismatches { modifiers, chars }
}

/// Drop from `keymap` every character with a key (the one after a dead key
/// included) outside the X keycodes `min..=max`, such as `€` on `KEY_EURO`
/// (X keycode 443): XTEST cannot press it, so `Keyboard::type_text` pastes
/// it like a character the layout lacks instead of failing mid-text.
fn drop_unsendable(keymap: &mut XkbKeymap, min: u8, max: u8) {
    let sendable = |tap: &KeyTap| {
        tap.keycode
            .checked_add(EVDEV_TO_X11_KEYCODE)
            .and_then(|x11| u8::try_from(x11).ok())
            .is_some_and(|x11| (min..=max).contains(&x11))
    };
    keymap.retain(|m| taps(m).all(sendable));
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
    /// release any key a previous connection left down. Characters on keys
    /// the server has no keycode for are dropped from `keymap`, so they are
    /// pasted.
    pub(crate) fn connect(keymap: &mut XkbKeymap) -> anyhow::Result<Self> {
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
        let setup = conn.setup();
        drop_unsendable(keymap, setup.min_keycode, setup.max_keycode);
        let root = setup.roots[screen].root;
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
        let mismatches = layout_mismatches(&server, keymap);
        if !mismatches.is_empty() {
            anyhow::bail!(
                "the X server's XTEST keyboard types differently than the detected layout \
                 (keys for {} differ); run setxkbmap with your layout, or, if an XKB option \
                 moves one of these keys (e.g. ctrl:swapcaps), use backend = \"auto\"",
                mismatches.sample()
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

    fn server(layout: &str, variant: &str) -> Group1Keysyms {
        server_with_options(layout, variant, "")
    }

    /// The group 1 keysyms of an XTEST keyboard with this layout and XKB
    /// options, compiled by xkbcommon so no X server is needed, without the
    /// keys past the X keycode range that a server cannot have.
    fn server_with_options(layout: &str, variant: &str, options: &str) -> Group1Keysyms {
        let context = xkbcommon::xkb::Context::new(xkbcommon::xkb::CONTEXT_NO_FLAGS);
        let keymap = xkbcommon::xkb::Keymap::new_from_names(
            &context,
            "",
            "",
            layout,
            variant,
            Some(options.to_string()),
            xkbcommon::xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .unwrap();
        let mut keysyms = Group1Keysyms::from_keymap(&keymap);
        keysyms
            .0
            .retain(|&keycode, _| keycode + EVDEV_TO_X11_KEYCODE <= u16::from(u8::MAX));
        keysyms
    }

    /// What a `detected` keymap, filtered as `connect` does, would type
    /// differently on `server`.
    fn mismatches_on(detected_layout: (&str, &str), server: &Group1Keysyms) -> Mismatches {
        let mut keymap = detected(detected_layout.0, detected_layout.1);
        drop_unsendable(&mut keymap, 8, 255);
        layout_mismatches(server, &keymap)
    }

    fn mismatches(detected_layout: (&str, &str), server_layout: (&str, &str)) -> Mismatches {
        mismatches_on(detected_layout, &server(server_layout.0, server_layout.1))
    }

    #[test]
    fn de_on_us_xtest_keyboard_fails() {
        let m = mismatches(("de", ""), ("us", ""));
        assert!(m.chars.contains(&'y') && m.chars.contains(&'z'), "{m:?}");
    }

    #[test]
    fn altgr_level_differences_fail() {
        // `é` is AltGr+e on us(altgr-intl); plain `us` has no third level.
        let m = mismatches(("us", "altgr-intl"), ("us", ""));
        assert!(m.chars.contains(&'é'), "{m:?}");
    }

    #[test]
    fn dead_key_layout_on_us_xtest_keyboard_fails() {
        // us(intl) turns `'` into dead_acute and moves `'` to AltGr.
        let m = mismatches(("us", "intl"), ("us", ""));
        assert!(m.chars.contains(&'\''), "{m:?}");
    }

    #[test]
    fn matching_layouts_pass() {
        for (layout, variant) in [("us", ""), ("us", "altgr-intl"), ("us", "intl"), ("de", "")] {
            let m = mismatches((layout, variant), (layout, variant));
            assert!(m.is_empty(), "{layout}({variant}): {m:?}");
        }
    }

    #[test]
    fn second_group_passes_on_its_first_group() {
        // `ф` comes from group 2 and is typed with group 1's `a` key, the
        // same on a `us` XTEST keyboard as on a `us,ru` one.
        assert!(detected("us,ru", "").lookup('ф').is_some());
        let m = mismatches(("us,ru", ""), ("us", ""));
        assert!(m.is_empty(), "{m:?}");
        let m = mismatches(("us,ru", ""), ("us,ru", ""));
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn altgr_characters_fail_without_level3_shift() {
        let keymap = detected("us", "altgr-intl");
        let mut server = server("us", "altgr-intl");
        server.0.get_mut(&keymap.level3_keycode()).unwrap()[0] = xkbcommon::xkb::keysyms::KEY_Alt_R;
        let m = mismatches_on(("us", "altgr-intl"), &server);
        assert_eq!(m.modifiers, ["ISO_Level3_Shift"], "{m:?}");
        assert_eq!(m.sample(), "[ISO_Level3_Shift]");
    }

    #[test]
    fn moved_escape_passes() {
        // caps:swapescape puts Caps_Lock on Escape's key; Escape is never
        // typed from dictated text.
        let server = server_with_options("us", "", "caps:swapescape");
        let esc = KeyTap {
            keycode: Key::KEY_ESC.code(),
            shift: false,
            altgr: false,
        };
        assert_eq!(server.keysym(&esc), xkbcommon::xkb::keysyms::KEY_Caps_Lock);
        let m = mismatches_on(("us", ""), &server);
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn moved_ctrl_fails() {
        // ctrl:swapcaps makes the paste's Ctrl+V a Caps Lock and a `V`.
        let m = mismatches_on(("us", ""), &server_with_options("us", "", "ctrl:swapcaps"));
        assert_eq!(m.modifiers, ["Control_L"], "{m:?}");
        assert_eq!(m.sample(), "[Control_L]");
    }

    #[test]
    fn keys_past_the_x_keycode_range_are_pasted() {
        let mut keymap = detected("us", "");
        assert!(keymap.lookup('€').is_some(), "`€` is on KEY_EURO");
        drop_unsendable(&mut keymap, 8, 255);
        assert!(keymap.lookup('€').is_none());
        assert!(keymap.lookup('a').is_some());
    }

    #[test]
    fn keys_outside_the_server_keycodes_are_pasted() {
        // A server whose keycodes stop at 25 has `q` (24) but not `u` (30),
        // so `ú` goes although its dead acute (21) fits.
        let mut keymap = detected("de", "");
        let acute_u = *keymap.lookup('ú').unwrap();
        assert_eq!(acute_u.main.keycode + EVDEV_TO_X11_KEYCODE, 21);
        assert_eq!(acute_u.follow.unwrap().keycode + EVDEV_TO_X11_KEYCODE, 30);
        drop_unsendable(&mut keymap, 8, 25);
        assert!(keymap.lookup('q').is_some());
        assert!(keymap.lookup('w').is_some(), "`w` (25) is the last keycode");
        assert!(keymap.lookup('ú').is_none());
        assert!(keymap.lookup('u').is_none());
        // And keycodes below the server's minimum.
        let mut keymap = detected("us", "");
        drop_unsendable(&mut keymap, 25, 255);
        assert!(keymap.lookup('q').is_none());
        assert!(
            keymap.lookup('w').is_some(),
            "`w` (25) is the first keycode"
        );
        assert!(keymap.lookup('a').is_some());
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
