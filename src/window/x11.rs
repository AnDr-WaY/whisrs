//! X11 window tracking via the `x11rb` crate.

use anyhow::Context;
use tracing::debug;
use x11rb::connection::Connection;
use x11rb::errors::ReplyError;
use x11rb::properties::WmClass;
use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _, InputFocus, Window};
use x11rb::rust_connection::RustConnection;

use super::WindowTracker;

/// Window tracker for X11 sessions.
///
/// Queries `_NET_ACTIVE_WINDOW` to get the focused window and uses
/// `set_input_focus` to restore it.
pub struct X11Tracker {
    conn: RustConnection,
    root: Window,
    net_active_window_atom: u32,
}

impl X11Tracker {
    /// Connect to the X11 display and look up needed atoms.
    pub fn new() -> anyhow::Result<Self> {
        let (conn, screen_num) =
            RustConnection::connect(None).context("failed to connect to X11 display")?;

        let screen = &conn.setup().roots[screen_num];
        let root = screen.root;

        let atom_cookie = conn.intern_atom(false, b"_NET_ACTIVE_WINDOW")?;
        let atom_reply = atom_cookie
            .reply()
            .context("failed to intern _NET_ACTIVE_WINDOW atom")?;

        Ok(Self {
            conn,
            root,
            net_active_window_atom: atom_reply.atom,
        })
    }

    /// Resolve the currently active window from `_NET_ACTIVE_WINDOW`.
    ///
    /// Shared by `get_focused_window` and `get_focused_window_class` so there
    /// is a single copy of this logic. Deliberately not `get_input_focus`:
    /// that returns the focus-proxy child window used by GTK and Swing, which
    /// carries no `WM_CLASS` of its own.
    fn active_window(&self) -> anyhow::Result<Window> {
        let cookie = self.conn.get_property(
            false,
            self.root,
            self.net_active_window_atom,
            AtomEnum::WINDOW,
            0,
            1,
        )?;

        let reply = cookie
            .reply()
            .context("failed to get _NET_ACTIVE_WINDOW property")?;

        if reply.value_len == 0 {
            anyhow::bail!("_NET_ACTIVE_WINDOW returned empty value");
        }

        // The property value is a 32-bit window ID.
        let window_id = u32::from_ne_bytes(
            reply.value[..4]
                .try_into()
                .context("unexpected _NET_ACTIVE_WINDOW value length")?,
        );

        if window_id == 0 {
            anyhow::bail!("no active window (root focused)");
        }

        Ok(window_id)
    }
}

/// Resolve one half of a `WM_CLASS` property into a usable class string.
///
/// `WM_CLASS` holds two NUL-separated strings, instance first and class
/// second. x11rb hands back everything after the first NUL as the class and
/// strips only a single trailing NUL, so a malformed property with more than
/// two fields (`b"World\0Good\0Day"`) still arrives here with interior NULs.
/// Cut at the first one: a string containing a NUL can never match a terminal
/// list entry, so passing it through would silently disable terminal
/// detection instead of degrading to the first field.
///
/// The property is not guaranteed to be UTF-8 (ASCII is the norm), so decode
/// lossily rather than discard a window whose class holds one odd byte.
fn parse_wm_class(class_bytes: &[u8]) -> Option<String> {
    let end = class_bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(class_bytes.len());

    let class = String::from_utf8_lossy(&class_bytes[..end]);
    let class = class.trim();

    if class.is_empty() {
        None
    } else {
        Some(class.to_string())
    }
}

/// Which half of a `WM_CLASS` property a resolved class came from.
#[derive(Debug, PartialEq, Eq)]
enum ClassField {
    /// The class (second) half — the normal case.
    Class,
    /// The instance (first) half, used only when the class half is unusable.
    Instance,
}

/// Resolve a whole `WM_CLASS` property into a usable class string.
///
/// Contract: the **class** (second) field wins, falling back to the
/// **instance** (first) field only when the class is empty or absent, and
/// `None` when neither yields anything. This matches the Sway backend, whose
/// XWayland fallback reads `window_properties.class` first; the two have to
/// agree, or the same terminal gets classified differently depending on which
/// compositor the user happens to be running.
///
/// `is_terminal_class` (`src/daemon/injection.rs`) lowercases before matching,
/// so case alone is harmless — but the two fields are not case variants of one
/// another in general, so which one gets reported can decide the verdict.
///
/// Returns the field it used alongside the class, because the caller logs the
/// instance fallback differently from a plain class hit.
///
/// Pure: takes an already-fetched property, so the precedence is unit-testable
/// without a display.
fn class_from_wm_class(wm_class: &WmClass) -> Option<(String, ClassField)> {
    if let Some(class) = parse_wm_class(wm_class.class()) {
        return Some((class, ClassField::Class));
    }

    parse_wm_class(wm_class.instance()).map(|instance| (instance, ClassField::Instance))
}

impl WindowTracker for X11Tracker {
    fn get_focused_window(&self) -> anyhow::Result<String> {
        let window_id = self.active_window()?;

        debug!("X11 focused window: 0x{window_id:x}");
        Ok(window_id.to_string())
    }

    /// Report the focused window's `WM_CLASS`.
    ///
    /// The class-before-instance precedence lives in `class_from_wm_class`;
    /// this fetches the property and turns the outcome into the right
    /// `debug!` line.
    fn get_focused_window_class(&self) -> Option<String> {
        let window_id = match self.active_window() {
            Ok(window_id) => window_id,
            Err(err) => {
                debug!("X11 focused window class: active window lookup failed: {err}");
                return None;
            }
        };

        // WM_CLASS is a predefined atom, so this needs no intern_atom round
        // trip. x11rb owns the property fetch and the NUL split.
        let wm_class = WmClass::get(&self.conn, window_id)
            .map_err(ReplyError::from)
            .and_then(|cookie| cookie.reply());

        let wm_class = match wm_class {
            Ok(Some(wm_class)) => wm_class,
            Ok(None) => {
                debug!("X11 focused window class: no WM_CLASS property on 0x{window_id:x}");
                return None;
            }
            Err(err) => {
                debug!("X11 focused window class: WM_CLASS request failed: {err}");
                return None;
            }
        };

        match class_from_wm_class(&wm_class) {
            Some((class, ClassField::Class)) => {
                debug!("X11 focused window class: {class}");
                Some(class)
            }
            Some((instance, ClassField::Instance)) => {
                debug!("X11 focused window class: {instance} (class field empty, used instance)");
                Some(instance)
            }
            None => {
                debug!("X11 focused window class: empty (WM_CLASS has neither class nor instance)");
                None
            }
        }
    }

    fn focus_window(&self, id: &str) -> anyhow::Result<()> {
        let window_id: u32 = id
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid X11 window ID: {id}"))?;

        debug!("focusing X11 window: 0x{window_id:x}");

        self.conn
            .set_input_focus(InputFocus::PARENT, window_id, x11rb::CURRENT_TIME)?;
        self.conn
            .flush()
            .context("failed to flush X11 connection")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use x11rb::protocol::xproto::GetPropertyReply;

    /// Build the `WM_CLASS` reply an X server would send, so `WmClass` can be
    /// exercised end to end without a display.
    fn wm_class_reply(value: &[u8]) -> GetPropertyReply {
        GetPropertyReply {
            format: 8,
            type_: AtomEnum::STRING.into(),
            value_len: value.len() as u32,
            value: value.to_vec(),
            ..GetPropertyReply::default()
        }
    }

    /// Split a raw `WM_CLASS` property the way `WmClass::get` would.
    fn wm_class_from(value: &[u8]) -> WmClass {
        WmClass::from_reply(wm_class_reply(value))
            .expect("WM_CLASS reply should parse")
            .expect("STRING-typed reply should yield a WmClass")
    }

    #[test]
    fn parses_a_plain_class() {
        assert_eq!(parse_wm_class(b"Alacritty"), Some("Alacritty".to_string()));
    }

    #[test]
    fn trailing_nul_and_no_trailing_nul_agree() {
        // x11rb strips at most one trailing NUL, so both WM_CLASS spellings
        // reach us as the same class and must resolve identically.
        assert_eq!(parse_wm_class(b"World"), Some("World".to_string()));
        assert_eq!(parse_wm_class(b"World\0"), Some("World".to_string()));
    }

    #[test]
    fn truncates_at_the_first_interior_nul() {
        // A malformed WM_CLASS with more than two fields, raw property
        // b"Hello\0World\0Good\0Day": x11rb splits at the first NUL only, so
        // the class half reaches parse_wm_class as the bytes below, interior
        // NULs intact. A string holding a NUL could never match a terminal-list
        // entry, so we cut at the first one and keep the leading field.
        assert_eq!(
            parse_wm_class(b"World\0Good\0Day"),
            Some("World".to_string())
        );
    }

    #[test]
    fn empty_input_is_none() {
        assert_eq!(parse_wm_class(b""), None);
    }

    #[test]
    fn whitespace_only_input_is_none() {
        assert_eq!(parse_wm_class(b"   \t \n "), None);
        // A lone NUL truncates to nothing, which is also nothing usable.
        assert_eq!(parse_wm_class(b"\0"), None);
    }

    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(parse_wm_class(b"  XTerm  "), Some("XTerm".to_string()));
    }

    #[test]
    fn non_utf8_bytes_do_not_panic() {
        // WM_CLASS is not guaranteed UTF-8. Lossy decoding keeps the window
        // identifiable instead of dropping it.
        let class = parse_wm_class(b"Alac\xffritty").expect("lossy decode should yield a class");
        assert!(class.starts_with("Alac"));
        assert!(class.ends_with("ritty"));
    }

    #[test]
    fn reports_the_class_field_not_the_instance() {
        // xterm's real WM_CLASS: instance "xterm", class "XTerm".
        let wm_class = wm_class_from(b"xterm\0XTerm\0");

        assert_eq!(
            parse_wm_class(wm_class.class()),
            Some("XTerm".to_string()),
            "the class half must win"
        );
        assert_eq!(
            parse_wm_class(wm_class.instance()),
            Some("xterm".to_string())
        );
    }

    #[test]
    fn class_from_wm_class_prefers_class_over_instance() {
        // The precedence itself, not just the halves it picks between:
        // swapping the two branches in class_from_wm_class turns xterm's real
        // WM_CLASS into Some("xterm") and fails here. The Sway backend has the
        // matching test, and the two orders have to stay in step.
        assert_eq!(
            class_from_wm_class(&wm_class_from(b"xterm\0XTerm\0")),
            Some(("XTerm".to_string(), ClassField::Class))
        );
    }

    #[test]
    fn falls_back_to_the_instance_when_the_class_half_is_absent() {
        // A window that set only one field: x11rb reports an empty class, so
        // the instance is all we have.
        let wm_class = wm_class_from(b"Hello World");

        assert_eq!(parse_wm_class(wm_class.class()), None);
        assert_eq!(
            parse_wm_class(wm_class.instance()),
            Some("Hello World".to_string())
        );

        // The same shape with the separator NUL present, this time through the
        // precedence helper rather than a single half.
        assert_eq!(
            class_from_wm_class(&wm_class_from(b"xterm\0")),
            Some(("xterm".to_string(), ClassField::Instance))
        );
    }

    #[test]
    fn an_empty_wm_class_property_yields_nothing() {
        let wm_class = wm_class_from(b"");

        assert_eq!(parse_wm_class(wm_class.class()), None);
        assert_eq!(parse_wm_class(wm_class.instance()), None);

        // Both halves empty, with and without the separating NUL.
        assert_eq!(class_from_wm_class(&wm_class), None);
        assert_eq!(class_from_wm_class(&wm_class_from(b"\0")), None);
    }
}
