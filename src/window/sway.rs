//! Sway/i3 window tracking via the `swayipc` crate.

use swayipc::Connection;
use tracing::debug;

use super::WindowTracker;

/// Window tracker for Sway (and i3 with sway-compatible IPC).
///
/// Uses `swayipc` to query the window tree and focus windows by con_id.
pub struct SwayTracker;

impl Default for SwayTracker {
    fn default() -> Self {
        Self
    }
}

impl SwayTracker {
    pub fn new() -> Self {
        Self
    }
}

impl WindowTracker for SwayTracker {
    fn get_focused_window(&self) -> anyhow::Result<String> {
        let mut conn = Connection::new().map_err(|e| anyhow::anyhow!("sway IPC connect: {e}"))?;
        let tree = conn
            .get_tree()
            .map_err(|e| anyhow::anyhow!("sway get_tree: {e}"))?;

        // Walk the tree to find the focused node.
        let focused = find_focused(&tree);
        match focused {
            Some(node) => {
                let id = node.id;
                debug!("sway focused window con_id: {id}");
                Ok(id.to_string())
            }
            None => anyhow::bail!("no focused window found in sway tree"),
        }
    }

    fn get_focused_window_class(&self) -> Option<String> {
        let mut conn = match Connection::new() {
            Ok(conn) => conn,
            Err(err) => {
                debug!("sway focused window class: IPC connect failed: {err}");
                return None;
            }
        };

        let tree = match conn.get_tree() {
            Ok(tree) => tree,
            Err(err) => {
                debug!("sway focused window class: get_tree failed: {err}");
                return None;
            }
        };

        let Some(node) = find_focused(&tree) else {
            debug!("sway focused window class: no focused node in sway tree");
            return None;
        };

        match class_from_node(node) {
            Some(class) => {
                debug!("sway focused window class: {class}");
                Some(class)
            }
            None => {
                debug!(
                    "sway focused window class: empty (focused node reported no app_id, class or instance)"
                );
                None
            }
        }
    }

    fn focus_window(&self, id: &str) -> anyhow::Result<()> {
        let con_id: i64 = id
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid sway con_id: {id}"))?;

        debug!("focusing sway window con_id: {con_id}");

        let mut conn = Connection::new().map_err(|e| anyhow::anyhow!("sway IPC connect: {e}"))?;
        let command = format!("[con_id={con_id}] focus");
        let results = conn
            .run_command(&command)
            .map_err(|e| anyhow::anyhow!("sway run_command: {e}"))?;

        for result in results {
            if let Err(e) = result {
                anyhow::bail!("sway focus command failed: {e}");
            }
        }

        Ok(())
    }
}

/// Recursively find the focused node in the sway tree.
///
/// Depth-first, first match wins: the node itself, then its tiling `nodes`,
/// then its `floating_nodes`.
fn find_focused(node: &swayipc::Node) -> Option<&swayipc::Node> {
    if node.focused {
        return Some(node);
    }
    for child in &node.nodes {
        if let Some(found) = find_focused(child) {
            return Some(found);
        }
    }
    for child in &node.floating_nodes {
        if let Some(found) = find_focused(child) {
            return Some(found);
        }
    }
    None
}

/// Extract a window-class-like identifier from a sway tree node.
///
/// Resolution order:
/// 1. `app_id` — set only for xdg-shell (native Wayland) views.
/// 2. `window_properties.class` — XWayland views only.
/// 3. `window_properties.instance` — XWayland fallback.
///
/// **Why `class` before `instance`:** X11's `WM_CLASS` property is the pair
/// (instance, class), and the X11 backend reports the *class* field. Sway's
/// XWayland fallback has to agree with it, or the same terminal would be
/// classified differently depending on which compositor the user is running.
/// `is_terminal_class` (`src/daemon/injection.rs`) lowercases before matching,
/// so case alone is harmless — but the two fields are not case variants of one
/// another in general, so which one gets reported can decide the verdict.
///
/// A candidate that trims to empty is skipped rather than returned, so a view
/// that advertises an empty `app_id` still gets its XWayland properties
/// consulted. Returns `None` only when no candidate yields a non-empty value.
///
/// Pure: does no IPC, so it is unit-testable against tree fixtures.
fn class_from_node(node: &swayipc::Node) -> Option<String> {
    let window_properties = node.window_properties.as_ref();
    [
        node.app_id.as_deref(),
        window_properties.and_then(|props| props.class.as_deref()),
        window_properties.and_then(|props| props.instance.as_deref()),
    ]
    .into_iter()
    .flatten()
    .map(str::trim)
    .find(|candidate| !candidate.is_empty())
    .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal JSON for a sway tree node. Only the keys without a serde default
    /// are spelled out; `extra` carries the view fields under test and must
    /// start with a comma when non-empty.
    fn node_json(id: i64, focused: bool, nodes: &str, floating_nodes: &str, extra: &str) -> String {
        let rect = r#"{"x":0,"y":0,"width":0,"height":0}"#;
        format!(
            r#"{{"id":{id},"type":"con","border":"none","current_border_width":0,
"layout":"none","rect":{rect},"window_rect":{rect},"deco_rect":{rect},
"geometry":{rect},"urgent":false,"focused":{focused},"focus":[],
"nodes":[{nodes}],"floating_nodes":[{floating_nodes}],"sticky":false{extra}}}"#
        )
    }

    fn parse(json: &str) -> swayipc::Node {
        serde_json::from_str(json).expect("fixture should deserialize as a swayipc::Node")
    }

    /// A focused leaf view carrying only the class-bearing fields.
    fn view(extra: &str) -> swayipc::Node {
        parse(&node_json(1, true, "", "", extra))
    }

    #[test]
    fn class_from_node_uses_app_id_for_wayland_views() {
        let node = view(r#","app_id":"Alacritty""#);

        assert_eq!(class_from_node(&node).as_deref(), Some("Alacritty"));
    }

    #[test]
    fn class_from_node_prefers_class_over_instance_for_xwayland_views() {
        // WM_CLASS is (instance, class); the X11 backend reports class, so this
        // one must too.
        let node =
            view(r#","app_id":null,"window_properties":{"instance":"xterm","class":"XTerm"}"#);

        assert_eq!(class_from_node(&node).as_deref(), Some("XTerm"));
    }

    #[test]
    fn class_from_node_falls_back_to_instance_when_class_is_null() {
        let node = view(r#","app_id":null,"window_properties":{"instance":"xterm","class":null}"#);

        assert_eq!(class_from_node(&node).as_deref(), Some("xterm"));
    }

    #[test]
    fn class_from_node_skips_empty_app_id() {
        let node = view(r#","app_id":"","window_properties":{"instance":"xterm","class":"XTerm"}"#);

        assert_eq!(class_from_node(&node).as_deref(), Some("XTerm"));
    }

    #[test]
    fn class_from_node_skips_whitespace_only_candidates() {
        let node = view(r#","app_id":"  ","window_properties":{"instance":"xterm","class":"   "}"#);

        assert_eq!(class_from_node(&node).as_deref(), Some("xterm"));
    }

    #[test]
    fn class_from_node_returns_none_when_nothing_is_reported() {
        let bare = view("");
        assert_eq!(class_from_node(&bare), None);

        let all_null = view(r#","app_id":null,"window_properties":{"instance":null,"class":null}"#);
        assert_eq!(class_from_node(&all_null), None);
    }

    #[test]
    fn find_focused_walks_into_tiling_children() {
        let leaf = node_json(3, true, "", "", r#","app_id":"Alacritty""#);
        let workspace = node_json(2, false, &leaf, "", "");
        let root = parse(&node_json(1, false, &workspace, "", ""));

        let focused = find_focused(&root).expect("nested tiling leaf should be found");

        assert_eq!(focused.id, 3);
        assert_eq!(class_from_node(focused).as_deref(), Some("Alacritty"));
    }

    #[test]
    fn find_focused_walks_into_floating_children() {
        let floating_leaf = node_json(4, true, "", "", r#","app_id":"org.gnome.Nautilus""#);
        let workspace = node_json(2, false, "", &floating_leaf, "");
        let root = parse(&node_json(1, false, &workspace, "", ""));

        let focused = find_focused(&root).expect("nested floating leaf should be found");

        assert_eq!(focused.id, 4);
        assert_eq!(
            class_from_node(focused).as_deref(),
            Some("org.gnome.Nautilus")
        );
    }

    #[test]
    fn find_focused_returns_none_when_nothing_is_focused() {
        let leaf = node_json(3, false, "", "", r#","app_id":"Alacritty""#);
        let floating_leaf = node_json(4, false, "", "", r#","app_id":"firefox""#);
        let workspace = node_json(2, false, &leaf, &floating_leaf, "");
        let root = parse(&node_json(1, false, &workspace, "", ""));

        assert!(find_focused(&root).is_none());
    }
}
