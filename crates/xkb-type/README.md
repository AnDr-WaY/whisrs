# xkb-type

Layout-aware text injection for Linux — types correct characters regardless of the user's active keyboard layout.

## Problem

Linux has no "type this text at the cursor" API. Every tool that injects text (wtype, ydotool, xdotool) sends fixed keycodes, which produce wrong characters when the user has a non-US keyboard layout. German users get "z" and "y" swapped; French users get "a" and "q" swapped; Russian users get complete gibberish.

## Solution

`xkb-type` uses `xkbcommon` to read the active XKB keyboard layout, builds a reverse mapping from every Unicode character to the physical key + modifiers needed to produce it, then injects the correct keystrokes via a Linux uinput virtual keyboard.

Characters not found in the keymap (emoji, rare symbols) are pasted via clipboard (Ctrl+V).

## Usage

```rust
use xkb_type::{Keyboard, Key};
use std::time::Duration;

// Auto-detect layout and clipboard backend.
let mut kb = Keyboard::new(Duration::from_millis(2))?;
kb.type_text("Hello — こんにちは — €100 — 😀")?;
kb.backspace(5)?;
kb.send_combo(&[Key::KEY_LEFTCTRL, Key::KEY_C])?;
```

To force a specific XKB layout (skips compositor probe):

```rust
use xkb_type::Keyboard;
use std::time::Duration;

let mut kb = Keyboard::with_layout("de", None, Duration::from_millis(5))?;
kb.type_text("Schöne Grüße")?;
```

To inject your own keymap and clipboard backend:

```rust
use xkb_type::{Keyboard, XkbKeymap, KeyboardLayout, default_clipboard};
use std::time::Duration;

let layout = KeyboardLayout::detect();
let keymap = XkbKeymap::from_layout(&layout)?;
let mut kb = Keyboard::with_components(keymap, default_clipboard(), Duration::from_millis(5))?;
```

To pause or stop between keys, type through the `KeyInjector` trait. Its `type_text` asks a gate before every key with the byte offset of the next character; the gate may block, and `ControlFlow::Break` stops before that character:

```rust
use std::ops::ControlFlow;
use std::time::Duration;
use xkb_type::{KeyInjector, Keyboard, Typed};

let mut kb: Box<dyn KeyInjector> = Box::new(Keyboard::new(Duration::from_millis(2))?);
let text = "hello world";
let mut gate = |at: usize| if at < 5 { ControlFlow::Continue(()) } else { ControlFlow::Break(()) };
if let Typed::StoppedAt(at) = kb.type_text(text, &mut gate)? {
    println!("not typed: {:?}", &text[at..]);
}
```

## Requirements

- Linux
- Write access to `/dev/uinput` (user in `input` group or udev rule)
- `libxkbcommon` (installed by default on most Linux desktops)
- `wl-clipboard` for Wayland clipboard operations

## Installation

```fish
cargo add xkb-type
```

## License

MIT
