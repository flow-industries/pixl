//! Interactive terminal-image gallery.
//!
//! Shows generated pixel art inline (Kitty/iTerm2/Sixel via `ratatui-image`,
//! unicode-halfblock fallback elsewhere), navigable with the arrow keys, with
//! save / rerun / edit-prompt actions. `run_live` streams images as a background
//! [`actor`] generates them; `run_static` browses a finished directory.

#[cfg(feature = "gen")]
pub mod actor;
pub mod gallery;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Duration;

#[cfg(feature = "gen")]
use std::collections::HashSet;
#[cfg(feature = "gen")]
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{Resize, StatefulImage};

use gallery::{save_dest, Gallery, Slot};

#[cfg(feature = "gen")]
#[derive(Clone)]
enum Status {
    Loading,
    Downloading {
        file: String,
        done: u64,
        total: u64,
    },
    Generating {
        done: usize,
        total: u32,
        step: usize,
        steps: usize,
    },
    Idle,
    Error(String),
}

/// A toggleable prompt modifier: appends `pos` to the prompt and `neg` to the
/// negative prompt when enabled. Negatives only bite on `--model sdxl` (cfg > 1).
#[cfg(feature = "gen")]
struct PromptMod {
    label: &'static str,
    pos: &'static str,
    neg: &'static str,
}

#[cfg(feature = "gen")]
const MODIFIERS: &[PromptMod] = &[
    PromptMod {
        label: "single subject (isolated, centered)",
        pos: "a single subject, centered, isolated object",
        neg: "scene, multiple objects, collage, tilesheet, grid",
    },
    PromptMod {
        label: "plain background",
        pos: "on a plain solid-color background, simple background",
        neg: "landscape, scenery, background details",
    },
    PromptMod {
        label: "item icon framing",
        pos: "game item icon, inventory sprite, game asset",
        neg: "",
    },
    PromptMod {
        label: "no shadow / ground",
        pos: "",
        neg: "shadow, ground, grass, floor",
    },
    PromptMod {
        label: "keyable magenta background",
        pos: "on a solid flat magenta background",
        neg: "",
    },
];

#[cfg(feature = "gen")]
const PARAM_LABELS: &[&str] = &["count", "cfg", "steps", "colors", "seed"];

/// Editable generation settings, persisted to `~/.pixl/config.json` so a no-arg
/// run picks up the last configuration.
#[cfg(feature = "gen")]
#[derive(Clone)]
struct Settings {
    count: u32,
    cfg: f32,
    steps: u32,
    colors: u16,
    seed: u64,
    mods: Vec<bool>,
}

#[cfg(feature = "gen")]
fn config_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".pixl").join("config.json"))
}

#[cfg(feature = "gen")]
impl Settings {
    fn placeholder() -> Self {
        Self {
            count: 4,
            cfg: 1.0,
            steps: 8,
            colors: 16,
            seed: 0,
            mods: vec![false; MODIFIERS.len()],
        }
    }

    /// Defaults (model-aware) <- persisted config <- explicit CLI flags.
    fn load(args: &crate::cli::GenerateArgs, cli_count: Option<u32>) -> Self {
        let mut s = Self {
            count: cli_count.unwrap_or(4),
            cfg: crate::resolve_cfg(args),
            steps: crate::resolve_steps(args),
            colors: args.colors.unwrap_or(16),
            seed: args.seed.unwrap_or(0),
            mods: vec![false; MODIFIERS.len()],
        };
        if let Some(v) = config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        {
            if let Some(n) = v["count"].as_u64() {
                s.count = n as u32;
            }
            if let Some(n) = v["cfg"].as_f64() {
                s.cfg = n as f32;
            }
            if let Some(n) = v["steps"].as_u64() {
                s.steps = n as u32;
            }
            if let Some(n) = v["colors"].as_u64() {
                s.colors = n as u16;
            }
            if let Some(n) = v["seed"].as_u64() {
                s.seed = n;
            }
            if let Some(arr) = v["mods"].as_array() {
                let mut m: Vec<bool> = arr.iter().map(|b| b.as_bool().unwrap_or(false)).collect();
                m.resize(MODIFIERS.len(), false);
                s.mods = m;
            }
        }
        // explicit CLI flags win over the persisted config for this run
        if let Some(n) = cli_count {
            s.count = n;
        }
        if let Some(c) = args.cfg {
            s.cfg = c;
        }
        if let Some(n) = args.steps {
            s.steps = n;
        }
        if let Some(c) = args.colors {
            s.colors = c;
        }
        if let Some(n) = args.seed {
            s.seed = n;
        }
        s
    }

    fn save(&self) {
        let Some(path) = config_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let v = serde_json::json!({
            "count": self.count,
            "cfg": self.cfg,
            "steps": self.steps,
            "colors": self.colors,
            "seed": self.seed,
            "mods": self.mods,
        });
        if let Ok(t) = serde_json::to_string_pretty(&v) {
            let _ = std::fs::write(path, t);
        }
    }
}

/// The currently displayed image: the decoded source is kept resident so pane
/// resizes can re-derive the scale without touching disk, plus a protocol
/// pre-scaled by the largest integer factor that fits the pane (a clean, uniform
/// pixel grid, never a fractional upscale).
struct Shown {
    index: usize,
    src: image::DynamicImage,
    scale: u32,
    /// Footprint of the scaled image in terminal cells, for centering.
    cells: (u16, u16),
    proto: StatefulProtocol,
}

/// Largest integer factor `k` such that the image at `k` times still fits the pane.
fn integer_scale(iw: u32, ih: u32, area: Rect, font: ratatui_image::FontSize) -> u32 {
    if iw == 0 || ih == 0 {
        return 1;
    }
    let pane_w = (area.width as u32).saturating_mul(font.width as u32);
    let pane_h = (area.height as u32).saturating_mul(font.height as u32);
    (pane_w / iw).min(pane_h / ih).max(1)
}

/// Centered rect (in cells) where an `iw`x`ih` image renders at its integer scale.
#[cfg(feature = "gen")]
fn image_rect(area: Rect, iw: u32, ih: u32, font: ratatui_image::FontSize) -> Rect {
    let scale = integer_scale(iw, ih, area, font);
    let w = ((iw * scale).div_ceil(font.width.max(1) as u32) as u16).min(area.width);
    let h = ((ih * scale).div_ceil(font.height.max(1) as u32) as u16).min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(w) / 2,
        y: area.y + area.height.saturating_sub(h) / 2,
        width: w,
        height: h,
    }
}

#[cfg(feature = "gen")]
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Truncate `s` to at most `max` columns, ending with an ellipsis if it was cut.
fn truncate_ellipsis(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

struct App {
    gallery: Gallery,
    picker: Picker,
    /// The currently shown image (only the visible one is kept resident).
    proto: Option<Shown>,
    saved_dir: PathBuf,
    toast: Option<String>,
    quit: bool,

    #[cfg(feature = "gen")]
    static_mode: bool,
    #[cfg(feature = "gen")]
    actor: Option<actor::Actor>,
    #[cfg(feature = "gen")]
    status: Status,
    #[cfg(feature = "gen")]
    input: Option<String>,
    #[cfg(feature = "gen")]
    last_prompt: String,
    #[cfg(feature = "gen")]
    base_negative: String,
    #[cfg(feature = "gen")]
    settings: Settings,
    #[cfg(feature = "gen")]
    model_line: Option<String>,
    #[cfg(feature = "gen")]
    panel: bool,
    #[cfg(feature = "gen")]
    panel_row: usize,
    #[cfg(feature = "gen")]
    gen_size: (u32, u32),
    #[cfg(feature = "gen")]
    tick: u64,
    #[cfg(feature = "gen")]
    skip: Arc<Mutex<HashSet<usize>>>,
    #[cfg(feature = "gen")]
    out_dir: PathBuf,
}

impl App {
    fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        let mut dirty = true;
        while !self.quit {
            #[cfg(feature = "gen")]
            {
                let mut evs = Vec::new();
                if let Some(a) = &self.actor {
                    while let Ok(ev) = a.events.try_recv() {
                        evs.push(ev);
                    }
                }
                if !evs.is_empty() {
                    dirty = true;
                }
                for ev in evs {
                    self.apply(ev);
                }
                if self.viewing_placeholder() {
                    dirty = true;
                }
            }

            if dirty {
                #[cfg(feature = "gen")]
                {
                    self.tick = self.tick.wrapping_add(1);
                }
                terminal.draw(|f| self.render(f))?;
                dirty = false;
            }

            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    Event::Key(k) if k.kind != KeyEventKind::Release => {
                        self.on_key(k);
                        dirty = true;
                    }
                    Event::Resize(_, _) => dirty = true,
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Decode the current image (if changed) and (re)build its protocol pre-scaled
    /// to the largest integer multiple that fits `area`.
    fn ensure_proto(&mut self, area: Rect) {
        let path = match self.gallery.current_done() {
            Some(e) => e.path.clone(),
            None => {
                self.proto = None;
                return;
            }
        };
        let idx = self.gallery.current;
        let font = self.picker.font_size();

        // Nothing to do if it's the same image already built at the right scale.
        if let Some(s) = &self.proto {
            if s.index == idx && s.scale == integer_scale(s.src.width(), s.src.height(), area, font)
            {
                return;
            }
        }

        // Reuse the decoded source across pane resizes; hit disk only on a new image.
        let src = match &self.proto {
            Some(s) if s.index == idx => s.src.clone(),
            _ => match image::open(&path) {
                Ok(img) => img,
                Err(_) => {
                    self.proto = None;
                    return;
                }
            },
        };

        let scale = integer_scale(src.width(), src.height(), area, font);
        let scaled = if scale <= 1 {
            src.clone()
        } else {
            src.resize_exact(
                src.width() * scale,
                src.height() * scale,
                image::imageops::FilterType::Nearest,
            )
        };
        let cells = (
            scaled.width().div_ceil(font.width.max(1) as u32) as u16,
            scaled.height().div_ceil(font.height.max(1) as u32) as u16,
        );
        let proto = self.picker.new_resize_protocol(scaled);
        self.proto = Some(Shown {
            index: idx,
            src,
            scale,
            cells,
            proto,
        });
    }

    fn render(&mut self, f: &mut Frame) {
        let chunks = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(f.area());

        self.render_title(f, chunks[0]);
        self.render_main(f, chunks[1]);
        f.render_widget(self.status_line(), chunks[2]);
        self.render_footer(f, chunks[3]);
    }

    fn render_title(&self, f: &mut Frame, area: Rect) {
        let total = self.gallery.len();
        let idx = if total == 0 {
            0
        } else {
            self.gallery.current + 1
        };
        let saved = self
            .gallery
            .current_done()
            .map(|e| e.saved)
            .unwrap_or(false);

        let mut spans = vec![
            Span::styled(
                " pixl ",
                Style::new()
                    .bg(Color::Cyan)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(" {idx}/{total}")),
        ];
        if saved {
            spans.push(Span::styled("  saved", Style::new().fg(Color::Green)));
        }
        // Truncate the prompt to the remaining width so the title stays one line.
        let used = 6 + format!(" {idx}/{total}").chars().count() + if saved { 7 } else { 0 } + 2;
        let budget = (area.width as usize).saturating_sub(used);
        let prompt = truncate_ellipsis(&self.title_prompt(), budget);
        if !prompt.is_empty() {
            spans.push(Span::styled(
                format!("  {prompt}"),
                Style::new().fg(Color::Gray),
            ));
        }
        f.render_widget(Line::from(spans), area);
    }

    fn render_image(&mut self, f: &mut Frame, area: Rect) {
        self.ensure_proto(area);
        if let Some(shown) = &mut self.proto {
            let w = shown.cells.0.min(area.width);
            let h = shown.cells.1.min(area.height);
            let rect = Rect {
                x: area.x + area.width.saturating_sub(w) / 2,
                y: area.y + area.height.saturating_sub(h) / 2,
                width: w,
                height: h,
            };
            f.render_stateful_widget(
                StatefulImage::default().resize(Resize::Fit(None)),
                rect,
                &mut shown.proto,
            );
        } else {
            let msg = if self.gallery.is_empty() {
                "waiting for the first image…"
            } else {
                "could not load image"
            };
            f.render_widget(
                Paragraph::new(msg)
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true }),
                area,
            );
        }
    }

    fn render_main(&mut self, f: &mut Frame, area: Rect) {
        #[cfg(feature = "gen")]
        if self.panel {
            self.render_panel(f, area);
            return;
        }
        if self.gallery.is_empty() {
            f.render_widget(
                Paragraph::new(self.idle_message())
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true }),
                area,
            );
            return;
        }
        if matches!(self.gallery.current_slot(), Some(Slot::Done(_))) {
            self.render_image(f, area);
        } else {
            #[cfg(feature = "gen")]
            self.render_pending(f, area);
        }
    }

    fn idle_message(&self) -> &'static str {
        #[cfg(feature = "gen")]
        if self.last_prompt.trim().is_empty() {
            return "press e to set a prompt · t for settings";
        }
        "waiting for the first image…"
    }

    fn title_prompt(&self) -> String {
        if let Some(e) = self.gallery.current_done() {
            return e.prompt.clone();
        }
        #[cfg(feature = "gen")]
        {
            self.last_prompt.clone()
        }
        #[cfg(not(feature = "gen"))]
        {
            String::new()
        }
    }

    #[cfg(feature = "gen")]
    fn viewing_placeholder(&self) -> bool {
        matches!(
            self.gallery.current_slot(),
            Some(Slot::Queued) | Some(Slot::Generating(None))
        )
    }

    #[cfg(feature = "gen")]
    fn render_pending(&mut self, f: &mut Frame, area: Rect) {
        let preview = match self.gallery.current_slot() {
            Some(Slot::Generating(Some(img))) => Some(img.clone()),
            _ => None,
        };
        if let Some(img) = preview {
            self.render_preview(f, area, img);
        } else {
            let label = match self.gallery.current_slot() {
                Some(Slot::Queued) => "queued",
                _ => "generating",
            };
            self.render_placeholder(f, area, label);
        }
    }

    #[cfg(feature = "gen")]
    fn render_placeholder(&self, f: &mut Frame, area: Rect, label: &str) {
        let (iw, ih) = self.gen_size;
        let rect = image_rect(area, iw.max(1), ih.max(1), self.picker.font_size());
        let block =
            ratatui::widgets::Block::bordered().border_style(Style::new().fg(Color::DarkGray));
        let inner = block.inner(rect);
        f.render_widget(block, rect);
        let frame = SPINNER[(self.tick as usize) % SPINNER.len()];
        let line = Line::from(Span::styled(
            format!("{frame} {label}"),
            Style::new().fg(Color::DarkGray),
        ));
        let mid = Rect {
            x: inner.x,
            y: inner.y + inner.height / 2,
            width: inner.width.max(1),
            height: 1,
        };
        f.render_widget(Paragraph::new(line).alignment(Alignment::Center), mid);
    }

    #[cfg(feature = "gen")]
    fn render_preview(&mut self, f: &mut Frame, area: Rect, src: image::DynamicImage) {
        // Fill + center exactly like the final image (the preview is a real VAE
        // decode at the same resolution, so it lines up when it snaps in).
        let font = self.picker.font_size();
        let scale = integer_scale(src.width(), src.height(), area, font);
        let scaled = if scale <= 1 {
            src
        } else {
            src.resize_exact(
                src.width() * scale,
                src.height() * scale,
                image::imageops::FilterType::Nearest,
            )
        };
        let w = (scaled.width().div_ceil(font.width.max(1) as u32) as u16).min(area.width);
        let h = (scaled.height().div_ceil(font.height.max(1) as u32) as u16).min(area.height);
        let rect = Rect {
            x: area.x + area.width.saturating_sub(w) / 2,
            y: area.y + area.height.saturating_sub(h) / 2,
            width: w,
            height: h,
        };
        let mut proto = self.picker.new_resize_protocol(scaled);
        f.render_stateful_widget(
            StatefulImage::default().resize(Resize::Fit(None)),
            rect,
            &mut proto,
        );
    }

    #[cfg(feature = "gen")]
    fn param_value(&self, i: usize) -> String {
        match i {
            0 => self.settings.count.to_string(),
            1 => format!("{:.1}", self.settings.cfg),
            2 => self.settings.steps.to_string(),
            3 => {
                if self.settings.colors == 0 {
                    "all".into()
                } else {
                    self.settings.colors.to_string()
                }
            }
            4 => self.settings.seed.to_string(),
            _ => String::new(),
        }
    }

    #[cfg(feature = "gen")]
    fn render_panel(&self, f: &mut Frame, area: Rect) {
        let np = PARAM_LABELS.len();
        let mut lines = vec![
            Line::from(Span::styled(
                "settings",
                Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "↑↓ select · ←→ adjust · space toggle · Enter generate · Esc close",
                Style::new().fg(Color::DarkGray),
            )),
            Line::from(""),
        ];
        for (i, label) in PARAM_LABELS.iter().enumerate() {
            let sel = self.panel_row == i;
            let cur = if sel { "›" } else { " " };
            let lstyle = if sel {
                Style::new().fg(Color::Cyan)
            } else {
                Style::new().fg(Color::Gray)
            };
            lines.push(Line::from(vec![
                Span::styled(format!(" {cur} {label:<7} "), lstyle),
                Span::styled(self.param_value(i), Style::new().fg(Color::White)),
            ]));
        }
        lines.push(Line::from(""));
        for (j, m) in MODIFIERS.iter().enumerate() {
            let sel = self.panel_row == np + j;
            let on = self.settings.mods.get(j).copied().unwrap_or(false);
            let cur = if sel { "›" } else { " " };
            let mark = if on { "[x]" } else { "[ ]" };
            let style = if on {
                Style::new().fg(Color::Green)
            } else if sel {
                Style::new().fg(Color::Cyan)
            } else {
                Style::new().fg(Color::Gray)
            };
            lines.push(Line::from(vec![
                Span::styled(format!(" {cur} {mark} "), style),
                Span::raw(m.label),
            ]));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "negative-prompt toggles need --model sdxl (cfg > 1)",
            Style::new().fg(Color::DarkGray),
        )));
        f.render_widget(Paragraph::new(lines), area);
    }

    fn render_footer(&self, f: &mut Frame, area: Rect) {
        #[cfg(feature = "gen")]
        if let Some(input) = &self.input {
            let label = "new prompt: ";
            // Keep the cursor (end of input) on screen: scroll to show the tail.
            let avail = (area.width as usize).saturating_sub(label.chars().count() + 1);
            let chars: Vec<char> = input.chars().collect();
            let shown: String = if chars.len() > avail {
                let start = chars.len() - avail.saturating_sub(1);
                let mut t = String::from("…");
                t.extend(&chars[start..]);
                t
            } else {
                input.clone()
            };
            let line = Line::from(vec![
                Span::styled(label, Style::new().fg(Color::Cyan)),
                Span::raw(shown),
                Span::styled("▌", Style::new().fg(Color::Cyan)),
            ]);
            f.render_widget(Paragraph::new(line), area);
            return;
        }
        f.render_widget(
            Paragraph::new(Span::styled(
                self.footer_hints(),
                Style::new().fg(Color::DarkGray),
            )),
            area,
        );
    }

    fn footer_hints(&self) -> &'static str {
        #[cfg(feature = "gen")]
        {
            if self.static_mode {
                "←/→ nav · s save · q quit"
            } else {
                "←/→ nav · s save · x discard · t settings · r rerun · e edit · c cancel · q quit"
            }
        }
        #[cfg(not(feature = "gen"))]
        {
            "←/→ nav · s save · q quit"
        }
    }

    #[cfg(feature = "gen")]
    fn status_line(&self) -> Line<'static> {
        if let Some(t) = &self.toast {
            return Line::from(Span::styled(t.clone(), Style::new().fg(Color::Yellow)));
        }
        match &self.status {
            Status::Loading => {
                Line::from(Span::styled("loading model…", Style::new().fg(Color::Cyan)))
            }
            Status::Downloading { file, done, total } => {
                let txt = if *total > 0 {
                    format!(
                        "downloading {file}  {} / {}",
                        crate::human(*done),
                        crate::human(*total)
                    )
                } else {
                    format!("downloading {file}  {}", crate::human(*done))
                };
                Line::from(Span::styled(txt, Style::new().fg(Color::Cyan)))
            }
            Status::Generating {
                done,
                total,
                step,
                steps,
            } => Line::from(Span::styled(
                format!("generating {}/{total} · step {step}/{steps}", done + 1),
                Style::new().fg(Color::Cyan),
            )),
            Status::Idle => {
                let s = self.model_line.clone().unwrap_or_else(|| "idle".into());
                Line::from(Span::styled(s, Style::new().fg(Color::DarkGray)))
            }
            Status::Error(e) => Line::from(Span::styled(
                format!("error: {e}"),
                Style::new().fg(Color::Red),
            )),
        }
    }

    #[cfg(not(feature = "gen"))]
    fn status_line(&self) -> Line<'static> {
        if let Some(t) = &self.toast {
            return Line::from(Span::styled(t.clone(), Style::new().fg(Color::Yellow)));
        }
        Line::from(Span::styled("viewing", Style::new().fg(Color::DarkGray)))
    }

    fn on_key(&mut self, key: KeyEvent) {
        self.toast = None;

        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
        }

        #[cfg(feature = "gen")]
        if self.input.is_some() {
            self.on_key_input(key);
            return;
        }
        #[cfg(feature = "gen")]
        if self.panel {
            self.on_key_panel(key);
            return;
        }

        // Navigation repeats on a held key; one-shot actions do not, so holding
        // `r` can't queue a runaway of reruns.
        let repeat = key.kind == KeyEventKind::Repeat;
        match key.code {
            KeyCode::Left | KeyCode::Char('h') => self.gallery.prev(),
            KeyCode::Right | KeyCode::Char('l') => self.gallery.next(),
            KeyCode::Home | KeyCode::Char('g') => self.gallery.first(),
            KeyCode::End | KeyCode::Char('G') => self.gallery.last(),
            KeyCode::Char('s') | KeyCode::Char(' ') if !repeat => self.save_current(),
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            #[cfg(feature = "gen")]
            KeyCode::Char('t') if !repeat => {
                if !self.static_mode {
                    self.panel = true;
                }
            }
            #[cfg(feature = "gen")]
            KeyCode::Char('r') if !repeat => self.rerun(),
            #[cfg(feature = "gen")]
            KeyCode::Char('e') if !repeat => self.begin_edit(),
            #[cfg(feature = "gen")]
            KeyCode::Char('c') if !repeat => self.cancel_gen(),
            #[cfg(feature = "gen")]
            KeyCode::Char('x') if !repeat => self.discard_current(),
            _ => {}
        }
    }

    fn save_current(&mut self) {
        let Some(entry) = self.gallery.current_done().cloned() else {
            self.toast = Some("nothing to save yet".into());
            return;
        };
        let dest = save_dest(&self.saved_dir, &entry, |p| p.exists());
        let res = std::fs::create_dir_all(&self.saved_dir)
            .and_then(|_| std::fs::copy(&entry.path, &dest).map(|_| ()));
        match res {
            Ok(()) => {
                if let Some(e) = self.gallery.current_done_mut() {
                    e.saved = true;
                }
                self.toast = Some(format!("saved → {}", dest.display()));
            }
            Err(e) => self.toast = Some(format!("save failed: {e}")),
        }
    }

    #[cfg(feature = "gen")]
    fn rerun(&mut self) {
        if self.static_mode {
            return;
        }
        self.submit();
    }

    #[cfg(feature = "gen")]
    fn on_key_panel(&mut self, key: KeyEvent) {
        let total = PARAM_LABELS.len() + MODIFIERS.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.panel_row = self.panel_row.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                if self.panel_row + 1 < total {
                    self.panel_row += 1;
                }
            }
            KeyCode::Left => self.adjust(false),
            KeyCode::Right => self.adjust(true),
            KeyCode::Char(' ') => self.toggle_row(),
            KeyCode::Enter => {
                self.panel = false;
                self.settings.save();
                self.submit();
            }
            KeyCode::Esc | KeyCode::Char('t') => {
                self.panel = false;
                self.settings.save();
            }
            _ => {}
        }
    }

    #[cfg(feature = "gen")]
    fn adjust(&mut self, up: bool) {
        let np = PARAM_LABELS.len();
        if self.panel_row >= np {
            self.toggle_row();
            return;
        }
        let s = &mut self.settings;
        match self.panel_row {
            0 => {
                s.count = if up {
                    (s.count + 1).min(64)
                } else {
                    s.count.saturating_sub(1).max(1)
                }
            }
            1 => {
                s.cfg = if up {
                    (s.cfg + 0.5).min(15.0)
                } else {
                    (s.cfg - 0.5).max(1.0)
                }
            }
            2 => {
                s.steps = if up {
                    (s.steps + 1).min(60)
                } else {
                    s.steps.saturating_sub(1).max(1)
                }
            }
            3 => {
                s.colors = if up {
                    (s.colors + 1).min(64)
                } else {
                    s.colors.saturating_sub(1)
                }
            }
            4 => {
                s.seed = if up {
                    s.seed.wrapping_add(1)
                } else {
                    s.seed.wrapping_sub(1)
                }
            }
            _ => {}
        }
    }

    #[cfg(feature = "gen")]
    fn toggle_row(&mut self) {
        let np = PARAM_LABELS.len();
        if self.panel_row >= np {
            let i = self.panel_row - np;
            if let Some(b) = self.settings.mods.get_mut(i) {
                *b = !*b;
            }
        }
    }

    /// Compose the full (prompt, negative) sent to the generator. The base
    /// subject + base negative for now; the modifier toggles fold in here.
    #[cfg(feature = "gen")]
    fn compose(&self) -> (String, String) {
        let mut pos = self.last_prompt.clone();
        let mut neg: Vec<String> = Vec::new();
        if !self.base_negative.trim().is_empty() {
            neg.push(self.base_negative.clone());
        }
        for (i, m) in MODIFIERS.iter().enumerate() {
            if self.settings.mods.get(i).copied().unwrap_or(false) {
                if !m.pos.is_empty() {
                    pos.push_str(", ");
                    pos.push_str(m.pos);
                }
                if !m.neg.is_empty() {
                    neg.push(m.neg.to_string());
                }
            }
        }
        (pos, neg.join(", "))
    }

    #[cfg(feature = "gen")]
    fn submit(&self) {
        if self.last_prompt.trim().is_empty() {
            return;
        }
        if let Some(a) = &self.actor {
            let (p, n) = self.compose();
            let s = &self.settings;
            a.generate(p, n, s.count, s.cfg, s.steps, s.seed, s.colors);
        }
    }

    #[cfg(feature = "gen")]
    fn begin_edit(&mut self) {
        if self.static_mode {
            return;
        }
        self.input = Some(self.last_prompt.clone());
    }

    #[cfg(feature = "gen")]
    fn cancel_gen(&mut self) {
        if let Some(a) = &self.actor {
            a.cancel();
        }
    }

    /// Discard the current slot: drop it from the gallery, tell the generator to
    /// skip it (queued) or drop its result (in-flight), and delete its run file
    /// if already written.
    #[cfg(feature = "gen")]
    fn discard_current(&mut self) {
        if let Some((id, slot)) = self.gallery.remove_current() {
            if !self.static_mode {
                if let Ok(mut set) = self.skip.lock() {
                    set.insert(id);
                }
                if let Slot::Done(e) = &slot {
                    if e.path.starts_with(&self.out_dir) {
                        let _ = std::fs::remove_file(&e.path);
                    }
                }
            }
            self.toast = Some("discarded".into());
        }
    }

    #[cfg(feature = "gen")]
    fn on_key_input(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(c) => {
                if let Some(s) = &mut self.input {
                    s.push(c);
                }
            }
            KeyCode::Backspace => {
                if let Some(s) = &mut self.input {
                    s.pop();
                }
            }
            KeyCode::Enter => {
                if let Some(s) = self.input.take() {
                    let p = s.trim().to_string();
                    if !p.is_empty() {
                        self.last_prompt = p;
                        self.submit();
                    }
                }
            }
            KeyCode::Esc => self.input = None,
            _ => {}
        }
    }

    #[cfg(feature = "gen")]
    fn apply(&mut self, ev: actor::GenEvent) {
        use actor::GenEvent::*;
        match ev {
            Loading => self.status = Status::Loading,
            Download { file, done, total } => {
                self.status = Status::Downloading { file, done, total }
            }
            Loaded {
                model,
                cached,
                lora,
                merged,
            } => {
                let mut s = format!("{model} ({})", if cached { "cached" } else { "fetched" });
                if let Some((name, scale)) = lora {
                    s.push_str(&format!(" · lora {name}@{scale}"));
                }
                if merged {
                    s.push_str(" · merged");
                }
                self.model_line = Some(s);
                self.status = Status::Idle;
            }
            BatchStarted { start, count } => {
                self.gallery.push_queued(start, count);
                self.status = Status::Generating {
                    done: 0,
                    total: count,
                    step: 0,
                    steps: 0,
                };
            }
            ImageStarted { index } => self.gallery.start(index),
            Step { step, steps } => {
                if let Status::Generating {
                    step: st,
                    steps: sn,
                    ..
                } = &mut self.status
                {
                    *st = step;
                    *sn = steps;
                }
            }
            Preview { index, image } => {
                self.gallery
                    .set_preview(index, image::DynamicImage::ImageRgb8(image));
            }
            ImageReady { index, entry } => {
                self.gallery.finish(index, entry);
                if let Status::Generating { done, .. } = &mut self.status {
                    *done += 1;
                }
            }
            ImageFailed { index, error } => {
                self.toast = Some(format!("image {index} failed: {error}"))
            }
            BatchDone => self.status = Status::Idle,
            Error(e) => self.status = Status::Error(e),
        }
    }
}

/// Browse a finished directory of images (no generation). Works in any build
/// with the `view` feature.
pub fn run_static(dir: PathBuf, saved_dir: PathBuf) -> Result<()> {
    if !std::io::stdout().is_terminal() {
        anyhow::bail!("pixl view needs an interactive terminal");
    }
    let entries = load_dir(&dir)?;
    if entries.is_empty() {
        anyhow::bail!("no images (.png/.jpg) found in {}", dir.display());
    }
    let mut terminal = ratatui::init();
    let picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
    let mut app = App {
        gallery: Gallery::fixed(entries),
        picker,
        proto: None,
        saved_dir,
        toast: None,
        quit: false,
        #[cfg(feature = "gen")]
        static_mode: true,
        #[cfg(feature = "gen")]
        actor: None,
        #[cfg(feature = "gen")]
        status: Status::Idle,
        #[cfg(feature = "gen")]
        input: None,
        #[cfg(feature = "gen")]
        last_prompt: String::new(),
        #[cfg(feature = "gen")]
        base_negative: String::new(),
        #[cfg(feature = "gen")]
        settings: Settings::placeholder(),
        #[cfg(feature = "gen")]
        model_line: None,
        #[cfg(feature = "gen")]
        panel: false,
        #[cfg(feature = "gen")]
        panel_row: 0,
        #[cfg(feature = "gen")]
        gen_size: (0, 0),
        #[cfg(feature = "gen")]
        tick: 0,
        #[cfg(feature = "gen")]
        skip: Arc::new(Mutex::new(HashSet::new())),
        #[cfg(feature = "gen")]
        out_dir: PathBuf::new(),
    };
    let res = app.run(&mut terminal);
    ratatui::restore();
    res
}

fn load_dir(dir: &PathBuf) -> Result<Vec<gallery::Entry>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .as_deref(),
                Some("png" | "jpg" | "jpeg")
            )
        })
        .collect();
    paths.sort();
    let prompt = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    Ok(paths
        .into_iter()
        .map(|path| gallery::Entry {
            path,
            prompt: prompt.clone(),
            seed: None,
            saved: false,
        })
        .collect())
}

/// Live gallery: stream images from a background generator. Returns `Ok(false)`
/// if it declined (terminal has no graphics protocol and `--view` wasn't forced),
/// so the caller can fall back to the headless batch.
#[cfg(feature = "gen")]
pub fn run_live(
    prompt: &str,
    count: Option<u32>,
    out_dir: PathBuf,
    w: u32,
    h: u32,
    args: &crate::cli::GenerateArgs,
    saved_dir: PathBuf,
) -> Result<bool> {
    use ratatui_image::picker::ProtocolType;

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating output dir {}", out_dir.display()))?;

    let mut terminal = ratatui::init();
    let picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
    let graphics = matches!(
        picker.protocol_type(),
        ProtocolType::Kitty | ProtocolType::Sixel | ProtocolType::Iterm2
    );
    if !graphics && !args.view {
        ratatui::restore();
        return Ok(false);
    }

    let skip = Arc::new(Mutex::new(HashSet::new()));
    let settings = Settings::load(args, count);
    let actor = actor::Actor::spawn(args.clone(), w, h, out_dir.clone(), skip.clone());
    let mut app = App {
        gallery: Gallery::live(),
        picker,
        proto: None,
        saved_dir,
        toast: None,
        quit: false,
        static_mode: false,
        actor: Some(actor),
        status: Status::Loading,
        input: None,
        last_prompt: prompt.to_string(),
        base_negative: args.negative.clone(),
        settings,
        model_line: None,
        panel: false,
        panel_row: 0,
        gen_size: (w, h),
        tick: 0,
        skip,
        out_dir: out_dir.clone(),
    };
    app.submit();
    let res = app.run(&mut terminal);

    if let Some(a) = &app.actor {
        a.cancel();
    }
    drop(app.actor.take()); // close the command channel -> actor exits
    ratatui::restore();
    res?;

    let saved = app
        .gallery
        .slots
        .iter()
        .filter_map(|(_, s)| s.done())
        .filter(|e| e.saved)
        .count();
    println!(
        "gallery: {} generated, {saved} saved -> {}",
        app.gallery.len(),
        out_dir.display()
    );
    Ok(true)
}
