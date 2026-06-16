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

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{Resize, StatefulImage};

use gallery::{save_dest, Gallery};

#[cfg(feature = "gen")]
#[derive(Clone)]
enum Status {
    Loading,
    Generating {
        done: usize,
        total: u32,
        step: usize,
        steps: usize,
    },
    Idle,
    Error(String),
}

struct App {
    gallery: Gallery,
    picker: Picker,
    /// Decoded + encoded protocol for the currently shown image, with its index.
    /// Only the visible image is kept resident; switching re-decodes (cheap for
    /// small sprites, and cached by the protocol for repeated renders).
    proto: Option<(usize, StatefulProtocol)>,
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
    count: u32,
    #[cfg(feature = "gen")]
    model_line: Option<String>,
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
            }

            if dirty {
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

    /// Decode + prepare the protocol for the current image if it changed.
    fn ensure_proto(&mut self) {
        let want = if self.gallery.is_empty() {
            None
        } else {
            Some(self.gallery.current)
        };
        if want == self.proto.as_ref().map(|(i, _)| *i) {
            return;
        }
        self.proto = None;
        let Some(idx) = want else { return };
        let path = self.gallery.entries[idx].path.clone();
        if let Ok(img) = image::open(&path) {
            self.proto = Some((idx, self.picker.new_resize_protocol(img)));
        }
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
        self.render_image(f, chunks[1]);
        f.render_widget(self.status_line(), chunks[2]);
        self.render_footer(f, chunks[3]);
    }

    fn render_title(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let total = self.gallery.len();
        let idx = if total == 0 {
            0
        } else {
            self.gallery.current + 1
        };
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
        if self.gallery.current().map(|e| e.saved).unwrap_or(false) {
            spans.push(Span::styled("  saved", Style::new().fg(Color::Green)));
        }
        if let Some(p) = self.gallery.current().map(|e| e.prompt.clone()) {
            if !p.is_empty() {
                spans.push(Span::styled(format!("  {p}"), Style::new().fg(Color::Gray)));
            }
        }
        f.render_widget(Line::from(spans), area);
    }

    fn render_image(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        self.ensure_proto();
        if let Some((_, proto)) = &mut self.proto {
            f.render_stateful_widget(
                StatefulImage::default().resize(Resize::Fit(None)),
                area,
                proto,
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

    fn render_footer(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        #[cfg(feature = "gen")]
        if let Some(input) = &self.input {
            let line = Line::from(vec![
                Span::styled("new prompt: ", Style::new().fg(Color::Cyan)),
                Span::raw(input.as_str()),
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
                "←/→ nav · s save · r rerun · e edit · c cancel · q quit"
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

        match key.code {
            KeyCode::Left | KeyCode::Char('h') => self.gallery.prev(),
            KeyCode::Right | KeyCode::Char('l') => self.gallery.next(),
            KeyCode::Home | KeyCode::Char('g') => self.gallery.first(),
            KeyCode::End | KeyCode::Char('G') => self.gallery.last(),
            KeyCode::Char('s') | KeyCode::Char(' ') => self.save_current(),
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            #[cfg(feature = "gen")]
            KeyCode::Char('r') => self.rerun(),
            #[cfg(feature = "gen")]
            KeyCode::Char('e') => self.begin_edit(),
            #[cfg(feature = "gen")]
            KeyCode::Char('c') => self.cancel_gen(),
            _ => {}
        }
    }

    fn save_current(&mut self) {
        let Some(entry) = self.gallery.current().cloned() else {
            self.toast = Some("nothing to save".into());
            return;
        };
        let dest = save_dest(&self.saved_dir, &entry, |p| p.exists());
        let res = std::fs::create_dir_all(&self.saved_dir)
            .and_then(|_| std::fs::copy(&entry.path, &dest).map(|_| ()));
        match res {
            Ok(()) => {
                if let Some(e) = self.gallery.current_mut() {
                    e.saved = true;
                }
                self.toast = Some(format!("saved → {}", dest.display()));
            }
            Err(e) => self.toast = Some(format!("save failed: {e}")),
        }
    }

    #[cfg(feature = "gen")]
    fn rerun(&mut self) {
        if self.static_mode || self.last_prompt.is_empty() {
            return;
        }
        if let Some(a) = &self.actor {
            a.generate(self.last_prompt.clone(), self.count);
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
                        self.last_prompt = p.clone();
                        if let Some(a) = &self.actor {
                            a.generate(p, self.count);
                        }
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
            BatchStarted { total } => {
                self.status = Status::Generating {
                    done: 0,
                    total,
                    step: 0,
                    steps: 0,
                }
            }
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
            ImageReady(entry) => {
                self.gallery.push(entry);
                if let Status::Generating { done, .. } = &mut self.status {
                    *done += 1;
                }
            }
            ImageFailed { idx, error } => self.toast = Some(format!("image {idx} failed: {error}")),
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
        count: 0,
        #[cfg(feature = "gen")]
        model_line: None,
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
    count: u32,
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

    let actor = actor::Actor::spawn(args.clone(), w, h, out_dir.clone());
    actor.generate(prompt.to_string(), count);

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
        count,
        model_line: None,
    };
    let res = app.run(&mut terminal);

    if let Some(a) = &app.actor {
        a.cancel();
    }
    drop(app.actor.take()); // close the command channel -> actor exits
    ratatui::restore();
    res?;

    let saved = app.gallery.entries.iter().filter(|e| e.saved).count();
    println!(
        "gallery: {} generated, {saved} saved -> {}",
        app.gallery.len(),
        out_dir.display()
    );
    Ok(true)
}
