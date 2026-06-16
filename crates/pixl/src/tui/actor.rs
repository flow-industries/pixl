//! Background generation actor.
//!
//! Owns the loaded `CandleSdxlGenerator` on its own thread and is driven by a
//! command channel, emitting per-image events. Keeping the generator resident
//! means "rerun" and "edit prompt" never reload the model.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use pixl_gen::{CandleSdxlGenerator, GenRequest, Generator};

use crate::cli::GenerateArgs;
use crate::tui::gallery::Entry;

/// A request to the actor. Dropping the [`Sender`] (channel close) is the
/// shutdown signal; cancellation of an in-flight batch goes through the shared
/// `cancel` flag instead, since the actor is busy generating and not polling
/// this channel during a batch.
pub enum GenCommand {
    Generate {
        prompt: String,
        negative: String,
        count: u32,
    },
}

/// Events streamed back to the gallery.
pub enum GenEvent {
    Loading,
    Loaded {
        model: String,
        cached: bool,
        lora: Option<(String, f32)>,
        merged: bool,
    },
    BatchStarted {
        total: u32,
    },
    Step {
        step: usize,
        steps: usize,
    },
    ImageReady(Entry),
    ImageFailed {
        idx: usize,
        error: String,
    },
    BatchDone,
    Error(String),
}

/// Handle to the generation thread held by the gallery.
pub struct Actor {
    cmd: Sender<GenCommand>,
    pub events: Receiver<GenEvent>,
    cancel: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

impl Actor {
    /// Spawn the actor; it loads the generator, emits `Loading`/`Loaded`, then
    /// serves `Generate` commands until the command channel closes.
    pub fn spawn(args: GenerateArgs, w: u32, h: u32, out_dir: std::path::PathBuf) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<GenCommand>();
        let (evt_tx, evt_rx) = crossbeam_channel::unbounded::<GenEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let handle = {
            let cancel = cancel.clone();
            std::thread::spawn(move || run(args, w, h, out_dir, cmd_rx, evt_tx, cancel))
        };
        Self {
            cmd: cmd_tx,
            events: evt_rx,
            cancel,
            _handle: handle,
        }
    }

    pub fn generate(&self, prompt: String, negative: String, count: u32) {
        let _ = self.cmd.send(GenCommand::Generate {
            prompt,
            negative,
            count,
        });
    }

    /// Ask the current batch to stop after the in-flight image.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

fn run(
    args: GenerateArgs,
    w: u32,
    h: u32,
    out_dir: std::path::PathBuf,
    cmd_rx: Receiver<GenCommand>,
    evt_tx: Sender<GenEvent>,
    cancel: Arc<AtomicBool>,
) {
    let _ = evt_tx.send(GenEvent::Loading);
    let (model, loras) = crate::model_and_loras(&args);
    let (mut generator, report) = match CandleSdxlGenerator::load(model, w, h, &loras) {
        Ok(g) => g,
        Err(e) => {
            let _ = evt_tx.send(GenEvent::Error(format!("loading generator: {e}")));
            return;
        }
    };

    // Per-step progress for the image currently rendering.
    {
        let evt = evt_tx.clone();
        generator.set_step_callback(Box::new(move |step, steps| {
            let _ = evt.send(GenEvent::Step { step, steps });
        }));
    }

    let _ = evt_tx.send(GenEvent::Loaded {
        model: report.model.to_string(),
        cached: report.weights_cached,
        lora: report.lora.clone(),
        merged: !matches!(report.merge, pixl_gen::MergeState::None),
    });

    // Global image index across batches/reruns so every image gets a fresh seed
    // and a unique filename.
    let mut next_index = 0usize;
    while let Ok(GenCommand::Generate {
        prompt,
        negative,
        count,
    }) = cmd_rx.recv()
    {
        cancel.store(false, Ordering::Relaxed);
        let _ = evt_tx.send(GenEvent::BatchStarted { total: count });
        let req = GenRequest {
            prompt: prompt.clone(),
            negative: negative.clone(),
            params: crate::gen_params(&args),
        };
        let slug = crate::slugify(&prompt);
        for _ in 0..count {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            match generator.generate(&req, next_index) {
                Ok(gi) => match crate::pixelize_and_save(gi, next_index, &out_dir, &slug, &args) {
                    Ok(saved) => {
                        let _ = evt_tx.send(GenEvent::ImageReady(Entry {
                            path: saved.path,
                            prompt: prompt.clone(),
                            seed: Some(saved.seed),
                            saved: false,
                        }));
                    }
                    Err(e) => {
                        let _ = evt_tx.send(GenEvent::ImageFailed {
                            idx: next_index,
                            error: e.to_string(),
                        });
                    }
                },
                Err(e) => {
                    let _ = evt_tx.send(GenEvent::ImageFailed {
                        idx: next_index,
                        error: e.to_string(),
                    });
                }
            }
            next_index += 1;
        }
        let _ = evt_tx.send(GenEvent::BatchDone);
    }
}
