//! Background generation actor.
//!
//! Owns the loaded `CandleSdxlGenerator` on its own thread and is driven by a
//! command channel, emitting per-image events. Keeping the generator resident
//! means "rerun" and "edit prompt" never reload the model.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
        cfg: f32,
        steps: u32,
        seed: u64,
        colors: u16,
    },
}

/// Events streamed back to the gallery.
pub enum GenEvent {
    Loading,
    Download {
        file: String,
        done: u64,
        total: u64,
    },
    Loaded {
        model: String,
        cached: bool,
        lora: Option<(String, f32)>,
        merged: bool,
    },
    BatchStarted {
        start: usize,
        count: u32,
    },
    ImageStarted {
        index: usize,
    },
    Step {
        step: usize,
        steps: usize,
    },
    Preview {
        index: usize,
        image: image::RgbImage,
    },
    ImageReady {
        index: usize,
        entry: Entry,
    },
    ImageFailed {
        index: usize,
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
    interrupt: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

impl Actor {
    /// Spawn the actor; it loads the generator, emits `Loading`/`Loaded`, then
    /// serves `Generate` commands until the command channel closes.
    pub fn spawn(
        args: GenerateArgs,
        w: u32,
        h: u32,
        out_dir: std::path::PathBuf,
        skip: Arc<Mutex<HashSet<usize>>>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<GenCommand>();
        let (evt_tx, evt_rx) = crossbeam_channel::unbounded::<GenEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let interrupt = Arc::new(AtomicBool::new(false));
        let handle = {
            let cancel = cancel.clone();
            let interrupt = interrupt.clone();
            std::thread::spawn(move || {
                run(args, w, h, out_dir, cmd_rx, evt_tx, cancel, interrupt, skip)
            })
        };
        Self {
            cmd: cmd_tx,
            events: evt_rx,
            cancel,
            interrupt,
            _handle: handle,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: String,
        negative: String,
        count: u32,
        cfg: f32,
        steps: u32,
        seed: u64,
        colors: u16,
    ) {
        let _ = self.cmd.send(GenCommand::Generate {
            prompt,
            negative,
            count,
            cfg,
            steps,
            seed,
            colors,
        });
    }

    /// Stop the batch and abort the in-flight image.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.interrupt.store(true, Ordering::Relaxed);
    }

    /// Abort just the in-flight image (the batch continues) — used when its slot
    /// is discarded mid-generation.
    pub fn interrupt(&self) {
        self.interrupt.store(true, Ordering::Relaxed);
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    args: GenerateArgs,
    w: u32,
    h: u32,
    out_dir: std::path::PathBuf,
    cmd_rx: Receiver<GenCommand>,
    evt_tx: Sender<GenEvent>,
    cancel: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    skip: Arc<Mutex<HashSet<usize>>>,
) {
    let mut args = args;
    let _ = evt_tx.send(GenEvent::Loading);
    let (model, loras) = crate::model_and_loras(&args);
    let prog: pixl_gen::ProgressFn = {
        let evt = evt_tx.clone();
        Box::new(move |p: pixl_gen::DownloadProgress| {
            let _ = evt.send(GenEvent::Download {
                file: p.file,
                done: p.done,
                total: p.total,
            });
        })
    };
    let (mut generator, report) = match CandleSdxlGenerator::load(model, w, h, &loras, Some(prog)) {
        Ok(g) => g,
        Err(e) => {
            let _ = evt_tx.send(GenEvent::Error(format!("loading generator: {e}")));
            return;
        }
    };

    // Index of the image currently rendering, so step/preview callbacks (which
    // don't know it) can tag their events with the right slot.
    let cur = Arc::new(AtomicUsize::new(0));
    {
        let evt = evt_tx.clone();
        generator.set_step_callback(Box::new(move |step, steps| {
            let _ = evt.send(GenEvent::Step { step, steps });
        }));
    }
    {
        let evt = evt_tx.clone();
        let cur = cur.clone();
        generator.set_preview_callback(Box::new(move |image| {
            let _ = evt.send(GenEvent::Preview {
                index: cur.load(Ordering::Relaxed),
                image,
            });
        }));
    }
    generator.set_interrupt(interrupt.clone());

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
        cfg,
        steps,
        seed,
        colors,
    }) = cmd_rx.recv()
    {
        args.cfg = Some(cfg);
        args.steps = Some(steps);
        args.seed = Some(seed);
        args.colors = Some(colors);
        cancel.store(false, Ordering::Relaxed);
        let _ = evt_tx.send(GenEvent::BatchStarted {
            start: next_index,
            count,
        });
        let req = GenRequest {
            prompt: prompt.clone(),
            negative: negative.clone(),
            params: crate::gen_params(&args),
        };
        let slug = crate::slugify(&prompt);
        let is_skipped = |id: usize| skip.lock().map(|s| s.contains(&id)).unwrap_or(false);
        for _ in 0..count {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let id = next_index;
            next_index += 1;
            if is_skipped(id) {
                continue; // discarded while queued: don't spend the GPU
            }
            interrupt.store(false, Ordering::Relaxed);
            cur.store(id, Ordering::Relaxed);
            let _ = evt_tx.send(GenEvent::ImageStarted { index: id });
            match generator.generate(&req, id) {
                Ok(gi) => {
                    if is_skipped(id) {
                        continue; // discarded mid-flight: drop the result, don't save
                    }
                    match crate::pixelize_and_save(gi, id, &out_dir, &slug, &args) {
                        Ok(saved) => {
                            let _ = evt_tx.send(GenEvent::ImageReady {
                                index: id,
                                entry: Entry {
                                    path: saved.path,
                                    prompt: prompt.clone(),
                                    seed: Some(saved.seed),
                                    saved: false,
                                },
                            });
                        }
                        Err(e) => {
                            let _ = evt_tx.send(GenEvent::ImageFailed {
                                index: id,
                                error: e.to_string(),
                            });
                        }
                    }
                }
                Err(pixl_gen::GenError::Cancelled) => {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    continue; // discarded mid-flight: on to the next
                }
                Err(e) => {
                    let _ = evt_tx.send(GenEvent::ImageFailed {
                        index: id,
                        error: e.to_string(),
                    });
                }
            }
        }
        let _ = evt_tx.send(GenEvent::BatchDone);
    }
}
