//! Gallery state and pure helpers. No terminal or rendering dependencies, so the
//! navigation and save-path logic unit-test without a TTY or graphics protocol.

use std::path::{Path, PathBuf};

/// One finished image in the gallery.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub path: PathBuf,
    pub prompt: String,
    /// The seed that produced it, when known (`None` for `pixl view`).
    pub seed: Option<u64>,
    /// Whether the user has copied this one into the saved folder.
    pub saved: bool,
}

/// A gallery slot, covering an image's whole lifecycle so the user can see the
/// entire queued batch up front, not just finished images.
pub enum Slot {
    /// Queued but not started — shown as a placeholder.
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    Queued,
    /// Generating, with its latest in-flight preview (if any yet).
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    Generating(Option<image::DynamicImage>),
    /// Finished.
    Done(Entry),
}

impl Slot {
    pub fn done(&self) -> Option<&Entry> {
        match self {
            Slot::Done(e) => Some(e),
            _ => None,
        }
    }
    fn done_mut(&mut self) -> Option<&mut Entry> {
        match self {
            Slot::Done(e) => Some(e),
            _ => None,
        }
    }
}

/// Default keepers folder: `~/.pixl/saved` (consistent with the run dirs).
pub fn default_saved_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".pixl")
        .join("saved")
}

/// Destination path for saving `entry` into `saved_dir` without clobbering a
/// different existing file: keep the source name; on collision disambiguate with
/// the seed (`name_<seed>.png`), then a numeric suffix. `exists` is injected so
/// this stays pure and testable.
pub fn save_dest(saved_dir: &Path, entry: &Entry, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let stem = entry
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("img");
    let ext = entry
        .path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    let direct = saved_dir.join(format!("{stem}.{ext}"));
    if !exists(&direct) {
        return direct;
    }
    if let Some(seed) = entry.seed {
        let with_seed = saved_dir.join(format!("{stem}_{seed}.{ext}"));
        if !exists(&with_seed) {
            return with_seed;
        }
    }
    let mut n = 1u32;
    loop {
        let cand = saved_dir.join(format!("{stem}_{n}.{ext}"));
        if !exists(&cand) {
            return cand;
        }
        n += 1;
    }
}

/// The navigable list of slots plus a cursor.
///
/// `follow_edge` tracks whether the cursor is riding the live edge: while it is,
/// the cursor follows the image currently being generated; once the user steps
/// back to inspect, their position is held.
#[derive(Default)]
pub struct Gallery {
    pub slots: Vec<Slot>,
    pub current: usize,
    follow_edge: bool,
}

impl Gallery {
    /// An empty gallery that follows the live edge (for streaming generation).
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    pub fn live() -> Self {
        Self {
            slots: Vec::new(),
            current: 0,
            follow_edge: true,
        }
    }

    /// A fixed gallery over already-known images (for `pixl view`).
    pub fn fixed(entries: Vec<Entry>) -> Self {
        Self {
            slots: entries.into_iter().map(Slot::Done).collect(),
            current: 0,
            follow_edge: false,
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn current_slot(&self) -> Option<&Slot> {
        self.slots.get(self.current)
    }

    pub fn current_done(&self) -> Option<&Entry> {
        self.slots.get(self.current).and_then(Slot::done)
    }

    pub fn current_done_mut(&mut self) -> Option<&mut Entry> {
        self.slots.get_mut(self.current).and_then(Slot::done_mut)
    }

    fn at_edge(&self) -> bool {
        self.current + 1 >= self.slots.len()
    }

    pub fn next(&mut self) {
        if self.current + 1 < self.slots.len() {
            self.current += 1;
        }
        self.follow_edge = self.at_edge();
    }

    pub fn prev(&mut self) {
        self.current = self.current.saturating_sub(1);
        self.follow_edge = false;
    }

    pub fn first(&mut self) {
        self.current = 0;
        self.follow_edge = self.slots.len() <= 1;
    }

    pub fn last(&mut self) {
        self.current = self.slots.len().saturating_sub(1);
        self.follow_edge = true;
    }

    /// Append `n` queued slots for a starting batch.
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    pub fn push_queued(&mut self, n: u32) {
        for _ in 0..n {
            self.slots.push(Slot::Queued);
        }
    }

    /// Mark a slot as generating; follow it if we're riding the edge.
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    pub fn start(&mut self, index: usize) {
        if let Some(s) = self.slots.get_mut(index) {
            *s = Slot::Generating(None);
        }
        if self.follow_edge && index < self.slots.len() {
            self.current = index;
        }
    }

    /// Update a generating slot's latest preview.
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    pub fn set_preview(&mut self, index: usize, img: image::DynamicImage) {
        if let Some(Slot::Generating(p)) = self.slots.get_mut(index) {
            *p = Some(img);
        }
    }

    /// Mark a slot finished.
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    pub fn finish(&mut self, index: usize, entry: Entry) {
        if let Some(s) = self.slots.get_mut(index) {
            *s = Slot::Done(entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, seed: Option<u64>) -> Entry {
        Entry {
            path: PathBuf::from(format!("/run/{name}.png")),
            prompt: "p".into(),
            seed,
            saved: false,
        }
    }

    #[test]
    fn queued_then_generates_then_done() {
        let mut g = Gallery::live();
        g.push_queued(3);
        assert_eq!(g.len(), 3);
        assert!(matches!(g.current_slot(), Some(Slot::Queued)));
        // start image 0 -> follows it
        g.start(0);
        assert_eq!(g.current, 0);
        assert!(matches!(g.current_slot(), Some(Slot::Generating(None))));
        g.set_preview(0, image::DynamicImage::new_rgb8(1, 1));
        assert!(matches!(g.current_slot(), Some(Slot::Generating(Some(_)))));
        g.finish(0, entry("a", Some(0)));
        assert!(matches!(g.current_slot(), Some(Slot::Done(_))));
        // next image starts; following -> cursor moves to it
        g.start(1);
        assert_eq!(g.current, 1);
    }

    #[test]
    fn navigating_back_holds_position() {
        let mut g = Gallery::live();
        g.push_queued(3);
        g.start(0);
        g.finish(0, entry("a", None));
        g.start(1);
        assert_eq!(g.current, 1);
        g.prev();
        assert_eq!(g.current, 0, "stepped back to inspect");
        g.start(2);
        assert_eq!(g.current, 0, "held position; not following");
    }

    #[test]
    fn nav_clamps_at_both_ends() {
        let mut g = Gallery::fixed(vec![entry("a", None), entry("b", None)]);
        g.prev();
        assert_eq!(g.current, 0);
        g.next();
        g.next();
        assert_eq!(g.current, 1);
    }

    #[test]
    fn save_dest_uniquifies_on_collision() {
        let dir = Path::new("/saved");
        let e = entry("house_000", Some(42));
        assert_eq!(
            save_dest(dir, &e, |_| false),
            PathBuf::from("/saved/house_000.png")
        );
        let taken = PathBuf::from("/saved/house_000.png");
        assert_eq!(
            save_dest(dir, &e, |p| p == taken),
            PathBuf::from("/saved/house_000_42.png")
        );
        let taken2 = [
            PathBuf::from("/saved/house_000.png"),
            PathBuf::from("/saved/house_000_42.png"),
        ];
        assert_eq!(
            save_dest(dir, &e, |p| taken2.contains(&p.to_path_buf())),
            PathBuf::from("/saved/house_000_1.png")
        );
    }
}
