//! Gallery state and pure helpers. No terminal or rendering dependencies, so the
//! navigation and save-path logic unit-test without a TTY or graphics protocol.

use std::path::{Path, PathBuf};

/// One image in the gallery.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub path: PathBuf,
    pub prompt: String,
    /// The seed that produced it, when known (`None` for `pixl view`).
    pub seed: Option<u64>,
    /// Whether the user has copied this one into the saved folder.
    pub saved: bool,
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

/// The navigable list of images plus a cursor.
///
/// `follow_edge` tracks whether the cursor is "riding the live edge": while it is,
/// newly pushed images auto-advance the cursor to the newest; once the user steps
/// back to inspect, their position is held and only the count grows.
#[derive(Default)]
pub struct Gallery {
    pub entries: Vec<Entry>,
    pub current: usize,
    follow_edge: bool,
}

impl Gallery {
    /// An empty gallery that follows the live edge (for streaming generation).
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    pub fn live() -> Self {
        Self {
            entries: Vec::new(),
            current: 0,
            follow_edge: true,
        }
    }

    /// A fixed gallery over already-known images (for `pixl view`).
    pub fn fixed(entries: Vec<Entry>) -> Self {
        Self {
            entries,
            current: 0,
            follow_edge: false,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn current(&self) -> Option<&Entry> {
        self.entries.get(self.current)
    }

    pub fn current_mut(&mut self) -> Option<&mut Entry> {
        self.entries.get_mut(self.current)
    }

    fn at_edge(&self) -> bool {
        self.current + 1 >= self.entries.len()
    }

    pub fn next(&mut self) {
        if self.current + 1 < self.entries.len() {
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
        self.follow_edge = self.entries.len() <= 1;
    }

    pub fn last(&mut self) {
        self.current = self.entries.len().saturating_sub(1);
        self.follow_edge = true;
    }

    /// Append a freshly generated image, advancing the cursor only if we're
    /// following the live edge.
    #[cfg_attr(not(feature = "gen"), allow(dead_code))]
    pub fn push(&mut self, entry: Entry) {
        self.entries.push(entry);
        if self.follow_edge {
            self.current = self.entries.len() - 1;
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
    fn follow_edge_autoadvances_then_holds() {
        let mut g = Gallery::live();
        g.push(entry("a", None));
        assert_eq!(g.current, 0);
        g.push(entry("b", None));
        assert_eq!(g.current, 1, "rides the live edge");
        // step back to inspect -> stop following
        g.prev();
        assert_eq!(g.current, 0);
        g.push(entry("c", None));
        assert_eq!(g.current, 0, "held position while more stream in");
        assert_eq!(g.len(), 3);
        // jumping to last re-attaches to the edge
        g.last();
        g.push(entry("d", None));
        assert_eq!(g.current, 3);
    }

    #[test]
    fn nav_clamps_at_both_ends() {
        let mut g = Gallery::fixed(vec![entry("a", None), entry("b", None)]);
        g.prev();
        assert_eq!(g.current, 0);
        g.next();
        g.next();
        assert_eq!(g.current, 1, "clamps at the last image");
    }

    #[test]
    fn save_dest_uniquifies_on_collision() {
        let dir = Path::new("/saved");
        let e = entry("house_000", Some(42));
        // nothing exists yet -> keep the source name
        assert_eq!(
            save_dest(dir, &e, |_| false),
            PathBuf::from("/saved/house_000.png")
        );
        // direct name taken -> fall back to the seed
        let taken = PathBuf::from("/saved/house_000.png");
        assert_eq!(
            save_dest(dir, &e, |p| p == taken),
            PathBuf::from("/saved/house_000_42.png")
        );
        // both taken -> numeric suffix
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
