//! Reproducibility: an SGF of the game next to the exact prompt sent each turn.
//!
//! When the model plays something absurd at move 130 you want to see precisely
//! what it saw, not a reconstruction.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::goban::{Color, Move, MoveRecord};

pub struct Recorder {
    dir: PathBuf,
    turns: Vec<String>,
}

impl Recorder {
    pub fn new(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            turns: Vec::new(),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Log one model turn: the fat block it was given and what came back.
    pub fn log_turn(&mut self, move_number: u32, prompt: &str, outcome: &str) -> io::Result<()> {
        let entry = format!(
            "=== move {move_number} ===\n--- fat state block sent ---\n{prompt}\n\
             --- outcome ---\n{outcome}\n\n"
        );
        self.turns.push(entry.clone());
        let path = self.dir.join("prompts.log");
        let existing = fs::read_to_string(&path).unwrap_or_default();
        fs::write(path, existing + &entry)
    }

    pub fn write_sgf(&self, history: &[MoveRecord], black: &str, white: &str) -> io::Result<PathBuf> {
        let path = self.dir.join("game.sgf");
        fs::write(&path, sgf(history, black, white))?;
        Ok(path)
    }
}

/// SGF FF[4] with Japanese-style komi. Coordinates are `a`-`s`, column then
/// row, both counted from the top-left — the opposite origin from the display
/// grid, which is exactly where transcription bugs come from.
pub fn sgf(history: &[MoveRecord], black: &str, white: &str) -> String {
    let mut s = String::from("(;FF[4]GM[1]CA[UTF-8]SZ[19]KM[6.5]RU[Japanese]");
    s.push_str(&format!("PB[{}]PW[{}]", escape(black), escape(white)));
    for r in history {
        let tag = match r.color {
            Color::Black => "B",
            Color::White => "W",
        };
        let coord = match r.mv {
            Move::Pass => String::new(),
            Move::Play(p) => format!(
                "{}{}",
                (b'a' + p.col as u8) as char,
                (b'a' + p.row as u8) as char
            ),
        };
        s.push_str(&format!(";{tag}[{coord}]"));
    }
    s.push(')');
    s
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace(']', "\\]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goban::{Game, Point};

    fn pt(s: &str) -> Point {
        Point::parse(s).unwrap()
    }

    /// SGF counts rows from the top while the display counts from the bottom.
    /// Q16 is row 4 from the top => `pd`, the standard SGF opening.
    #[test]
    fn sgf_coordinates_use_the_top_left_origin() {
        let mut g = Game::new();
        g.play(Move::Play(pt("Q16"))).unwrap();
        g.play(Move::Play(pt("D4"))).unwrap();
        let out = sgf(g.history(), "Human", "claude");
        assert!(out.contains(";B[pd]"), "{out}");
        assert!(out.contains(";W[dp]"), "{out}");
        assert!(out.starts_with("(;FF[4]GM[1]"));
        assert!(out.ends_with(')'));
    }

    #[test]
    fn sgf_encodes_a_pass_as_an_empty_value() {
        let mut g = Game::new();
        g.play(Move::Pass).unwrap();
        let out = sgf(g.history(), "a", "b");
        assert!(out.contains(";B[]"), "{out}");
    }

    #[test]
    fn sgf_corners_map_to_the_expected_letters() {
        let mut g = Game::new();
        g.play(Move::Play(pt("A19"))).unwrap(); // top-left
        g.play(Move::Play(pt("T1"))).unwrap(); // bottom-right
        let out = sgf(g.history(), "a", "b");
        assert!(out.contains(";B[aa]"), "{out}");
        assert!(out.contains(";W[ss]"), "{out}");
    }

    #[test]
    fn player_names_are_escaped() {
        let out = sgf(&[], "a]b", "c\\d");
        assert!(out.contains("PB[a\\]b]"), "{out}");
        assert!(out.contains("PW[c\\\\d]"), "{out}");
    }
}
