//! Rendering the model's view of the game.
//!
//! Two shapes matter. The **fat block** is the full board plus everything the
//! model reads badly off a grid — liberties, atari, ko — computed here so it
//! never has to derive them. The **thin stub** is what that block collapses to
//! once it has been answered. Exactly one fat block exists at any time.
//!
//! Both are pure functions of [`Game`] state. Nothing is ever edited
//! incrementally, which is what keeps the model's view from drifting away from
//! the authoritative board.

use std::fmt::Write as _;

use crate::goban::{Color, Game, Group, Move, Point};

/// Two-tier scratchpad. Strategy is durable; tactics expire on their own terms
/// and are meant to be overwritten every few moves.
#[derive(Debug, Clone, Default)]
pub struct Notes {
    pub strategy: String,
    pub tactics: String,
}

impl Notes {
    /// Roughly 500 tokens total, enforced in characters so it cannot creep.
    const STRATEGY_CAP: usize = 1200;
    const TACTICS_CAP: usize = 800;

    pub fn set_strategy(&mut self, s: &str) {
        self.strategy = truncate(s, Self::STRATEGY_CAP);
    }

    pub fn set_tactics(&mut self, s: &str) {
        self.tactics = truncate(s, Self::TACTICS_CAP);
    }

    fn render(&self) -> String {
        let strategy = if self.strategy.is_empty() {
            "(none yet)"
        } else {
            &self.strategy
        };
        let tactics = if self.tactics.is_empty() {
            "(none yet)"
        } else {
            &self.tactics
        };
        format!("  strategy: {strategy}\n  tactics:  {tactics}")
    }
}

fn truncate(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    let mut end = cap;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

pub const SYSTEM_PROMPT: &str = "\
You are playing Go on a 19x19 board against a human opponent.

The board is shown with columns A-T (the letter I is skipped, so there are 19 \
columns) and rows 1-19 numbered from the bottom. `X` is a Black stone, `O` is a \
White stone, `+` is an empty star point, `.` is an empty intersection.

Trust the analysis section over your own reading of the grid. Liberty counts, \
atari, and the ko point are computed for you and are authoritative. Do not \
recount them off the ASCII board.

The referee owns the rules. If you propose an illegal move you will get a tool \
error explaining why; read it and choose a different point. Do not argue with \
the referee.

Play the whole board. Open in the corners, extend along the sides, and only \
fight when you have a reason to. Keep your groups connected and out of atari.";

/// Everything the model needs to choose a move. ~700-900 tokens.
pub fn fat_state_block(game: &Game, me: Color, notes: &Notes) -> String {
    let groups = game.groups();
    let mut s = String::with_capacity(2048);

    let _ = write!(s, "{}\n\n", game.render_board());
    let _ = writeln!(s, "--- position analysis (authoritative) ---");
    let _ = writeln!(s, "Move number: {}", game.move_number());
    let _ = writeln!(s, "You are {me} ({}).", stone_char(me));
    let _ = writeln!(
        s,
        "Groups in atari (1 liberty): {}",
        list_groups(&groups, |g| g.liberties.len() == 1, me)
    );
    let _ = writeln!(
        s,
        "Groups at 2 liberties:       {}",
        list_groups(&groups, |g| g.liberties.len() == 2, me)
    );
    let _ = writeln!(
        s,
        "Ko point (illegal this turn): {}",
        game.ko_point()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "none".into())
    );
    let _ = writeln!(
        s,
        "Captures so far: Black {}, White {}",
        game.captures(Color::Black),
        game.captures(Color::White)
    );
    let _ = writeln!(s, "Recent moves: {}", recent_moves(game, 6));
    let _ = writeln!(s, "\nYour scratchpad:\n{}", notes.render());
    let _ = write!(
        s,
        "\nYou are {me} and it is your turn. Call play_move with your chosen point."
    );
    s
}

/// What a fat block collapses to once answered. ~20 tokens.
pub fn thin_stub(number: u32, own: Move, reply: Option<Move>) -> String {
    match reply {
        Some(r) => format!("Move {number} — you played {own}. Opponent replied {r}."),
        None => format!("Move {number} — you played {own}."),
    }
}

fn stone_char(c: Color) -> &'static str {
    match c {
        Color::Black => "X",
        Color::White => "O",
    }
}

/// Groups matching `pred`, tagged yours/theirs — the distinction the model
/// most often gets backwards when left to read it off the grid.
fn list_groups(groups: &[Group], pred: impl Fn(&Group) -> bool, me: Color) -> String {
    let mut items: Vec<String> = groups
        .iter()
        .filter(|g| pred(g))
        .map(|g| {
            let who = if g.color == me { "yours" } else { "theirs" };
            let libs = g
                .liberties
                .iter()
                .map(Point::to_string)
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{} ({who}, {} stone{}, liberties: {libs})",
                anchor(g),
                g.stones.len(),
                if g.stones.len() == 1 { "" } else { "s" }
            )
        })
        .collect();
    if items.is_empty() {
        return "none".into();
    }
    items.sort();
    items.join("; ")
}

/// Name a group by a stable representative stone.
fn anchor(g: &Group) -> String {
    g.stones
        .iter()
        .min()
        .map(Point::to_string)
        .unwrap_or_default()
}

fn recent_moves(game: &Game, n: usize) -> String {
    let hist = game.history();
    if hist.is_empty() {
        return "none (empty board)".into();
    }
    let start = hist.len().saturating_sub(n);
    hist[start..]
        .iter()
        .map(|r| {
            let cap = if r.captured > 0 {
                format!(" (captured {})", r.captured)
            } else {
                String::new()
            };
            format!("{}. {} {}{}", r.number, r.color, r.mv, cap)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goban::Move;

    fn pt(s: &str) -> Point {
        Point::parse(s).unwrap()
    }

    #[test]
    fn fat_block_reports_atari_with_ownership_and_liberties() {
        let mut g = Game::new();
        // Put the black D4 stone in atari.
        for mv in ["D4", "D5", "T1", "D3", "T2", "C4"] {
            g.play(Move::Play(pt(mv))).unwrap();
        }
        let text = fat_state_block(&g, Color::Black, &Notes::default());
        let atari_line = text
            .lines()
            .find(|l| l.starts_with("Groups in atari"))
            .unwrap();
        assert!(atari_line.contains("D4"), "got: {atari_line}");
        assert!(atari_line.contains("yours"), "got: {atari_line}");
        assert!(atari_line.contains("E4"), "liberty listed: {atari_line}");
    }

    #[test]
    fn fat_block_reports_no_atari_on_a_quiet_board() {
        let mut g = Game::new();
        g.play(Move::Play(pt("Q16"))).unwrap();
        let text = fat_state_block(&g, Color::White, &Notes::default());
        assert!(text.contains("Groups in atari (1 liberty): none"));
        assert!(text.contains("Ko point (illegal this turn): none"));
        assert!(text.contains("You are White"));
    }

    #[test]
    fn fat_block_surfaces_the_ko_point() {
        let mut g = Game::new();
        for mv in ["D3", "E3", "C4", "F4", "D5", "E5", "E4", "T1", "T2", "D4"] {
            g.play(Move::Play(pt(mv))).unwrap();
        }
        let text = fat_state_block(&g, Color::Black, &Notes::default());
        assert!(text.contains("Ko point (illegal this turn): E4"), "{text}");
    }

    /// The whole cost argument rests on the fat block being big and the stub
    /// being tiny. Guard the ratio so a careless edit cannot erase it.
    #[test]
    fn thin_stub_is_far_smaller_than_the_fat_block() {
        let g = Game::new();
        let fat = fat_state_block(&g, Color::Black, &Notes::default());
        let thin = thin_stub(47, Move::Play(pt("Q3")), Some(Move::Play(pt("R4"))));
        assert!(fat.len() > 900, "fat block was {} chars", fat.len());
        assert!(thin.len() < 80, "thin stub was {} chars", thin.len());
        assert_eq!(thin, "Move 47 — you played Q3. Opponent replied R4.");
    }

    #[test]
    fn notes_are_capped() {
        let mut n = Notes::default();
        n.set_strategy(&"a".repeat(5000));
        n.set_tactics(&"b".repeat(5000));
        assert!(n.strategy.len() <= Notes::STRATEGY_CAP + 4);
        assert!(n.tactics.len() <= Notes::TACTICS_CAP + 4);
        assert!(n.strategy.ends_with('…'));
    }

    #[test]
    fn recent_moves_lists_captures_and_truncates_to_six() {
        let mut g = Game::new();
        for mv in ["D5", "D4", "C4", "T1", "E4", "T2", "D3"] {
            g.play(Move::Play(pt(mv))).unwrap();
        }
        let text = recent_moves(&g, 6);
        assert!(text.contains("(captured 1)"), "{text}");
        assert_eq!(text.matches('.').count(), 6, "six entries: {text}");
        assert!(!text.contains("1. Black D5"), "oldest dropped: {text}");
    }
}
