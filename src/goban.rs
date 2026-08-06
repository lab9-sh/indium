//! Authoritative 19x19 Go rules.
//!
//! The environment — not the model — owns legality. Everything here is pure and
//! synchronous so the agent loop can treat a move as validated or rejected with
//! a specific reason.

use std::collections::HashSet;
use std::fmt;

pub const SIZE: usize = 19;

/// Columns skip `I` by convention, so 19 letters span A–T.
pub const COLS: &[u8] = b"ABCDEFGHJKLMNOPQRST";

/// Handicap / star points, in (row-from-top, col) form.
const STAR_LINES: [usize; 3] = [3, 9, 15];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Black,
    White,
}

impl Color {
    pub fn opposite(self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    fn index(self) -> usize {
        match self {
            Color::Black => 0,
            Color::White => 1,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Color::Black => "Black",
            Color::White => "White",
        })
    }
}

/// A board intersection. `row` is measured from the top, so row 0 renders as
/// board row 19 and `Point { row: 18, col: 0 }` is A1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Point {
    pub row: usize,
    pub col: usize,
}

impl Point {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    /// Parse standard Go coordinates ("Q16", "d4"). Rejects `I` and anything
    /// off the board — the same surface the model's tool schema constrains.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_uppercase();
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            return Err("empty coordinate".into());
        }
        let col = COLS
            .iter()
            .position(|&c| c == bytes[0])
            .ok_or_else(|| format!("bad column '{}' (use A-T, no I)", bytes[0] as char))?;
        let num: usize = s[1..]
            .parse()
            .map_err(|_| format!("bad row in '{s}' (use 1-19)"))?;
        if num < 1 || num > SIZE {
            return Err(format!("row {num} is off the board (use 1-19)"));
        }
        Ok(Point::new(SIZE - num, col))
    }

    fn neighbors(self) -> impl Iterator<Item = Point> {
        let (r, c) = (self.row, self.col);
        [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
            .into_iter()
            .filter_map(move |(dr, dc)| {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                (nr >= 0 && nr < SIZE as i32 && nc >= 0 && nc < SIZE as i32)
                    .then(|| Point::new(nr as usize, nc as usize))
            })
    }

    fn is_star(self) -> bool {
        STAR_LINES.contains(&self.row) && STAR_LINES.contains(&self.col)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", COLS[self.col] as char, SIZE - self.row)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Play(Point),
    Pass,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Move::Play(p) => write!(f, "{p}"),
            Move::Pass => f.write_str("pass"),
        }
    }
}

/// Why a placement was refused. Each variant renders into the tool error the
/// model sees, so the wording is part of the interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Illegal {
    Occupied(Color),
    Suicide,
    Ko,
    GameOver,
}

impl std::error::Error for Illegal {}

impl fmt::Display for Illegal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Illegal::Occupied(c) => write!(f, "occupied by {c}"),
            Illegal::Suicide => f.write_str("suicide — that group would have no liberties"),
            Illegal::Ko => f.write_str("forbidden by ko — you may not immediately recapture"),
            Illegal::GameOver => f.write_str("the game is already over"),
        }
    }
}

/// One group of connected same-color stones.
#[derive(Debug, Clone)]
pub struct Group {
    pub color: Color,
    pub stones: Vec<Point>,
    pub liberties: Vec<Point>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveRecord {
    pub number: u32,
    pub color: Color,
    pub mv: Move,
    pub captured: usize,
}

#[derive(Debug, Clone)]
pub struct Game {
    cells: Vec<Option<Color>>,
    to_move: Color,
    /// Simple ko: the single point that may not be retaken next move.
    ko: Option<Point>,
    captures: [u32; 2],
    history: Vec<MoveRecord>,
    consecutive_passes: u32,
    resigned: Option<Color>,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        Self {
            cells: vec![None; SIZE * SIZE],
            to_move: Color::Black,
            ko: None,
            captures: [0, 0],
            history: Vec::new(),
            consecutive_passes: 0,
            resigned: None,
        }
    }

    pub fn at(&self, p: Point) -> Option<Color> {
        self.cells[p.row * SIZE + p.col]
    }

    fn set(&mut self, p: Point, c: Option<Color>) {
        self.cells[p.row * SIZE + p.col] = c;
    }

    pub fn to_move(&self) -> Color {
        self.to_move
    }

    pub fn ko_point(&self) -> Option<Point> {
        self.ko
    }

    pub fn captures(&self, c: Color) -> u32 {
        self.captures[c.index()]
    }

    pub fn history(&self) -> &[MoveRecord] {
        &self.history
    }

    /// Number of the move about to be played (1-based).
    pub fn move_number(&self) -> u32 {
        self.history.len() as u32 + 1
    }

    pub fn is_over(&self) -> bool {
        self.resigned.is_some() || self.consecutive_passes >= 2
    }

    pub fn resigned_by(&self) -> Option<Color> {
        self.resigned
    }

    pub fn resign(&mut self, c: Color) {
        self.resigned = Some(c);
    }

    /// Validate and apply a move by the player to move.
    pub fn play(&mut self, mv: Move) -> Result<MoveRecord, Illegal> {
        if self.is_over() {
            return Err(Illegal::GameOver);
        }
        let color = self.to_move;
        let record = match mv {
            Move::Pass => {
                self.consecutive_passes += 1;
                self.ko = None;
                MoveRecord {
                    number: self.move_number(),
                    color,
                    mv: Move::Pass,
                    captured: 0,
                }
            }
            Move::Play(p) => {
                if let Some(c) = self.at(p) {
                    return Err(Illegal::Occupied(c));
                }
                if self.ko == Some(p) {
                    return Err(Illegal::Ko);
                }

                // Tentatively place, then resolve captures before checking
                // suicide — a move that kills is legal even with no liberties
                // of its own beforehand.
                self.set(p, Some(color));
                let mut captured: Vec<Point> = Vec::new();
                for n in p.neighbors() {
                    if self.at(n) == Some(color.opposite()) {
                        let g = self.group_at(n).expect("neighbor has a stone");
                        if g.liberties.is_empty() {
                            captured.extend(g.stones);
                        }
                    }
                }
                captured.sort();
                captured.dedup();

                if captured.is_empty() {
                    let own = self.group_at(p).expect("just placed");
                    if own.liberties.is_empty() {
                        self.set(p, None);
                        return Err(Illegal::Suicide);
                    }
                }

                for &c in &captured {
                    self.set(c, None);
                }
                self.captures[color.index()] += captured.len() as u32;

                // Simple ko: exactly one stone taken by a lone stone that is
                // itself left on one liberty.
                let own = self.group_at(p).expect("just placed");
                self.ko = (captured.len() == 1 && own.stones.len() == 1 && own.liberties.len() == 1)
                    .then(|| captured[0]);

                self.consecutive_passes = 0;
                MoveRecord {
                    number: self.move_number(),
                    color,
                    mv: Move::Play(p),
                    captured: captured.len(),
                }
            }
        };

        self.history.push(record);
        self.to_move = color.opposite();
        Ok(record)
    }

    /// Check legality without mutating, keeping the specific reason.
    ///
    /// The agent loop validates a proposed move before the caller applies it,
    /// so this must not have side effects.
    pub fn validate(&self, mv: Move) -> Result<(), Illegal> {
        let mut probe = self.clone();
        probe.play(mv).map(|_| ())
    }

    /// Would this move be legal? Used to pick opponent moves without mutating.
    pub fn is_legal(&self, mv: Move) -> bool {
        self.validate(mv).is_ok()
    }

    /// The group containing `p`, or `None` if the point is empty.
    pub fn group_at(&self, p: Point) -> Option<Group> {
        let color = self.at(p)?;
        let mut stones = Vec::new();
        let mut liberties = HashSet::new();
        let mut seen = HashSet::new();
        let mut stack = vec![p];
        seen.insert(p);

        while let Some(cur) = stack.pop() {
            stones.push(cur);
            for n in cur.neighbors() {
                match self.at(n) {
                    None => {
                        liberties.insert(n);
                    }
                    Some(c) if c == color && seen.insert(n) => stack.push(n),
                    _ => {}
                }
            }
        }

        stones.sort();
        let mut liberties: Vec<Point> = liberties.into_iter().collect();
        liberties.sort();
        Some(Group {
            color,
            stones,
            liberties,
        })
    }

    /// Every group on the board, each reported once.
    pub fn groups(&self) -> Vec<Group> {
        let mut seen: HashSet<Point> = HashSet::new();
        let mut out = Vec::new();
        for row in 0..SIZE {
            for col in 0..SIZE {
                let p = Point::new(row, col);
                if self.at(p).is_none() || seen.contains(&p) {
                    continue;
                }
                let g = self.group_at(p).expect("occupied");
                seen.extend(g.stones.iter().copied());
                out.push(g);
            }
        }
        out
    }

    /// All empty points where the player to move could legally play. Used by
    /// the scripted opponent; O(n) clones, fine at POC scale.
    pub fn legal_moves(&self) -> Vec<Point> {
        let mut out = Vec::new();
        for row in 0..SIZE {
            for col in 0..SIZE {
                let p = Point::new(row, col);
                if self.at(p).is_none() && self.is_legal(Move::Play(p)) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Tromp-Taylor area score (stones + surrounded empty area), before komi.
    ///
    /// No dead-stone detection: this is only meaningful once the position is
    /// played out, which is why it is reported as provisional.
    pub fn area_score(&self) -> (u32, u32) {
        let mut black = 0;
        let mut white = 0;
        let mut seen: HashSet<Point> = HashSet::new();

        for row in 0..SIZE {
            for col in 0..SIZE {
                let p = Point::new(row, col);
                match self.at(p) {
                    Some(Color::Black) => black += 1,
                    Some(Color::White) => white += 1,
                    None => {
                        if seen.contains(&p) {
                            continue;
                        }
                        // Flood the empty region and see which colors border it.
                        let mut region = Vec::new();
                        let mut borders = HashSet::new();
                        let mut stack = vec![p];
                        seen.insert(p);
                        while let Some(cur) = stack.pop() {
                            region.push(cur);
                            for n in cur.neighbors() {
                                match self.at(n) {
                                    Some(c) => {
                                        borders.insert(c);
                                    }
                                    None => {
                                        if seen.insert(n) {
                                            stack.push(n);
                                        }
                                    }
                                }
                            }
                        }
                        if borders.len() == 1 {
                            let owner = borders.into_iter().next().expect("one border");
                            match owner {
                                Color::Black => black += region.len() as u32,
                                Color::White => white += region.len() as u32,
                            }
                        }
                    }
                }
            }
        }
        (black, white)
    }

    /// Render with coordinates on all four sides. `X` black, `O` white, `+`
    /// star point, `.` empty.
    pub fn render_board(&self) -> String {
        let header: String = COLS
            .iter()
            .map(|&c| format!("{} ", c as char))
            .collect::<String>()
            .trim_end()
            .to_string();

        let mut out = String::with_capacity(1024);
        out.push_str(&format!("   {header}\n"));
        for row in 0..SIZE {
            let num = SIZE - row;
            out.push_str(&format!("{num:>2} "));
            let cells: Vec<String> = (0..SIZE)
                .map(|col| {
                    let p = Point::new(row, col);
                    match self.at(p) {
                        Some(Color::Black) => "X".to_string(),
                        Some(Color::White) => "O".to_string(),
                        None if p.is_star() => "+".to_string(),
                        None => ".".to_string(),
                    }
                })
                .collect();
            out.push_str(&cells.join(" "));
            out.push_str(&format!(" {num:>2}\n"));
        }
        out.push_str(&format!("   {header}"));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(s: &str) -> Point {
        Point::parse(s).expect("valid coordinate")
    }

    /// The I-skip is the classic off-by-one: J must be the 9th column, and the
    /// letter I must not parse at all.
    #[test]
    fn coordinates_skip_i_and_round_trip() {
        assert_eq!(COLS.len(), SIZE);
        assert_eq!(pt("A1"), Point::new(18, 0));
        assert_eq!(pt("T19"), Point::new(0, 18));
        assert_eq!(pt("J10").col, 8);
        assert_eq!(pt("H10").col, 7);
        assert!(Point::parse("I10").is_err());
        for s in ["A1", "T19", "Q16", "D4", "K10", "J1"] {
            assert_eq!(pt(s).to_string(), s);
        }
    }

    #[test]
    fn out_of_range_rows_are_rejected() {
        assert!(Point::parse("A0").is_err());
        assert!(Point::parse("A20").is_err());
        assert!(Point::parse("").is_err());
        assert!(Point::parse("QQ").is_err());
    }

    /// Rendered rows must line up with the coordinate that produced them, or
    /// the model reads stones off the wrong row.
    #[test]
    fn render_places_stone_on_the_labeled_row() {
        let mut g = Game::new();
        g.play(Move::Play(pt("Q16"))).unwrap();
        let text = g.render_board();
        let row16 = text
            .lines()
            .find(|l| l.starts_with("16 "))
            .expect("row 16 present");
        // Q is the 16th column (0-based 15) once I is skipped.
        let cells: Vec<&str> = row16
            .trim_start_matches("16 ")
            .trim()
            .split_whitespace()
            .collect();
        assert_eq!(cells.len(), SIZE + 1); // 19 cells + trailing row label
        assert_eq!(cells[15], "X");
        assert_eq!(cells[SIZE], "16");
    }

    #[test]
    fn capture_removes_group_and_counts_stones() {
        let mut g = Game::new();
        // Black surrounds a lone white stone at D4.
        g.play(Move::Play(pt("D5"))).unwrap(); // B
        g.play(Move::Play(pt("D4"))).unwrap(); // W
        g.play(Move::Play(pt("C4"))).unwrap(); // B
        g.play(Move::Play(pt("T1"))).unwrap(); // W elsewhere
        g.play(Move::Play(pt("E4"))).unwrap(); // B
        g.play(Move::Play(pt("T2"))).unwrap(); // W elsewhere
        let rec = g.play(Move::Play(pt("D3"))).unwrap(); // B captures
        assert_eq!(rec.captured, 1);
        assert_eq!(g.at(pt("D4")), None);
        assert_eq!(g.captures(Color::Black), 1);
    }

    /// The order matters: captures resolve before the suicide check, so a
    /// stone placed with zero liberties of its own is legal when it kills.
    #[test]
    fn capturing_move_is_legal_even_with_no_liberties_of_its_own() {
        let mut g = Game::new();
        // B A3, W A2, B B3, W B1, B C2, W B2, B C1
        for mv in ["A3", "A2", "B3", "B1", "C2", "B2", "C1"] {
            g.play(Move::Play(pt(mv))).unwrap();
        }
        // White A2/B1/B2 is now in atari with its last liberty at A1.
        let white = g.group_at(pt("A2")).unwrap();
        assert_eq!(white.stones.len(), 3);
        assert_eq!(white.liberties, vec![pt("A1")]);

        g.play(Move::Play(pt("T19"))).unwrap(); // W tenuki
        let rec = g.play(Move::Play(pt("A1"))).unwrap(); // B fills its own last liberty
        assert_eq!(rec.captured, 3);
        assert_eq!(g.at(pt("A1")), Some(Color::Black));
        assert_eq!(g.captures(Color::Black), 3);
    }

    #[test]
    fn suicide_error_is_reported() {
        let mut g = Game::new();
        g.play(Move::Play(pt("A2"))).unwrap(); // B
        g.play(Move::Play(pt("T19"))).unwrap(); // W
        g.play(Move::Play(pt("B1"))).unwrap(); // B
        // White to move; A1 is a true suicide.
        assert_eq!(g.play(Move::Play(pt("A1"))), Err(Illegal::Suicide));
    }

    #[test]
    fn occupied_point_is_rejected_with_the_occupying_color() {
        let mut g = Game::new();
        g.play(Move::Play(pt("Q16"))).unwrap();
        assert_eq!(
            g.play(Move::Play(pt("Q16"))),
            Err(Illegal::Occupied(Color::Black))
        );
    }

    /// Simple ko: after a one-stone recapture the taken point is closed for
    /// exactly one move.
    #[test]
    fn ko_forbids_immediate_recapture_then_reopens() {
        let mut g = Game::new();
        // Standard ko shape around D4/E4.
        for (mv, _) in [
            ("D3", ()), // B
            ("E3", ()), // W
            ("C4", ()), // B
            ("F4", ()), // W
            ("D5", ()), // B
            ("E5", ()), // W
            ("E4", ()), // B  <- black stone that white will capture
            ("T1", ()), // W elsewhere
            ("T2", ()), // B elsewhere
            ("D4", ()), // W captures E4
        ] {
            g.play(Move::Play(pt(mv))).unwrap();
        }
        assert_eq!(g.at(pt("E4")), None);
        assert_eq!(g.ko_point(), Some(pt("E4")));
        // Black may not retake at once.
        assert_eq!(g.play(Move::Play(pt("E4"))), Err(Illegal::Ko));
        // Play elsewhere, and the ko clears.
        g.play(Move::Play(pt("T3"))).unwrap(); // B
        g.play(Move::Play(pt("T4"))).unwrap(); // W
        assert_eq!(g.ko_point(), None);
        assert!(g.play(Move::Play(pt("E4"))).is_ok());
    }

    #[test]
    fn liberties_and_atari_detection() {
        let mut g = Game::new();
        g.play(Move::Play(pt("D4"))).unwrap(); // B
        assert_eq!(g.group_at(pt("D4")).unwrap().liberties.len(), 4);
        g.play(Move::Play(pt("D5"))).unwrap(); // W
        g.play(Move::Play(pt("T1"))).unwrap(); // B
        g.play(Move::Play(pt("D3"))).unwrap(); // W
        g.play(Move::Play(pt("T2"))).unwrap(); // B
        g.play(Move::Play(pt("C4"))).unwrap(); // W
        let g4 = g.group_at(pt("D4")).unwrap();
        assert_eq!(g4.liberties.len(), 1);
        assert_eq!(g4.liberties[0], pt("E4"));
    }

    #[test]
    fn two_passes_end_the_game() {
        let mut g = Game::new();
        g.play(Move::Pass).unwrap();
        assert!(!g.is_over());
        g.play(Move::Pass).unwrap();
        assert!(g.is_over());
        assert_eq!(g.play(Move::Play(pt("D4"))), Err(Illegal::GameOver));
    }

    #[test]
    fn area_score_counts_surrounded_empty_region() {
        let mut g = Game::new();
        // Black walls off the A19/B19 corner; White takes stones far away so
        // the rest of the board borders both colors and stays neutral.
        for (b, w) in [("A18", "T1"), ("B18", "T2"), ("C19", "T3")] {
            g.play(Move::Play(pt(b))).unwrap();
            g.play(Move::Play(pt(w))).unwrap();
        }
        let (black, white) = g.area_score();
        // 3 stones + the 2-point corner (A19, B19).
        assert_eq!(black, 5);
        assert_eq!(white, 3);
    }

    #[test]
    fn groups_reports_each_group_once() {
        let mut g = Game::new();
        g.play(Move::Play(pt("D4"))).unwrap(); // B
        g.play(Move::Play(pt("Q16"))).unwrap(); // W
        g.play(Move::Play(pt("D5"))).unwrap(); // B, joins D4
        let groups = g.groups();
        assert_eq!(groups.len(), 2);
        let black = groups.iter().find(|g| g.color == Color::Black).unwrap();
        assert_eq!(black.stones.len(), 2);
    }
}
