//! Indium CLI.
//!
//! Two modes over the same loop:
//!   - `--opponent human` plays a real game at the terminal.
//!   - `--opponent bot` runs headless for N moves and prints the telemetry that
//!     tells us whether the tail-state-block pattern is actually holding.

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use hydrogen::{AnthropicConfig, Client, ToolChoice};
use indium::agent::Agent;
use indium::goban::{Color, Game, Move, Point};
use indium::record::Recorder;

const USAGE: &str = "\
indium — human vs LLM Go on 19x19

USAGE:
    indium [OPTIONS]

OPTIONS:
    --opponent <human|bot>   Who plays Black. Default: human.
    --moves <N>              Stop after N total moves. Default: 0 (unlimited).
    --model <ID>             Default: claude-opus-5.
    --tool-choice <MODE>     auto | required | forced. Default: auto.
                             NOTE: required/forced suppress the model's
                             reasoning entirely — see README.
    --collapse-retries       Drop assistant/tool_result pairs left by rejected
                             moves instead of keeping them.
    --max-attempts <N>       Illegal-move retries before forcing a pass. Default: 3.
    --keep-reasoning <N>     Keep reasoning blocks on only the N most recent
                             assistant turns. Default: all. Retained reasoning
                             is the dominant context-growth term.
    --seed <N>               Bot opponent seed. Default: 1.
    --out <DIR>              Where to write game.sgf and prompts.log.
                             Default: ./games/latest.
    -h, --help               Show this help.
";

struct Opts {
    bot: bool,
    moves: u32,
    model: String,
    tool_choice: ToolChoice,
    collapse_retries: bool,
    max_attempts: usize,
    seed: u64,
    out: String,
    keep_reasoning: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            bot: false,
            moves: 0,
            model: "claude-opus-5".into(),
            tool_choice: ToolChoice::Auto,
            collapse_retries: false,
            max_attempts: 3,
            seed: 1,
            out: "games/latest".into(),
            keep_reasoning: usize::MAX,
        }
    }
}

fn parse_args() -> Result<Option<Opts>, String> {
    let mut o = Opts::default();
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = |name: &str| -> Result<String, String> {
            args.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--opponent" => o.bot = matches!(next("--opponent")?.as_str(), "bot"),
            "--moves" => o.moves = next("--moves")?.parse().map_err(|e| format!("{e}"))?,
            "--model" => o.model = next("--model")?,
            "--max-attempts" => {
                o.max_attempts = next("--max-attempts")?.parse().map_err(|e| format!("{e}"))?
            }
            "--seed" => o.seed = next("--seed")?.parse().map_err(|e| format!("{e}"))?,
            "--keep-reasoning" => {
                o.keep_reasoning = next("--keep-reasoning")?
                    .parse()
                    .map_err(|e| format!("{e}"))?
            }
            "--out" => o.out = next("--out")?,
            "--collapse-retries" => o.collapse_retries = true,
            "--tool-choice" => {
                o.tool_choice = match next("--tool-choice")?.as_str() {
                    "auto" => ToolChoice::Auto,
                    "required" => ToolChoice::Required,
                    "forced" => ToolChoice::Tool("play_move".into()),
                    other => return Err(format!("unknown --tool-choice '{other}'")),
                }
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Some(o))
}

/// Minimal `.env` reader so the POC has no config dependency.
fn load_env() {
    let Ok(text) = fs::read_to_string(".env") else {
        return;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let v = v.trim().trim_matches('"').trim_matches('\'');
            if env::var(k.trim()).is_err() {
                unsafe { env::set_var(k.trim(), v) };
            }
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(Some(o)) => o,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    load_env();
    let Ok(key) = env::var("ANTHROPIC_API_KEY") else {
        eprintln!("error: ANTHROPIC_API_KEY is not set (checked env and ./.env)");
        return ExitCode::FAILURE;
    };

    match run(opts, key).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run(opts: Opts, key: String) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::anthropic(AnthropicConfig::new(key));
    // The model plays White, so the human (or bot) opens as Black.
    let mut agent = Agent::new(
        client,
        opts.model.clone(),
        Color::White,
        opts.tool_choice.clone(),
        opts.max_attempts,
        opts.collapse_retries,
        opts.keep_reasoning,
    );
    let mut game = Game::new();
    let mut rec = Recorder::new(&opts.out)?;
    let mut rng = Rng::new(opts.seed);

    println!("indium — you are Black (X), {} is White (O)", opts.model);
    println!("commands: a coordinate like Q16, or 'pass', 'resign', 'quit'\n");

    while !game.is_over() {
        if opts.moves > 0 && game.history().len() as u32 >= opts.moves {
            println!("\n[stopping after {} moves as requested]", opts.moves);
            break;
        }

        match game.to_move() {
            Color::Black => {
                println!("{}\n", game.render_board());
                let mv = if opts.bot {
                    let mv = bot_move(&game, &mut rng);
                    println!("Black (bot) plays {mv}");
                    mv
                } else {
                    match read_human_move(&game)? {
                        Some(mv) => mv,
                        None => {
                            println!("resigned.");
                            game.resign(Color::Black);
                            break;
                        }
                    }
                };
                let rec_ = game.play(mv)?;
                if rec_.captured > 0 {
                    println!("  captured {} stone(s)", rec_.captured);
                }
                // Show the stone landing before the model takes its turn.
                if !opts.bot {
                    println!("\n{}\n", game.render_board());
                }
            }
            Color::White => {
                let n = game.move_number();
                print!("White (model) is thinking… ");
                io::stdout().flush().ok();

                let (outcome, fat) = agent.take_turn(&game).await?;
                let applied = match game.validate(outcome.mv) {
                    Ok(()) => outcome.mv,
                    // Belt and braces: the environment stays authoritative even
                    // if validation and application ever diverge.
                    Err(why) => {
                        println!("\n  [referee rejected {} at apply time: {why}]", outcome.mv);
                        Move::Pass
                    }
                };
                let played = game.play(applied)?;

                println!("plays {}", played.mv);
                if outcome.forced_pass {
                    println!("  [forced pass after {} attempts]", outcome.attempts);
                }
                if played.captured > 0 {
                    println!("  captured {} stone(s)", played.captured);
                }
                // The play_move `reasoning` argument is the one line that is
                // present on every turn; prose only appears on the first.
                for line in [
                    outcome.rationale.trim(),
                    outcome.commentary.trim(),
                    outcome.reasoning.trim(),
                ] {
                    if !line.is_empty() {
                        println!("  \"{}\"", first_line(line, 160));
                        break;
                    }
                }

                rec.log_turn(
                    n,
                    &fat,
                    &format!(
                        "played {} (attempts {}, forced_pass {})\nrationale: {}\n\
                         commentary: {}\nreasoning: {}",
                        played.mv,
                        outcome.attempts,
                        outcome.forced_pass,
                        outcome.rationale,
                        outcome.commentary,
                        outcome.reasoning
                    ),
                )?;
            }
        }
    }

    agent.finish();
    println!("\n{}\n", game.render_board());
    report(&game, &agent, &opts);

    let sgf = rec.write_sgf(game.history(), "Human", &opts.model)?;
    println!("\nsgf:     {}", sgf.display());
    println!("prompts: {}", rec.dir().join("prompts.log").display());
    Ok(())
}

fn report(game: &Game, agent: &Agent, opts: &Opts) {
    let s = &agent.stats;
    let (b, w) = game.area_score();

    println!("=== result ===");
    if let Some(c) = game.resigned_by() {
        println!("{c} resigned.");
    }
    println!("captures: Black {}, White {}", game.captures(Color::Black), game.captures(Color::White));
    println!("provisional area score (no dead-stone removal): Black {b}, White {w}");

    println!("\n=== loop telemetry ===");
    println!("model moves:        {}", s.total_moves());
    println!("api calls:          {}", s.total_api_calls());
    println!(
        "illegal proposals:  {} ({:.1}% of calls)",
        s.rejections,
        s.rejection_rate() * 100.0
    );
    println!("no-tool-call turns: {}", s.no_tool_call);
    println!("parallel calls:     {} (expected 0)", s.parallel_calls_seen);
    println!("forced passes:      {}", s.forced_passes);
    println!("notes updates:      {}", s.notes_updates);
    println!("reasoning stripped: {}", s.reasoning_stripped);
    println!("final messages:     {}", agent.message_count());

    if s.turns.is_empty() {
        return;
    }
    println!("\n=== context growth (the thing the pattern exists to keep flat) ===");
    println!(
        "{:>5}  {:>5}  {:>9}  {:>9}  {:>9}  {:>7}",
        "move", "msgs", "total_in", "cache_rd", "uncached", "out"
    );
    for t in &s.turns {
        println!(
            "{:>5}  {:>5}  {:>9}  {:>9}  {:>9}  {:>7}",
            t.move_number,
            t.messages,
            t.total_input(),
            t.cache_read,
            t.input_tokens + t.cache_creation,
            t.output_tokens
        );
    }

    let first = s.turns.first().expect("non-empty");
    let last = s.turns.last().expect("non-empty");
    let moves = s.turns.len().max(2) - 1;
    if moves > 0 {
        let growth = last.total_input() as f64 - first.total_input() as f64;
        println!(
            "\ncontext grew {:.0} tokens over {} model moves = {:.0} tok/move",
            growth,
            moves,
            growth / moves as f64
        );
        println!("(a naive board-per-turn loop would grow by the full fat block, ~700-900/move)");
    }
    let cache_read: u32 = s.turns.iter().map(|t| t.cache_read).sum();
    let billed: u32 = s.turns.iter().map(|t| t.total_input()).sum();
    if billed > 0 {
        println!(
            "prompt cache served {cache_read}/{billed} input tokens ({:.0}%)",
            cache_read as f64 / billed as f64 * 100.0
        );
    }
    if matches!(opts.tool_choice, ToolChoice::Auto) {
        println!("tool_choice=auto: model reasoning preserved");
    } else {
        println!("tool_choice != auto: reasoning is suppressed by the API (see README)");
    }
}

fn first_line(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or_default();
    if line.chars().count() <= max {
        return line.to_string();
    }
    let cut: String = line.chars().take(max).collect();
    format!("{cut}…")
}

fn read_human_move(game: &Game) -> Result<Option<Move>, Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    loop {
        print!("Black> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(None); // EOF
        }
        let line = line.trim();
        match line.to_ascii_lowercase().as_str() {
            "" => continue,
            "quit" | "resign" => return Ok(None),
            "pass" => return Ok(Some(Move::Pass)),
            _ => {}
        }
        let point = match Point::parse(line) {
            Ok(p) => p,
            Err(e) => {
                println!("  {e}");
                continue;
            }
        };
        match game.validate(Move::Play(point)) {
            Ok(()) => return Ok(Some(Move::Play(point))),
            Err(e) => println!("  {point}: {e}"),
        }
    }
}

/// Opponent that prefers contact play, so the game reaches real fights (and so
/// the atari/liberty half of the fat block actually gets exercised).
fn bot_move(game: &Game, rng: &mut Rng) -> Move {
    let legal = game.legal_moves();
    if legal.is_empty() {
        return Move::Pass;
    }
    let contact: Vec<Point> = legal
        .iter()
        .copied()
        .filter(|p| has_neighbor_stone(game, *p))
        .collect();
    // Mostly play near existing stones; occasionally break away so the game
    // does not collapse into one corner.
    let pool = if !contact.is_empty() && rng.next_u32() % 4 != 0 {
        &contact
    } else {
        &legal
    };
    Move::Play(pool[(rng.next_u32() as usize) % pool.len()])
}

fn has_neighbor_stone(game: &Game, p: Point) -> bool {
    let (r, c) = (p.row as i32, p.col as i32);
    [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        .iter()
        .any(|(dr, dc)| {
            let (nr, nc) = (r + dr, c + dc);
            (0..19).contains(&nr)
                && (0..19).contains(&nc)
                && game.at(Point::new(nr as usize, nc as usize)).is_some()
        })
}

/// xorshift64*, so the bot is reproducible without pulling in `rand`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}
