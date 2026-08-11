//! The tail-state-block agent loop.
//!
//! Exactly one fat block exists in the transcript at any time, and it always
//! sits at the end. Once the model has answered it, it is **demoted in place**
//! to a one-line stub and a freshly rendered fat block is appended. Demotion
//! replaces content and never removes a message, so the assistant turn above
//! it is still answering something that exists.
//!
//! The fat block rides inside a `tool_result` whenever one is owed, which makes
//! the whole game a single continuous tool loop. That is why demotion has to
//! preserve the block *kind*: rewriting a `tool_result` into a plain text
//! message would orphan the assistant's `tool_use` and Anthropic would reject
//! the next request.

use hydrogen::types::{ContentBlock, Message, Role, TextBlock, ToolResultBlock, ToolUseBlock};
use hydrogen::{
    Client, Conversation, Error, RequestOptions, ThinkingEffort, ToolDef, ToolOutput, Usage,
};
use serde_json::json;

use crate::goban::{Color, Game, Illegal, Move, Point};
use crate::prompt::{fat_state_block, thin_stub, Notes};

/// Per-move telemetry. The growth curve of `total_input` across moves is the
/// thing the tail-state-block pattern exists to keep flat.
#[derive(Debug, Clone, Default)]
pub struct TurnStat {
    pub move_number: u32,
    pub messages: usize,
    pub api_calls: usize,
    pub rejections: usize,
    pub input_tokens: u32,
    pub cache_creation: u32,
    pub cache_read: u32,
    pub output_tokens: u32,
}

impl TurnStat {
    /// Every prompt token billed this move, cached or not. `input_tokens`
    /// alone reads as ~2 on a cached prefix and is useless for sizing.
    pub fn total_input(&self) -> u32 {
        self.input_tokens + self.cache_creation + self.cache_read
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub turns: Vec<TurnStat>,
    pub rejections: usize,
    pub no_tool_call: usize,
    pub forced_passes: usize,
    pub notes_updates: usize,
}

impl Stats {
    pub fn total_moves(&self) -> usize {
        self.turns.len()
    }

    pub fn total_api_calls(&self) -> usize {
        self.turns.iter().map(|t| t.api_calls).sum()
    }

    pub fn rejection_rate(&self) -> f64 {
        let calls = self.total_api_calls();
        if calls == 0 {
            return 0.0;
        }
        self.rejections as f64 / calls as f64
    }
}

/// Outcome of one model turn, for logging.
pub struct TurnOutcome {
    pub mv: Move,
    pub attempts: usize,
    /// Assistant prose. Only appears on the first turn: once the fat block
    /// rides in a `tool_result` the model continues the tool loop and emits
    /// `[thinking, tool_use]` with no text block.
    pub commentary: String,
    /// Summarized thinking, which is where the reasoning actually lives for
    /// every turn after the first.
    pub reasoning: String,
    /// The `reasoning` argument of the `play_move` call itself.
    pub rationale: String,
    pub forced_pass: bool,
}

pub struct Agent {
    client: Client,
    conv: Conversation,
    opts: RequestOptions,
    color: Color,
    notes: Notes,
    /// Index of the message currently holding the fat block.
    fat_carrier: Option<usize>,
    /// A `play_move` call we deliberately left unanswered so the next fat
    /// block can ride in its `tool_result`.
    pending_tool_id: Option<String>,
    /// Move number the outstanding fat block asked about, for its stub.
    pending_stub_for: Option<u32>,
    max_attempts: usize,
    pub stats: Stats,
}

impl Agent {
    pub fn new(
        client: Client,
        model: String,
        color: Color,
        thinking: ThinkingEffort,
        max_attempts: usize,
    ) -> Self {
        let opts = RequestOptions {
            model,
            system: Some(crate::prompt::SYSTEM_PROMPT.into()),
            tools: vec![play_move_tool(), update_notes_tool()],
            thinking: Some(thinking),
            max_tokens: Some(8192),
            // The tail message is rebuilt every turn, so the breakpoint has to
            // sit one back from it — on the last message that will not change.
            cache_breakpoint_from_end: Some(1),
            ..Default::default()
        };
        Self {
            client,
            conv: Conversation::new(),
            opts,
            color,
            notes: Notes::default(),
            fat_carrier: None,
            pending_tool_id: None,
            pending_stub_for: None,
            max_attempts,
            stats: Stats::default(),
        }
    }

    pub fn message_count(&self) -> usize {
        self.conv.messages().len()
    }

    pub fn conversation(&self) -> &Conversation {
        &self.conv
    }

    pub fn notes(&self) -> &Notes {
        &self.notes
    }

    /// Ask the model for one move. The game is *not* mutated here — the caller
    /// applies the returned move, keeping the environment authoritative.
    pub async fn take_turn(&mut self, game: &Game) -> Result<(TurnOutcome, String), Error> {
        let fat = fat_state_block(game, self.color, &self.notes);
        self.deliver_state(game, fat.clone());

        let mut stat = TurnStat {
            move_number: game.move_number(),
            ..Default::default()
        };
        let mut commentary = String::new();
        let mut reasoning = String::new();

        for _ in 0..self.max_attempts {
            let resp = self.client.send(&self.conv, &self.opts).await?;
            stat.api_calls += 1;
            accumulate(&mut stat, &resp.usage);

            let calls: Vec<ToolUseBlock> = resp
                .message
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse(t) => Some(t.clone()),
                    _ => None,
                })
                .collect();
            for b in &resp.message.content {
                match b {
                    ContentBlock::Text(t) => commentary.push_str(&t.text),
                    ContentBlock::Reasoning(r) => {
                        if let Some(s) = r.summary() {
                            reasoning.push_str(s);
                        }
                    }
                    _ => {}
                }
            }
            self.conv.push_response(resp);

            if calls.is_empty() {
                // Model answered in prose instead of calling a tool.
                self.stats.no_tool_call += 1;
                self.conv
                    .push_user("Place a stone by calling the play_move tool. Do that now.");
                continue;
            }

            // Handle every tool_use so none are left unanswered. Notes get an
            // immediate result; a successful play_move is left pending so the
            // next fat block can ride in its tool_result. Both in one response
            // is fine: notes apply to local state, then the move is returned.
            let mut accepted: Option<(Move, String)> = None;
            for call in calls {
                match call.name.as_str() {
                    "update_notes" => {
                        self.apply_notes(&call);
                        self.stats.notes_updates += 1;
                        self.conv
                            .push_tool_result(&call.id, ToolOutput::Text("notes updated".into()));
                    }
                    "play_move" => {
                        if self.pending_tool_id.is_some() {
                            self.conv.push_tool_result(
                                &call.id,
                                ToolOutput::Error(
                                    "already accepted a move this turn; call play_move once"
                                        .into(),
                                ),
                            );
                            continue;
                        }
                        match self.validate(game, &call) {
                            Ok(mv) => {
                                self.pending_tool_id = Some(call.id);
                                self.pending_stub_for = Some(game.move_number());
                                let rationale = call
                                    .input
                                    .get("reasoning")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .to_string();
                                accepted = Some((mv, rationale));
                            }
                            Err(why) => {
                                self.stats.rejections += 1;
                                stat.rejections += 1;
                                self.conv.push_tool_result(&call.id, ToolOutput::Error(why));
                            }
                        }
                    }
                    other => {
                        self.conv.push_tool_result(
                            &call.id,
                            ToolOutput::Error(format!("unknown tool '{other}'")),
                        );
                    }
                }
            }

            if let Some((mv, rationale)) = accepted {
                stat.messages = self.conv.messages().len();
                let attempts = stat.api_calls;
                self.stats.turns.push(stat);
                return Ok((
                    TurnOutcome {
                        mv,
                        attempts,
                        commentary,
                        reasoning,
                        rationale,
                        forced_pass: false,
                    },
                    fat,
                ));
            }
        }

        // Out of attempts. Pass rather than let a broken turn stall the game.
        self.stats.forced_passes += 1;
        self.pending_stub_for = Some(game.move_number());
        stat.messages = self.conv.messages().len();
        self.stats.turns.push(stat);
        Ok((
            TurnOutcome {
                mv: Move::Pass,
                attempts: self.max_attempts,
                commentary,
                reasoning,
                rationale: String::new(),
                forced_pass: true,
            },
            fat,
        ))
    }

    /// Demote the outstanding fat block, then append the new one.
    fn deliver_state(&mut self, game: &Game, fat: String) {
        if let (Some(idx), Some(number)) = (self.fat_carrier, self.pending_stub_for) {
            let stub = self.stub_text(game, number);
            demote_in_place(self.conv.messages_mut(), idx, stub);
        }

        match self.pending_tool_id.take() {
            Some(id) => self.conv.push_tool_result(&id, ToolOutput::Text(fat)),
            None => self.conv.push_message(Message {
                role: Role::User,
                content: vec![ContentBlock::Text(TextBlock::new(fat))],
            }),
        }
        self.fat_carrier = Some(self.conv.messages().len() - 1);
    }

    /// "Move 47 — you played Q3. Opponent replied R4."
    fn stub_text(&self, game: &Game, number: u32) -> String {
        let hist = game.history();
        let own = hist
            .iter()
            .find(|r| r.number == number)
            .map(|r| r.mv)
            .unwrap_or(Move::Pass);
        let reply = hist.iter().find(|r| r.number == number + 1).map(|r| r.mv);
        thin_stub(number, own, reply)
    }

    /// The environment is authoritative. Validation never mutates the game.
    fn validate(&self, game: &Game, call: &ToolUseBlock) -> Result<Move, String> {
        let raw = call
            .input
            .get("point")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if raw.eq_ignore_ascii_case("pass") {
            return Ok(Move::Pass);
        }
        let point: Point = Point::parse(raw).map_err(|e| format!("{raw}: {e}"))?;
        match game.validate(Move::Play(point)) {
            Ok(()) => Ok(Move::Play(point)),
            Err(Illegal::Occupied(c)) => Err(format!(
                "{point}: occupied by {c}. Pick an empty intersection."
            )),
            Err(e) => Err(format!("{point}: {e}")),
        }
    }

    fn apply_notes(&mut self, call: &ToolUseBlock) {
        if let Some(s) = call.input.get("strategy").and_then(|v| v.as_str()) {
            self.notes.set_strategy(s);
        }
        if let Some(s) = call.input.get("tactics").and_then(|v| v.as_str()) {
            self.notes.set_tactics(s);
        }
    }

    /// Answer any outstanding tool call so the saved transcript is well-formed.
    pub fn finish(&mut self) {
        if let Some(id) = self.pending_tool_id.take() {
            self.conv
                .push_tool_result(&id, ToolOutput::Text("game over".into()));
        }
    }
}

/// Replace a fat block with its stub, in place, keeping the block kind.
///
/// A `tool_result` must stay a `tool_result`: rewriting it as plain text would
/// leave the preceding assistant turn's `tool_use` unanswered, which Anthropic
/// rejects. The message is never removed, so the turn above it still has
/// something to answer.
pub fn demote_in_place(msgs: &mut [Message], idx: usize, stub: String) {
    let Some(msg) = msgs.get_mut(idx) else { return };
    let demoted = match msg.content.first() {
        Some(ContentBlock::ToolResult(tr)) => {
            ContentBlock::ToolResult(ToolResultBlock::new(tr.id.clone(), ToolOutput::Text(stub)))
        }
        _ => ContentBlock::Text(TextBlock::new(stub)),
    };
    msg.content = vec![demoted];
}

fn accumulate(stat: &mut TurnStat, u: &Usage) {
    stat.input_tokens += u.input_tokens;
    stat.cache_creation += u.cache_creation_input_tokens;
    stat.cache_read += u.cache_read_input_tokens;
    stat.output_tokens += u.output_tokens;
}

pub fn play_move_tool() -> ToolDef {
    ToolDef {
        name: "play_move".into(),
        description: "Place a stone at a point, or pass. Columns A-T excluding I; rows 1-19."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "point": {
                    "type": "string",
                    "pattern": "^([A-HJ-T](1[0-9]|[1-9])|pass)$",
                    "description": "Coordinate such as Q16, or 'pass'."
                },
                "reasoning": { "type": "string", "description": "One line: why here." }
            },
            "required": ["point"],
            "additionalProperties": false
        }),
    }
}

pub fn update_notes_tool() -> ToolDef {
    ToolDef {
        name: "update_notes".into(),
        description: "Rewrite your scratchpad. Strategy is durable; tactics are per-fight and \
                      expected to be replaced often. Both are truncated if long."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "strategy": { "type": "string", "description": "Durable plan, e.g. 'playing for influence on the left'." },
                "tactics":  { "type": "string", "description": "Current fight only; expires." }
            },
            "additionalProperties": false
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The move schema is the first line of defence, so pin its shape: `I` must
    /// be excluded and rows must not reach 20.
    #[test]
    fn play_move_schema_excludes_i_and_caps_rows() {
        let schema = play_move_tool().input_schema;
        let pattern = schema["properties"]["point"]["pattern"].as_str().unwrap();
        assert!(pattern.contains("A-HJ-T"), "I must be excluded: {pattern}");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["required"][0], "point");
    }

    #[test]
    fn notes_schema_has_both_tiers() {
        let schema = update_notes_tool().input_schema;
        assert!(schema["properties"]["strategy"].is_object());
        assert!(schema["properties"]["tactics"].is_object());
    }

    fn user_text(t: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text(TextBlock::new(t))],
        }
    }

    fn user_tool_result(id: &str, t: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResultBlock::new(
                id,
                ToolOutput::Text(t.into()),
            ))],
        }
    }

    /// The failure this guards against is silent until the API rejects it:
    /// demoting a tool_result into plain text orphans the assistant's tool_use.
    #[test]
    fn demotion_keeps_a_tool_result_a_tool_result_with_the_same_id() {
        let mut msgs = vec![user_tool_result("toolu_42", "FAT board state …")];
        demote_in_place(&mut msgs, 0, "Move 47 — you played Q3.".into());

        assert_eq!(msgs.len(), 1, "demotion must never remove a message");
        match &msgs[0].content[..] {
            [ContentBlock::ToolResult(tr)] => {
                assert_eq!(tr.id, "toolu_42", "tool_use id must survive");
                assert_eq!(tr.output, ToolOutput::Text("Move 47 — you played Q3.".into()));
            }
            other => panic!("expected a tool_result, got {other:?}"),
        }
    }

    #[test]
    fn demotion_keeps_a_plain_user_turn_plain() {
        let mut msgs = vec![user_text("FAT board state …")];
        demote_in_place(&mut msgs, 0, "Move 1 — you played Q16.".into());
        match &msgs[0].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "Move 1 — you played Q16."),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn demotion_of_a_missing_index_is_a_no_op() {
        let mut msgs = vec![user_text("only message")];
        demote_in_place(&mut msgs, 7, "stub".into());
        assert_eq!(msgs.len(), 1);
    }

    /// Only the tail is ever fat: after demoting, no earlier message may still
    /// carry a full board.
    #[test]
    fn only_one_fat_block_survives_a_demotion_cycle() {
        let mut msgs = vec![
            user_tool_result("t1", "FAT 1"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(TextBlock::new("I play D4"))],
            },
            user_tool_result("t2", "FAT 2"),
        ];
        demote_in_place(&mut msgs, 0, "Move 1 — you played Q16.".into());
        let fat_count = msgs
            .iter()
            .filter(|m| {
                m.content.iter().any(|b| match b {
                    ContentBlock::ToolResult(tr) => {
                        matches!(&tr.output, ToolOutput::Text(s) if s.starts_with("FAT"))
                    }
                    _ => false,
                })
            })
            .count();
        assert_eq!(fat_count, 1);
    }

    #[test]
    fn turn_stat_total_input_sums_cached_and_uncached() {
        let s = TurnStat {
            input_tokens: 2,
            cache_creation: 900,
            cache_read: 11_000,
            ..Default::default()
        };
        assert_eq!(s.total_input(), 11_902);
    }
}
