//! The tail-state-block agent loop (Shape B).
//!
//! Exactly one **volatile fat** user message exists in the transcript at a
//! time. After the model answers and the environment accepts a move, the next
//! `deliver_state`:
//!
//! 1. Answers the pending `play_move` with a thin **stable** `tool_result` ack.
//! 2. Demotes the previous fat user turn to a one-line stub (via hydrogen's
//!    volatile slot).
//! 3. Appends a freshly rendered fat user block as the new volatile.
//!
//! Tool results are never rewritten. See `PLAN-volatile-fat-blocks.md`.

use hydrogen::types::{ContentBlock, ToolUseBlock};
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
    /// Assistant prose when the model emits a text block (often first turn).
    pub commentary: String,
    /// Summarized thinking, which is where the reasoning often lives.
    pub reasoning: String,
    /// The `reasoning` argument of the `play_move` call itself.
    pub rationale: String,
    pub forced_pass: bool,
}

/// Shape B agent: thin stable `play_move` ack + volatile user fat board.
pub struct Agent {
    client: Client,
    conv: Conversation,
    opts: RequestOptions,
    color: Color,
    notes: Notes,
    /// A `play_move` call accepted this turn; answered with a thin ack on the
    /// next `deliver_state` (or `"game over"` on `finish`).
    pending_tool_id: Option<String>,
    /// Ack text for that call, e.g. `ok: Q16` / `ok: pass`.
    pending_ack: Option<String>,
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
            // Cache placement is hydrogen policy: Anthropic breakpoint is
            // derived automatically from Conversation::volatile_index().
            ..Default::default()
        };
        Self {
            client,
            conv: Conversation::new(),
            opts,
            color,
            notes: Notes::default(),
            pending_tool_id: None,
            pending_ack: None,
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

            // Handle every tool_use so none are left unanswered. Notes and
            // illegal moves get immediate stable results; a successful
            // play_move is left pending for a thin ack on the next deliver.
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
                                self.pending_ack = Some(ack_text(mv));
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
        // No pending tool_id if the model never produced a valid play_move.
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

    /// Shape B deliver: thin stable ack first (if any), then rotate/push fat user.
    fn deliver_state(&mut self, game: &Game, fat: String) {
        // Anthropic requires tool_use answered before more user content.
        if let Some(id) = self.pending_tool_id.take() {
            let ack = self
                .pending_ack
                .take()
                .unwrap_or_else(|| "ok".into());
            self.conv.push_tool_result(id, ToolOutput::Text(ack));
        }

        match self.pending_stub_for.take() {
            Some(number) => {
                let stub = self.stub_text(game, number);
                self.conv.rotate_volatile_user(stub, fat);
            }
            None => {
                self.conv
                    .push_volatile_user(fat)
                    .expect("no prior volatile on first deliver");
            }
        }
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

    /// Answer any outstanding tool call and demote the final fat for a clean
    /// saved transcript.
    pub fn finish(&mut self) {
        if let Some(id) = self.pending_tool_id.take() {
            let _ = self.pending_ack.take();
            self.conv
                .push_tool_result(&id, ToolOutput::Text("game over".into()));
        }
        if self.conv.volatile_index().is_some() {
            let _ = self.conv.demote_volatile("game over");
        }
    }
}

fn ack_text(mv: Move) -> String {
    match mv {
        Move::Pass => "ok: pass".into(),
        Move::Play(p) => format!("ok: {p}"),
    }
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
    use hydrogen::types::Role;
    use hydrogen::Response;

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

    /// `Response.provider` is crate-private in hydrogen; build fixtures via serde.
    fn assistant_response(text: &str) -> Response {
        serde_json::from_value(serde_json::json!({
            "message": {
                "role": "assistant",
                "content": [{"kind": "text", "text": text}]
            },
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "provider": "anthropic"
        }))
        .expect("fixture Response")
    }

    /// Shape B demotion is hydrogen's user-text path: content-only rewrite
    /// under later assistant turns, never the assistant itself.
    #[test]
    fn volatile_user_demotion_is_content_only_under_assistant() {
        let mut conv = Conversation::new();
        conv.push_volatile_user("FAT board state …").unwrap();
        conv.push_response(assistant_response("I play Q16"));
        let len = conv.messages().len();

        conv.demote_volatile("Move 1 — you played Q16.").unwrap();

        assert_eq!(conv.messages().len(), len);
        assert_eq!(conv.volatile_index(), None);
        match &conv.messages()[0].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "Move 1 — you played Q16."),
            other => panic!("expected demoted text, got {other:?}"),
        }
        assert_eq!(conv.messages()[1].role, Role::Assistant);
    }

    #[test]
    fn rotate_volatile_user_rewrites_marked_fat_and_appends_new() {
        let mut conv = Conversation::new();
        conv.push_volatile_user("FAT 1").unwrap();
        conv.push_response(assistant_response("ok"));
        conv.rotate_volatile_user("Move 1 — you played Q16.", "FAT 2");

        assert_eq!(conv.messages().len(), 3);
        assert_eq!(conv.volatile_index(), Some(2));
        match &conv.messages()[0].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "Move 1 — you played Q16."),
            other => panic!("expected stub, got {other:?}"),
        }
        match &conv.messages()[2].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "FAT 2"),
            other => panic!("expected new fat, got {other:?}"),
        }
    }

    /// deliver_state ordering: ack the pending play_move before rotating fat,
    /// so Anthropic never sees a new user turn with an unanswered tool_use.
    #[test]
    fn deliver_state_ack_then_rotate_fat() {
        // Simulate the Shape B bookkeeping without a live client.
        let mut conv = Conversation::new();
        conv.push_volatile_user("FAT 1").unwrap();
        conv.push_response(assistant_response("tool would be here"));
        // Environment accepted a move; pending thin ack + stub for move 1.
        let mut pending_tool_id = Some("toolu_1".to_string());
        let mut pending_ack = Some("ok: Q16".to_string());
        let mut pending_stub_for = Some(1u32);

        // --- deliver_state body ---
        if let Some(id) = pending_tool_id.take() {
            let ack = pending_ack.take().unwrap_or_else(|| "ok".into());
            conv.push_tool_result(id, ToolOutput::Text(ack));
        }
        match pending_stub_for.take() {
            Some(_) => conv.rotate_volatile_user("Move 1 — you played Q16.", "FAT 2"),
            None => conv.push_volatile_user("FAT 2").unwrap(),
        }

        // Message order: thin user, assistant, stable ack tool_result, new fat.
        assert_eq!(conv.messages().len(), 4);
        assert_eq!(conv.messages()[0].role, Role::User);
        assert_eq!(conv.messages()[1].role, Role::Assistant);
        assert_eq!(conv.messages()[2].role, Role::User);
        match &conv.messages()[2].content[..] {
            [ContentBlock::ToolResult(tr)] => {
                assert_eq!(tr.id, "toolu_1");
                assert_eq!(tr.output, ToolOutput::Text("ok: Q16".into()));
            }
            other => panic!("expected stable ack tool_result, got {other:?}"),
        }
        assert_eq!(conv.volatile_index(), Some(3));
        match &conv.messages()[3].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "FAT 2"),
            other => panic!("expected fat user, got {other:?}"),
        }
        // Demoted first fat is still user text (kind preserved as text).
        match &conv.messages()[0].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "Move 1 — you played Q16."),
            other => panic!("expected demoted stub, got {other:?}"),
        }
    }

    #[test]
    fn ack_text_formats_play_and_pass() {
        assert_eq!(ack_text(Move::Pass), "ok: pass");
        let p = Point::parse("Q16").unwrap();
        assert_eq!(ack_text(Move::Play(p)), "ok: Q16");
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
