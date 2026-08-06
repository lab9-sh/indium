# Proposal: hydrogen support for long-running agent loops

Status: draft / unimplemented
Motivating consumer: a human-vs-LLM Go game (19x19, ~300 moves, one model turn per move)

## Motivation

hydrogen today models a conversation as an append-only transcript that the caller
never edits. That is the right shape for chat. It does not work for a long-running
*environment* loop, where the same state is re-presented every turn in an updated
form and the transcript would otherwise accumulate hundreds of superseded copies.

The concrete driver is a Go game. Each turn the model needs to see the board plus
derived analysis (groups in atari, liberty counts, ko point, captures, its own
scratchpad). Rendered, that block is roughly 700–900 tokens. A full game is 250–350
moves. Carried naively:

| approach | context at move 250 | correctness |
|---|---|---|
| board in every turn | ~200k tokens | 249 of 250 boards are stale and wrong |
| board only in the tail, rebuilt each turn | ~12k tokens | exactly one board, always current |

The second approach — call it the **tail state block** — is what this proposal
enables. It is not reachable with the current API.

### The tail state block pattern

One "fat" block (board + analysis + scratchpad) always sits at the end of the
transcript. Once the model has responded to it, it is **demoted in place** to a
one-line stub, and a freshly rendered fat block is appended.

```
turn 47 sends:  [system, thin_1, a_1, ..., thin_46, a_46, FAT_47]
                                                            ^ ~900 tok

turn 48 sends:  [system, thin_1, a_1, ..., thin_46, a_46, thin_47, a_47, FAT_48]
                └──────────── unchanged, cache hit ──────┘ └── reprocessed ──┘
```

Demotion replaces rather than deletes, so the assistant turn above the block is
still answering something that exists. Because the edit is always at a fixed depth
from the end, the stable prefix grows monotonically and prompt caching keeps
working — unlike front-truncation, which invalidates every cached token.

## Summary

| # | Feature | Priority | Blocking? |
|---|---|---|---|
| 1 | [Mutable / constructible transcripts](#1-mutable--constructible-transcripts) | high | yes — pattern is impossible without it |
| 2 | [`tool_choice` and parallel-call control](#2-tool_choice-and-parallel-call-control) | high | no, but affects wire correctness |

---

## 1. Mutable / constructible transcripts

### Problem

`Conversation` is append-only *and* cannot be rebuilt from outside the crate.

- `Conversation.messages` is private, and `messages()` returns `&[Message]` only —
  no `messages_mut`, `retain`, `truncate`, or `drain`
  ([src/types/conversation.rs:36](src/types/conversation.rs#L36)).
- Rebuilding a filtered copy fails too: assistant turns can only be appended via
  `push_response(Response)`, and `Response.provider` is `pub(crate)`
  ([src/types/response.rs:12](src/types/response.rs#L12)), so downstream code cannot
  construct a `Response` at all.
- Even user-side content is unreachable: `TextBlock.extras` is `pub(crate)` with no
  public constructor ([src/types/content.rs:28](src/types/content.rs#L28)), so a
  struct literal will not compile downstream. `ToolUseBlock::new` exists;
  `TextBlock` has no equivalent.

Net effect: a consumer cannot edit, filter, or reconstruct a transcript by any
means the public API provides.

### Proposed API

```rust
impl Conversation {
    /// Mutable access to the transcript for in-place rewriting.
    pub fn messages_mut(&mut self) -> &mut Vec<Message>;

    /// Append a caller-constructed turn. Does not change the provider pin.
    pub fn push_message(&mut self, msg: Message);

    /// Rebuild from parts, preserving the provider pin. `cache_key` is fresh
    /// unless carried explicitly.
    pub fn from_parts(messages: Vec<Message>, provider: Option<ProviderKind>) -> Self;
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self;   // extras: None
}

impl ToolResultBlock {
    pub fn new(id: impl Into<String>, output: ToolOutput) -> Self;
}
```

`messages_mut` is the smallest thing that unblocks the pattern; `from_parts` is
worth having for loading a saved game. Both are needed because `Conversation`
serializes with private fields — a consumer restoring from disk today has to go
through serde, which works but is not an API.

### Client example

The whole game loop, with demotion:

```rust
use hydrogen::types::{ContentBlock, Message, Role, TextBlock, ToolOutput};
use hydrogen::{Client, Conversation, RequestOptions, StopReason, ThinkingEffort};

/// Rendered fresh from `Game` every turn — never edited incrementally, so the
/// model's view cannot drift from the authoritative board.
fn fat_state_block(game: &Game) -> Message {
    let text = format!(
        "{board}\n\n\
         Groups in atari: {atari}\n\
         Low liberties (<=2): {low_libs}\n\
         Ko point: {ko}\n\
         Captures: B {b_caps}, W {w_caps}\n\
         Last 6: {recent}\n\n\
         Your scratchpad:\n{scratchpad}\n\n\
         You are Black. Play your move with the play_move tool.",
        board      = game.render_board(),
        atari      = game.groups_in_atari(),
        low_libs   = game.low_liberty_groups(),
        ko         = game.ko_point_str(),
        b_caps     = game.captures(Color::Black),
        w_caps     = game.captures(Color::White),
        recent     = game.recent_moves(6),
        scratchpad = game.scratchpad(),
    );
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text(TextBlock::new(text))],
    }
}

/// What the fat block collapses to once it has been answered. ~20 tokens.
fn thin_record(n: u32, played: Move, reply: Move) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text(TextBlock::new(format!(
            "Move {n} — you played {played}. White replied {reply}."
        )))],
    }
}

async fn play_one_move(
    client: &Client,
    conv: &mut Conversation,
    game: &mut Game,
    opts: &RequestOptions,
) -> Result<(), hydrogen::Error> {
    // 1. Demote the previous fat block, then append the current one.
    //    The tail is the only fat block that ever exists.
    if let Some(last) = conv.messages_mut().last_mut() {
        *last = thin_record(game.move_number(), game.last_own_move(), game.last_reply());
    }
    conv.push_message(fat_state_block(game));

    // 2. Ask for a move. Retry loop covers illegal-move rejections.
    for _ in 0..3 {
        let resp = client.send(conv, opts).await?;
        let tool_call = resp.message.content.iter().find_map(|b| match b {
            ContentBlock::ToolUse(t) if t.name == "play_move" => Some(t.clone()),
            _ => None,
        });
        conv.push_response(resp);

        let Some(call) = tool_call else { continue };
        let point: String = call.input["point"].as_str().unwrap_or_default().into();

        match game.try_play(&point) {
            Ok(()) => {
                conv.push_tool_result(&call.id, ToolOutput::Text("accepted".into()));
                return Ok(());
            }
            // The environment is authoritative. Reject with a specific reason and
            // let the model correct itself.
            Err(why) => {
                conv.push_tool_result(&call.id, ToolOutput::Error(format!("{point}: {why}")));
            }
        }
    }

    game.pass();
    Ok(())
}
```

### Why it is required

Without `messages_mut` there is no demotion step, so every turn's board stays in
the transcript forever. At 900 tokens per board that is ~200k of context by the end
of a game, of which all but the last board is stale state the model has to reason
past. The failure mode is not just cost — it is a model that has seen 250
contradictory boards and confidently reads a stone off the wrong one.

Front-truncation is not a substitute. Dropping old turns invalidates the cached
prefix from the first surviving message onward, so every eviction costs a full
re-read; and it throws away the model's own commentary, which is the cheap part
worth keeping.

### Notes for implementation

- `push_message` must not touch the provider pin — only `push_response` observes a
  provider. A rebuilt transcript keeps whatever pin `from_parts` was given.
- `ReasoningBlock` has no public constructor (all fields `pub(crate)`,
  [src/types/content.rs:69](src/types/content.rs#L69)). That is fine and probably
  should stay that way — the payload is provider-signed. It does mean a rebuilt
  transcript silently loses reasoning blocks, which is acceptable between turns but
  **not** mid-tool-loop: Anthropic rejects a tool result whose preceding assistant
  turn lost its thinking block. Demote message *content*; never drop messages.

---

## 2. `tool_choice` and parallel-call control

### Problem

[`RequestOptions`](src/types/request.rs#L6) exposes `tools` but no way to require a
call. Two consequences for a game loop:

1. The model can answer a move request with prose instead of calling `play_move`,
   which means falling back to scraping a coordinate out of free text.
2. Nothing prevents two `play_move` calls in one turn. Every provider supports
   disabling this; hydrogen has no field for it.

All three backends already support both knobs, so this is purely a missing portable
surface:

| provider | forced call | disable parallel |
|---|---|---|
| Anthropic | `tool_choice: {"type": "tool", "name": …}` | `tool_choice.disable_parallel_tool_use` |
| OpenAI (Responses) | `tool_choice: {"type": "function", "name": …}` | `parallel_tool_calls: false` |
| xAI | OpenAI-compatible | OpenAI-compatible |

### Proposed API

```rust
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,           // model decides (current behavior)
    Required,       // must call some tool
    Tool(String),   // must call this specific tool
    None,           // tools visible but not callable
}

pub struct RequestOptions {
    // ...existing fields...
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// `None` leaves the provider default untouched.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
}
```

`parallel_tool_calls` is `Option<bool>` rather than `bool` deliberately:
`RequestOptions` derives `Default`, and a bare `bool` would default to `false`,
silently changing behavior for every existing caller. (The `web_search: bool` field
gets away with `false` because off *is* the current behavior;
[src/types/request.rs:17](src/types/request.rs#L17).)

### Client example

Two option sets over the same conversation — the move turn is forced, the
reflection turn is not:

```rust
use hydrogen::types::{ToolChoice, ToolDef};

fn play_move_tool() -> ToolDef {
    ToolDef {
        name: "play_move".into(),
        description: "Place a stone. Columns A-T excluding I; rows 1-19.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "point":     { "type": "string", "pattern": "^[A-HJ-T](1[0-9]|[1-9])$" },
                "reasoning": { "type": "string", "description": "One line: why here." }
            },
            "required": ["point"],
            "additionalProperties": false
        }),
    }
}

// Move turn: the model MUST emit exactly one play_move call.
let move_opts = RequestOptions {
    model: "claude-sonnet-4-20250514".into(),
    system: Some(GO_RULES_AND_CONVENTIONS.into()),
    tools: vec![play_move_tool(), update_notes_tool()],
    tool_choice: ToolChoice::Tool("play_move".into()),
    parallel_tool_calls: Some(false),
    thinking: Some(ThinkingEffort::High),
    ..Default::default()
};

// End-of-game review: let it talk instead.
let review_opts = RequestOptions {
    tool_choice: ToolChoice::None,
    ..move_opts.clone()
};
```

### Why it is required

Not strictly blocking — you can parse prose — but every turn parsed out of free
text is a turn that can fail in a new way, and a 300-move game will find all of
them. `ToolChoice::Tool` plus a `pattern`-constrained schema turns move extraction
from a parsing problem into a validation problem, and validation you were doing
anyway (occupied / suicide / ko).

`parallel_tool_calls: Some(false)` matters because the sensible tool set includes
`update_notes` alongside `play_move`. Without it the model will sometimes emit both
in one turn, and the loop has to decide whether the notes were written before or
after a move that has not been validated yet. Forbidding it at the wire level is
cheaper than ordering it after the fact.

---

## Non-goals

- **A built-in truncation or summarization helper.** Policy belongs to the
  consumer; the crate only needs to stop preventing it. `messages_mut` is enough.
- **Making `ReasoningBlock` constructible.** Payloads are provider-signed and
  hand-built substitutes are rejected. Opaque is correct.
- **Relaxing provider pinning.** Cross-provider transcript reuse is a real hazard
  and the current error is the right behavior.
- **A local tokenizer.** Three providers, three vocabularies, and the demotion
  pattern removes the need for pre-flight sizing.
- **Usage accounting on `Conversation`.** Per-`Response` `Usage` is already
  available; cumulative totals and demotion telemetry can be tracked by the
  consumer without crate support.
- **Retry policy for transient failures.** Typed errors (`RateLimited`,
  `Transport`, `Http`) already give the consumer everything needed to write a
  retry loop (and checkpoint the game between attempts).
