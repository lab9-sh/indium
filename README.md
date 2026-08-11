# indium

Human-vs-LLM Go on a 19x19 board, built on [hydrogen](../hydrogen). It exists to
validate the agent-loop features in [PROPOSAL-agent-loops.md](PROPOSAL-agent-loops.md)
under a real long-running environment loop, and it is a POC — the loop and the
telemetry are the deliverable, not the playing strength.

```bash
cargo run --release -- --opponent human          # play a game (Anthropic default)
cargo run --release -- --provider xai --opponent bot --moves 30
cargo run --release -- --help
```

`--provider` selects the backend (`anthropic` default, or `xai`). Keys come from
the environment or `./.env`: `ANTHROPIC_API_KEY` / `XAI_API_KEY`. Default models
are `claude-opus-5` and `grok-4.5` respectively (`--model` overrides). Each run
writes `game.sgf` and `prompts.log` (the exact fat block sent each turn, next to
what came back) to `--out`, default `games/latest`.

## What is being validated

Indium uses **Shape B** (thin stable `play_move` ack + volatile **user** fat
board). See [PLAN-volatile-fat-blocks.md](PLAN-volatile-fat-blocks.md).

| Feature | Status |
|---|---|
| `Conversation::push_volatile_user` / `rotate_volatile_user` / `demote_volatile` | used for the board tail |
| Automatic Anthropic cache placement from `volatile_index` | hydrogen policy; no consumer cache knobs |
| Stable thin `push_tool_result` ack per accepted move | Shape B; tool results never rewritten |
| `ThinkingEffort` | works; exposed as `--thinking high\|medium\|low` (default medium) |

Historical experiments briefly used free `messages_mut` and
`cache_breakpoint_from_end`; those are **not** on the current path. Findings
below remain useful as measurement history.

## Findings

### 1. Demotion destroys the prompt cache

This is the significant one, because the cost argument in the proposal rests on
it: *"the edit is always at a fixed depth from the end, so the cached prefix
grows monotonically."* That is not what happens.

Measured directly, same prompt shape, only the edit differs:

```
transcript grows by append only          transcript demotes, then appends
  turn 1  write=3835  read=0               turn 1  write=3835  read=0
  turn 2  write=2440  read=3835            turn 2  write=3866  read=0
  turn 3  write=2439  read=6275            turn 3  write=3896  read=0
                                           turn 4  write=3926  read=0
```

Pure append caches perfectly and the read grows monotonically. Demotion misses
**100% of the time** and re-writes the whole prompt every turn — strictly worse
than not caching, since cache writes bill at a premium. The live 30-move game
confirms it: cache served 6% of input tokens, and every read after move 4 was 0.

The cause is that hydrogen sends `cache_control` as a single top-level request
field, which puts the only cache breakpoint at the end of the prompt. The one
stored entry is therefore the *whole* previous prompt, and demotion changes a
message in the middle of the next one — so the stored entry is no longer a
prefix and nothing matches.

**Fixed.** Divergence between turn *n* and turn *n+1* is exactly at the fat
carrier (it becomes a stub), so everything up to the message *before* the
carrier is stable across the pair. Put the breakpoint there and reads return,
writes collapse to ~31/turn, and the cached prefix grows monotonically.

This needed a hydrogen change: a single implicit top-level marker cannot
express placement under demotion. The current design places the breakpoint
automatically from the conversation's **volatile mark** (no public
`RequestOptions` cache field). Consumers call `push_volatile_user` /
`rotate_volatile_user`; Anthropic encoding is adapter-local.

Live 30-move game, before and after (prior breakpoint experiment; same
placement rule as today's volatile mark):

| | cache served | `cache_rd` at move 30 | uncached/turn |
|---|---|---|---|
| top-level marker only | 1–6% | 0 | grows to 21k |
| breakpoint before volatile tail | **65%** | 4332, growing monotonically | **flat, ~900–1300** |

The uncached column is the one that matters, and it is now flat — which is the
proposal's actual claim ("you reprocess ~1000 tokens per turn and hit cache on
everything older"). It was true in principle and false in implementation.

### 2. Context growth is dominated by assistant turns, not boards

The proposal projects `system + 250×50 + 900` ≈ 12k for a full game. Measured
over 15 model moves with reasoning on, growth was **427 tokens/move**, which
projects to ~107k for a 250-move game — better than the ~200k naive case, but
about 8x the estimate.

The demotion itself is working perfectly. Cumulative assistant output over the
run was 5723 tokens and total context growth was 5975 — i.e. **essentially all
growth is retained assistant turns**, and the user-side fat blocks contribute
approximately zero net. The 50-tokens-per-move figure assumed a terse assistant
turn; real ones with reasoning ran 80–1284 output tokens.

Demotion only ever controlled the user half of the transcript. The assistant
half grows unbounded, and with thinking enabled it is the larger term.

A prior experiment (`--keep-reasoning N`) dropped old `ReasoningBlock`s via
transcript rewrite. It cut raw growth (427 → 153 tok/move) but **froze the
prompt cache** when combined with breakpoints, because the strip rewrote
history under the cache write point. Effective cost was worse with strip+cache
than with cache alone. That flag is **removed**: prefer caching over strip
when cost is the binding constraint. Raw-context truncation needs a separate
hydrogen design if it returns.

### 3. The proposal's example code orphans a tool result

In `play_one_move`, the demotion step is:

```rust
if let Some(last) = conv.messages_mut().last_mut() {
    *last = thin_record(...);          // builds a plain Text message
}
```

After a successful move the transcript ends `[…, assistant(tool_use), user(tool_result)]`,
so `last_mut()` is the **tool result**, and replacing it with a text message
leaves the assistant's `tool_use` unanswered. Anthropic rejects that request.

Demotion has to preserve the block *kind* — a `tool_result` stays a
`tool_result`, carrying the stub as its output and keeping the original id.
That is `agent::demote_in_place`, and it is the thing
`demotion_keeps_a_tool_result_a_tool_result_with_the_same_id` pins.

### 4. In a tool loop, the model stops emitting text — and looks like it stopped thinking

Observed as "I only saw the model explain itself on its first move." Across a
33-move game, exactly **1 of 33** turns produced any assistant prose.

The model had not gone quiet. Turn 1's fat block arrives as a plain user
message, so the reply is conversational: `[thinking, text, tool_use]`. From
turn 2 the fat block rides in a `tool_result`, so the model is continuing a
tool loop and replies `[thinking, tool_use]` — no text block at all. Collecting
only `ContentBlock::Text` therefore captures the first turn and nothing after,
even though output ran 500–1700 tokens a turn.

This is a direct consequence of the "fat block rides in a `tool_result`" choice
in finding 1's design, and it is worth knowing before you conclude a model is
being terse. Two things to read instead, both already available:

- `ReasoningBlock::summary()` — hydrogen exposes it; it is populated on most turns.
- The `reasoning` argument of the `play_move` call itself, which the schema
  already requires and which was present on **15 of 15** turns after the fix.

The UI now prefers the tool argument, falls back to prose, then to the thinking
summary. All three go to `prompts.log`.

### 5. `Usage` dropped the cache fields (fixed in hydrogen)

Anthropic returns `cache_creation_input_tokens` and `cache_read_input_tokens`;
hydrogen parsed neither, so the consumer could not observe cache behavior — the
exact telemetry the proposal's cost argument depends on. Worse, with caching on,
`input_tokens` alone reports the *uncached remainder*: it reads as `2` against a
2074-token cached prefix, so it is actively misleading for context sizing.

Added both fields plus `Usage::total_input_tokens()`. Additive and non-breaking;
non-Anthropic adapters report 0.

## What the loop does

One fat block ever exists, always at the end. It rides inside a `tool_result`
whenever one is owed, making the whole game a single continuous tool loop. Once
answered it is demoted in place to a one-line stub and a freshly rendered fat
block is appended — content is replaced, messages are never removed.

The fat block is a pure function of `Game` state, rendered fresh every turn, so
the model's view cannot drift from the authoritative board. It carries the
things models read badly off a 2D grid — groups in atari, groups at ≤2
liberties, the ko point, capture counts, last 6 moves — computed on this side
and marked authoritative in the system prompt.

The environment owns the rules. Illegal proposals get a tool error naming the
specific reason and the model retries, capped at `--max-attempts` before a
forced pass. **Rejection rate over the validation runs was 0/32** — worth
watching, since it is the canary for the board render being subtly wrong (an
off-by-one row is the classic, which is why `render_places_stone_on_the_labeled_row`
and the I-skip coordinate tests exist).

## Known POC limitations

- **Simple ko only**, not positional superko — long cycles are not detected.
- **Area scoring has no dead-stone removal**, so the score is reported as
  provisional and is only meaningful once a game is genuinely played out.
- **The bot opponent is a heuristic**, not a Go player: it prefers contact moves
  so that fights actually happen and the atari/liberty half of the fat block
  gets exercised. It is a loop driver, not a sparring partner.
- **No retry collapse or reasoning strip** — both rewrote mid-transcript
  history and are out of scope until hydrogen has a deliberate API for them.
  Illegal-move retries keep their assistant/tool_result pairs in the log.
