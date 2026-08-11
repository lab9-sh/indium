# Proposal: volatile fat blocks and automatic prompt caching

Status: **proposed** (not implemented on `main`)
Motivating consumer: [indium](.) — human-vs-LLM Go (19×19, ~300 moves, one model turn per move)

This document supersedes the earlier plan of shipping raw `messages_mut` plus an
Anthropic-shaped `RequestOptions::cache_breakpoint_from_end` for the game-loop
case. Those two features were briefly implemented, validated live, then rolled
back because they overcomplicated hydrogen’s public surface and leaked
provider-specific cache semantics into consumers.

The refined goal is narrower:

1. Games can maintain a single **volatile fat block** (board / world state) that
  is demoted after each turn — without arbitrary transcript editing.
2. **Prompt caching is hydrogen policy**, always on by default, with breakpoint
  placement derived from that mark so consumers never place `cache_control`.

**Open consumer decision** (does not block the hydrogen concept): where the fat
block lives in the transcript. Two viable loop shapes are documented in
[§1b](#1b-two-consumer-loop-shapes). They share demotion + cache placement;
they differ in whether a `tool_result` ever needs to be marked volatile.

---



## Motivation



### The tail state block pattern

Long-running environment loops re-present the same logical state every turn.
Carrying a full board (or equivalent) on every historical turn is both wrong
(stale copies) and expensive (~700–900 tokens × hundreds of turns).

The pattern that works:

```
turn 47 sends:  [system, thin_1, a_1, …, thin_46, a_46, FAT_47]
                                                          ^ ~900 tok

turn 48 sends:  [system, thin_1, a_1, …, thin_46, a_46, thin_47, a_47, FAT_48]
                └──────────── unchanged, cache hit ──────┘ └── reprocessed ──┘
```

After the model responds, the fat block is **demoted in place** to a one-line
stub; a freshly rendered fat block is appended. Demotion replaces rather than
deletes, so the assistant turn above the block still answers something that
exists.

`FAT_N` may be either a user text message or a `tool_result` message depending
on the consumer loop shape ([§1b](#1b-two-consumer-loop-shapes)). The cache
story is the same either way: the write ends *before* that marked tail.

### What went wrong with the first API


| Shipped (then reverted)                         | Problem                                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `messages_mut` / free rewrite                   | Consumers can orphan tool results, strip mid-loop reasoning, rewrite under a cache prefix |
| `cache_breakpoint_from_end` on `RequestOptions` | Anthropic-only semantics on a shared struct; placement is a consumer concern              |
| Top-level Anthropic `cache_control` only        | Correct for pure append; **catastrophic** under demotion                                  |


Live indium measurements (30-move games):


|                                 | cache served | uncached / turn                      |
| ------------------------------- | ------------ | ------------------------------------ |
| top-level marker + demotion     | 1–6%         | grows to ~21k (full rewrite @ 1.25×) |
| breakpoint before volatile tail | **~65%**     | **flat ~900–1300**                   |


Demotion without correct placement is strictly worse than disabling caching:
every turn pays a premium cache write for a prefix that never hits.

Parallel tool calls and `tool_choice` were also tried as hydrogen knobs and
turned out to be non-issues or non-consumer concerns for this loop; they are
out of scope here.

---



## Summary


| #   | Feature                                                               | Where                   | Blocking?                                                                      |
| --- | --------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------ |
| 1   | [Volatile message slot + demotion](#1-volatile-message-slot)          | hydrogen `Conversation` | yes — pattern is impossible without some rewrite                               |
| 1b  | [Two consumer loop shapes](#1b-two-consumer-loop-shapes)              | indium (policy)         | open — pick before porting; drives which `push_volatile_*` surface is required |
| 2   | [Automatic stable-prefix caching](#2-automatic-stable-prefix-caching) | hydrogen adapters       | yes for Anthropic cost under demotion                                          |
| —   | Usage cache telemetry                                                 | hydrogen (done)         | already on main                                                                |


Consumers express **intent** (“this is environment state; I'll replace it next turn”).  
Hydrogen owns **wire encoding** (Anthropic breakpoints, OpenAI/xAI cache keys).  
Which message *kind* carries the fat block is consumer policy, not a provider concern.

---



## 1. Volatile message slot



### Problem

hydrogen’s `Conversation` is append-only. That is right for chat (carbon) and
wrong for environment loops. Opening `&mut Vec<Message>` fixes the latter by
abandoning the former: the conversation model stops being a closed lifecycle.

What games actually need is smaller than full mutability:

- Append a fat state block as the tail of a send.
- After the model has responded (and possibly after further assistant / tool turns)**,  
**rewrite only that block** into a short stub.
- Append a new fat block for the next environment step.

** Something to consider, if the agent wants more game state than provided in the FAT block,  
or if the agent wants to take actions, maybe the consumer should do this on a separate  
turn that doesn't contain any volatile blocks, or maybe this additional functionality should be  
deferred for a future hydrogen feature once we have games that ask for it. I'm leaning towards  
the simplest API for the narrowest use case. This addresses the subtlety below: FAT block would  
never be the last message, but always the second-to-last. The point of the API feature isn't  
"edit last message", it's "replace FAT user message behind last model response." I also don't  
think multiple assistant turns *after* a FAT block would be economical. Safe to assume the consumer  
wants to repalce FAT block with thin as soon as possible.

### Subtlety: volatile is not always `messages.last()`

```
deliver_state:  […, FAT]                 // fat is last; send happens here
push_response:  […, FAT, assistant]      // fat is second-to-last
next turn:      demote FAT → thin, append FAT'
                […, thin, assistant, FAT']
```

If the API only allowed “edit the last message,” demotion would hit the
assistant turn (wrong) or force demotion before `push_response` (breaks “the
model answered this user/tool_result”).

The right constraint is therefore:

> At most one **designated volatile** message, created as the tail of a send,
> replaceable later even after assistant turns are appended on top of it.

Stricter than arbitrary mutability; correct for the fat-block pattern.

### Proposed API

The full surface covers **both** loop shapes in [§1b](#1b-two-consumer-loop-shapes).
Shape B only needs the user half; shape A needs tool_result volatiles as well.
Ship either the full set, or user-only first if indium commits to B.

```rust
impl Conversation {
    /// Append a user text turn and mark it as the volatile state block.
    /// Errors if a volatile message already exists (use demote/rotate first).
    pub fn push_volatile_user(&mut self, text: impl Into<String>) -> Result<(), Error>;

    /// Append/coalesce a tool_result and mark that message as volatile.
    /// Same coalescing rules as `push_tool_result`.
    /// Required for shape A (fat-in-tool_result); unused by shape B.
    pub fn push_volatile_tool_result(
        &mut self,
        id: impl Into<String>,
        output: ToolOutput,
    ) -> Result<(), Error>;

    /// Replace the volatile message's content with a short stub, preserving
    /// block kind (tool_result keeps its id). Clears the volatile mark.
    /// Errors if there is no volatile message.
    pub fn demote_volatile(&mut self, stub: impl Into<String>) -> Result<(), Error>;

    /// Demote previous volatile (if any), then push a new text volatile.
    pub fn rotate_volatile_user(
        &mut self,
        stub: impl Into<String>,
        fat: impl Into<String>,
    );

    /// Demote previous volatile (if any), then push a new tool_result volatile.
    /// Required for shape A; unused by shape B.
    pub fn rotate_volatile_tool_result(
        &mut self,
        stub: impl Into<String>,
        id: impl Into<String>,
        fat: ToolOutput,
    );

    /// Index of the current volatile message, if any (read-only).
    pub fn volatile_index(&self) -> Option<usize>;
}
```

Internal state (sketch):

```rust
struct Conversation {
    messages: Vec<Message>,
    provider: Option<ProviderKind>,
    cache_key: String,
    /// Single slot: the only prior message consumers may rewrite.
    volatile: Option<usize>,
}
```

`rotate_*` is sugar over demote-then-push; the load-bearing primitives are
`push_volatile_*` and `demote_volatile`.

### What “volatile” means


| Concern         | Behavior                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Cache**       | On send, the cache write ends *before* the volatile message. The tail is never part of the written prefix.                     |
| **Retention**   | Not automatic. Fat must be visible for the turn. The consumer calls `demote_volatile(stub)` when the environment step is done. |
| **Mutation**    | Only `demote_volatile` may rewrite a prior turn — and only the marked message. No middle-history edits, no `messages_mut`.     |
| **Wire safety** | Demotion preserves kind: `ToolResult` → `ToolResult` (same id); text → text. The message is never deleted.                     |




### Invariants

1. **At most one volatile.** Bare `push_volatile_`* while one exists returns
  `Err`; `rotate_*` is the happy path for “stub previous, attach next.”
2. **Only volatile APIs set the mark.** `push_user`, `push_tool_result`, and
  `push_response` never do. Assistant turns cannot be marked volatile.
3. **Demote is content-only.** `messages.len()`, roles, and tool ids are
  unchanged.
4. **Non-volatile tool results stay append-only.** e.g. `update_notes` acks,
  illegal-move errors, and (under shape B) the thin `play_move` ack use
   normal `push_tool_result` and become stable prefix.
5. **If the mark is invalid** (future truncation APIs), clear it rather than
  rewrite a random index.



### Lessons from the indium POC (must preserve regardless of shape)

1. **Demoting the wrong block kind orphans tool results.** If a message is a
  `tool_result`, demotion must keep it a `tool_result` with the same id.
   Rewriting it as plain text leaves the preceding `tool_use` unanswered;
   Anthropic rejects the next request. (Shape A demotes `tool_result`s every
   turn; shape B demotes only user text, but kind preservation still matters
   if hydrogen ever marks a tool_result volatile.)
2. **Assistant turns dominate growth.** Demotion only controls the user/state
  half. With thinking on, measured growth was ~427 tok/move — almost all
   retained assistant output. Stripping old reasoning is a separate concern
   (see non-goals); combining it with a stable cache prefix freezes
   `cache_read` and loses on effective cost. Prefer caching over strip when
   cost is the binding constraint.



### Deliberately out of scope for this API


| Need                           | Status                                                          |
| ------------------------------ | --------------------------------------------------------------- |
| Strip old `ReasoningBlock`s    | No API here. Reach for it only if raw context length binds.     |
| Collapse failed tool attempts  | Agent-local for now.                                            |
| Edit message *N* in the middle | Forbidden.                                                      |
| Delete the volatile message    | Forbidden — breaks tool_use pairing. Demote only.               |
| Full `messages_mut`            | Rejected for the game case; reopen only with a separate design. |


---



## 1b. Two consumer loop shapes

The hydrogen volatile slot answers *how* to rewrite one marked tail. It does
**not** dictate *which* message carries the board. That is an indium (or
future game) policy choice. Both shapes below produce a stable cacheable
prefix plus one uncached fat tail; they differ in message count, tool-loop
continuity, and which hydrogen methods they need.

### Shared skeleton

```
send FAT (marked volatile)
  → model: assistant(tool_use play_move [+ maybe update_notes])
  → validate; answer non-move tools immediately (stable push_tool_result)
  → accept or reject the move
  → on accept: environment applies the move
next turn:
  → demote previous FAT → thin stub (in place)
  → deliver new FAT as the new volatile tail
```

Rejected moves and `update_notes` always use ordinary `push_tool_result` in
both shapes. Only the fat observation path diverges.

### Shape A — fat rides in `tool_result` (current indium POC)

**Idea:** leave a successful `play_move` unanswered; the *next* turn’s board
is the answer to that call. The game is one continuous tool loop after the
opening user message.

```
// first turn only
[…, FAT_user]                                    // volatile user
→ assistant(play_move id=T0)                     // left pending

// turn N≥2 send
[…, thin_{N-2}, assistant(T_{N-2}), FAT as tool_result(T_{N-2}),
    assistant(T_{N-1})]                          // T_{N-1} still pending
// after demote + rotate for turn N+1:
[…, thin_{N-2}, assistant(T_{N-2}), thin_{N-1},  // demoted tool_result, same id
    assistant(T_{N-1}), FAT' as tool_result(T_{N-1})]
```

**Consumer sketch** (matches today’s `deliver_state` branches):

```rust
fn deliver_state(&mut self, game: &Game, fat: String) {
    let stub = self.pending_stub_for.map(|n| self.stub_text(game, n));

    match (self.pending_tool_id.take(), stub) {
        (Some(id), Some(stub)) => {
            self.conv
                .rotate_volatile_tool_result(stub, id, ToolOutput::Text(fat));
        }
        (Some(id), None) => {
            self.conv
                .push_volatile_tool_result(id, ToolOutput::Text(fat))
                .unwrap();
        }
        (None, Some(stub)) => {
            self.conv.rotate_volatile_user(stub, fat);
        }
        (None, None) => {
            self.conv.push_volatile_user(fat).unwrap();
        }
    }
}
```

On accept: set `pending_tool_id = call.id` and **do not** push a tool_result
yet. On next `deliver_state`, that id carries the fat board.


|                                |                                                                                                                                                            |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hydrogen surface**           | Full: `push/rotate_volatile_user` **and** `push/rotate_volatile_tool_result`                                                                               |
| **Messages per accepted move** | 1 new user-role message (the fat `tool_result`) plus the assistant turn                                                                                    |
| **What is volatile**           | Opening: user text. Thereafter: the `play_move` tool_result                                                                                                |
| **What becomes stable**        | Demoted thin stubs (still `tool_result`s with the original ids), notes acks, error results                                                                 |
| **Pros**                       | Matches common agent-env packaging (“tool returns observation”); one fewer message per move; mid-game shape is pure tool loop (`tool_result` → `tool_use`) |
| **Cons**                       | Demotion must be kind-preserving on tool_results every turn; API surface larger; “ack” and “new state” are fused (harder to log a bare acceptance)         |
| **Status**                     | Implemented in indium today via `messages_mut` / `fat_carrier` / `pending_tool_id`                                                                         |




### Shape B — thin stable ack + volatile user fat

**Idea:** answer `play_move` immediately with a short stable result, demote the
previous fat user message, and always deliver the board as a **user** text
volatile. Tool results are never rewritten.

```
// turn N send
[…, thin_{N-1}, assistant(T_{N-1}), ack(T_{N-1}), FAT_user]   // only FAT volatile
→ assistant(play_move id=T_N)

// after accept, before / at next deliver_state:
demote FAT_user → thin_N
push_tool_result(T_N, "ok: Q16")               // stable, never demoted
push_volatile_user(FAT')                       // or rotate_volatile_user

// turn N+1 send
[…, thin_{N-1}, assistant(T_{N-1}), ack(T_{N-1}), thin_N,
    assistant(T_N), ack(T_N), FAT'_user]
└────────────── stable / cacheable prefix ──────────────┘ └── uncached ──┘
```

**Consumer sketch:**

```rust
fn deliver_state(&mut self, game: &Game, fat: String) {
    // If a play_move was accepted since last send, answer it with a thin ack
    // *before* rotating the fat (ordering: tool_use must be answered before
    // more user content on Anthropic).
    if let Some(id) = self.pending_tool_id.take() {
        let ack = self.pending_ack.take().unwrap_or_else(|| "ok".into());
        self.conv.push_tool_result(id, ToolOutput::Text(ack));
    }

    match self.pending_stub_for.take() {
        Some(n) => {
            let stub = self.stub_text(game, n);
            self.conv.rotate_volatile_user(stub, fat);
        }
        None => {
            self.conv.push_volatile_user(fat).unwrap();
        }
    }
}
```

On accept: record `pending_tool_id` **and** a short ack string (e.g. point
played); still return the move to the environment. Next `deliver_state`
pushes the stable ack, then rotates the user fat. (If the environment applies
the move before the next deliver, the stub can still name own-move + opponent
reply as today.)


|                                |                                                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hydrogen surface**           | User-only: `push_volatile_user`, `rotate_volatile_user`, `demote_volatile`                                                                                  |
| **Messages per accepted move** | 2 user-role messages (thin `tool_result` ack + fat user) plus the assistant turn                                                                            |
| **What is volatile**           | Always the fat user text message                                                                                                                            |
| **What becomes stable**        | Every `tool_result` (acks, errors, notes) and demoted thin user stubs                                                                                       |
| **Pros**                       | Smaller hydrogen API; tool_results stay append-only forever; easier mental model (“volatile = board user message”); ack is independently loggable/cacheable |
| **Cons**                       | Extra ~message and tokens per move; mid-game alternates tool_result and user text; slightly less “classic” continuous tool loop                             |
| **Status**                     | Not implemented; coherent with Anthropic pairing if ack is pushed before the new fat user                                                                   |




### Comparison


| Dimension                              | A: fat-in-`tool_result`              | B: thin ack + fat user                                   |
| -------------------------------------- | ------------------------------------ | -------------------------------------------------------- |
| Volatile kind after move 1             | `ToolResult`                         | `User` text                                              |
| `push_volatile_tool_result` needed?    | **yes**                              | **no**                                                   |
| Kind-preserving demotion load-bearing? | every turn                           | only if somehow marking tool_results (not in this shape) |
| Stable tool_results                    | notes, errors only                   | notes, errors, **and** play_move acks                    |
| Cache breakpoint                       | before marked volatile (either kind) | before marked volatile (user)                            |
| Measured POC                           | yes (current indium)                 | no live numbers yet                                      |
| Transcript density                     | denser                               | +1 user message / accepted move                          |


Caching does **not** prefer one shape: under correct placement, uncached input
is ~one fat block either way. The choice is packaging and API surface, not
hit rate.

### Investigation checklist (for a later decision)

Use this when choosing which shape indium ports to:

- [ ] **API minimalism:** is avoiding `*_volatile_tool_result` worth an extra
  ```
  message per move?
  ```
- [ ] **Model behavior:** does the pure tool loop (A) keep the model more
  ```
  reliably on `play_move` than a fat user turn (B)? Smoke both for
  no-tool-call rate.
  ```
- [ ] **Token delta:** measure thin ack size vs any change in fat prompt
  ```
  wording; confirm uncached/turn stays flat for both.
  ```
- [ ] **Logging / SGF / prompts.log:** is a distinct “accepted Q16” tool_result
  ```
  useful, or is the fat board enough?
  ```
- [ ] **Multi-tool responses:** both shapes already handle `update_notes` +
  ```
  `play_move` in one assistant turn; confirm B’s ack ordering stays valid
  when notes results are pushed first.
  ```
- [ ] **Forced pass / game over:** A answers a pending id with fat or
  ```
  `"game over"`; B answers with a thin ack then may push a final user
  message — spell out `finish()` for each.
  ```
- [ ] **Hydrogen ship strategy:** full volatile API (A+B ready) vs user-only
  ```
  first (B only). Revisit if a second consumer needs A.
  ```

Until that decision, the hydrogen design below assumes the **full** surface so
neither shape is blocked. Implementation may still stage user-only first.

---



## 2. Automatic stable-prefix caching



### Problem

Providers differ:


| Provider                                      | Mechanism                                                                                                              | Consumer placement?                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| **OpenAI / xAI**                              | Automatic prefix match + `prompt_cache_key`                                                                            | No — hydrogen already sends the key |
| **Anthropic (top-level** `cache_control`**)** | Writes a cache entry at the **last** cacheable block                                                                   | Implicit; wrong under demotion      |
| **Anthropic (block-level)**                   | Up to 4 explicit breakpoints; writes **only** at those points; reads walk back ≤20 blocks looking for prior **writes** | Yes, unless hydrogen does it        |


Anthropic lookback does **not** discover stable content behind a changing tail;
it only finds entries earlier requests already wrote. Top-level automatic
caching is perfect for pure append and actively hostile to demotion: the
previous write included the fat block, demotion rewrites it, the stored prefix
is gone, and every turn pays a 1.25× full rewrite.

xAI/OpenAI never needed a placement knob: they match the longest identical
prefix server-side. Demotion “just works.”

### Portable concept

Consumers care about:

> Keep a growing stable prefix byte-identical when you can; put volatility at
> the marked tail.

They must **not** care about:

- `cache_control` / ephemeral / TTL
- `cache_breakpoint_from_end`
- which adapter places markers
- whether the volatile tail is user text or a `tool_result` (shape A vs B)

Hydrogen’s job on every `send`:

1. **Always cache by default** (desired for Anthropic even at 1.25× writes).
2. **Anthropic adapter:** place breakpoints so the write lands on the last
  block of the stable prefix.
3. **OpenAI/xAI adapters:** no-op for placement; keep `prompt_cache_key`.
4. **Telemetry stays portable** (`cache_read` / `cache_creation` /
  `total_input_tokens`).



### Placement rules (Anthropic)

When `volatile_index` is `Some(i)` and `i > 0`:

1. Attach `cache_control: { type: ephemeral }` on the last content block of
  message `i - 1`.
2. Omit top-level automatic `cache_control` (keeping both would re-anchor the
  write on the volatile tail).

When there is no volatile mark (carbon, plain chat):

- Keep always-on caching via top-level `cache_control`, **or** treat the last
message as volatile by heuristic (`from_end = 1`). Either is fine for
append-only; the volatile mark makes the game case precise.

```
messages:  [ stable… | VOLATILE ]
cache:       ^^^^^^^^^^ write ends here
                       ^^^^^^^^ uncached input

after demote + new volatile:
[ stable… | thin | assistant… | VOLATILE' ]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^  hit + advance write
                                ^^^^^^^^^ uncached
```

No public `RequestOptions` cache fields. The mark *is* the portable signal;
Anthropic is one encoder.

### Optional upgrade: longest common prefix

A more general Anthropic strategy (no consumer API): store a fingerprint of
messages as last successfully sent (keyed by `cache_key`), place the breakpoint
at the end of the longest common prefix with the previous send, update the
snapshot only after a successful response starts.


| Pattern                    | LCP behavior                                   |
| -------------------------- | ---------------------------------------------- |
| Pure append                | Common prefix = everything but the new tail    |
| Volatile demotion          | Common prefix stops before the demoted carrier |
| Reasoning strip / collapse | Common prefix freezes at the rewrite (correct) |


With a volatile slot as the only rewrite path, the simpler “breakpoint before
`volatile`” rule is sufficient and easier to reason about. LCP remains a
possible internal upgrade if mid-transcript rewrites are ever added.

### Cost model

With correct placement on Anthropic:

- **Write (1.25×)** on the new stable span as the write point advances
- **Read (0.1×)** on the growing prefix
- **Uncached (1.0×)** on the volatile tail (~fat block)

Broken top-level + demotion is the only regime where “always cache” is a bad
default. Correct placement makes always-on strictly better than off for
multi-turn agents.

Shape A vs B does not change this model in the large: both leave ~one fat
block uncached. Shape B adds a thin stable ack into the prefix (cheap write
once, then reads).

### What not to do


| Idea                                     | Problem                                              |
| ---------------------------------------- | ---------------------------------------------------- |
| Public `cache_breakpoint_from_end`       | Anthropic leak on a shared struct                    |
| Top-level auto only under demotion       | Negative cache value (1.25× full rewrite every turn) |
| Breakpoint on every historical turn      | Max **4** breakpoints; does not scale                |
| Consumer-set block-level `cache_control` | Re-introduces provider leakage                       |


“Breakpoint per turn” only means: one well-placed write on the stable prefix,
advanced as that prefix grows — not one marker per message kept forever in the
request.

---



## Impact on “one conversation model”


| Promise                                    | Effect                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| One shared type across providers           | **intact** — still `Conversation` / `Message` / `ContentBlock`           |
| One `Client::send(conv, opts)` entry point | **intact**                                                               |
| Hydrogen owns multi-turn invariants        | **mostly intact** — single demote path, kind-preserving, no free rewrite |
| Append-only for chat consumers             | **intact** — volatile APIs are opt-in; carbon never calls them           |
| Provider-agnostic request options          | **intact** — no cache placement fields                                   |


This is the narrow demotion API preferred in hydrogen’s agent-loop proposal,
named for the portable concept (volatile state) rather than the Anthropic wire
detail (breakpoint index). Consumer loop shape (A vs B) stays outside
hydrogen.

---



## Non-goals

- **A built-in truncation or summarization helper.** Policy belongs to the
consumer; hydrogen only needs to stop preventing demotion and to place
caches correctly.
- **Picking shape A vs B inside hydrogen.** That is indium policy
([§1b](#1b-two-consumer-loop-shapes)); hydrogen only supplies the slot.
- **Making** `ReasoningBlock` **constructible.** Payloads are provider-signed.
- **Relaxing provider pinning.** Cross-provider transcript reuse remains a
hazard.
- **A local tokenizer.** Three providers, three vocabularies.
- **Usage accounting on** `Conversation`**.** Per-`Response` `Usage` is enough.
- **Public tool_choice / parallel_tool_calls** unless a later consumer proves
they are load-bearing (indium did not).
- **A second conversation type** (“chat” vs “agent”). One type, closed
lifecycle with one optional volatile slot.

---



## Implementation order

1. **Volatile slot on** `Conversation` — at minimum user half
  (`push_volatile_user`, `demote_volatile`, `rotate_volatile_user`,
   `volatile_index`); add `*_volatile_tool_result` unless indium has already
   committed to shape B only. Unit tests: single-slot enforcement;
   kind-preserving demotion for whichever kinds are supported.
2. **Anthropic adapter:** breakpoint immediately before `volatile` when set;
  always-on caching otherwise; no `RequestOptions` changes. Independent of
   fat kind.
3. **Confirm OpenAI/xAI** still send `prompt_cache_key`; no placement work.
4. **Decide shape A vs B** for indium using the [investigation
  checklist](#investigation-checklist-for-a-later-decision). Optional: short
   dual smoke (10–30 moves each) comparing no-tool-call rate and uncached/turn.
5. **Port indium** off `messages_mut` / `fat_carrier` /
  `cache_breakpoint_from_end` onto the matching `rotate_volatile_*` path.
6. **Live check:** 30-move game — `cache_read` grows monotonically; uncached
  per turn stays flat; no orphaned `tool_use` (A: demoted tool_results keep
   ids; B: every accepted move has a thin ack before the next fat user).

Ship (1) and (2) together. Demotion without placement is a cost footgun;
placement without demotion is unused for games. Step (4) can precede or follow
(1) if the goal is only to pick a loop shape; it is not a hydrogen blocker.

---



## Acceptance criteria



### Hydrogen (either consumer shape)

- [ ] At most one volatile message; second `push_volatile_*` without demote fails
- [ ] `demote_volatile` is content-only (length, roles, tool ids unchanged)
- [ ] Anthropic: with volatile fat + demotion, cache read grows monotonically
  ```
  over a 30-move game and uncached tokens stay roughly flat
  ```
- [ ] Anthropic: pure append (no volatile) still caches (carbon path)
- [ ] OpenAI / xAI: no regressions; `prompt_cache_key` unchanged
- [ ] Carbon and other append-only consumers compile without API churn on
  ```
  `RequestOptions`
  ```



### Shape-dependent (indium port)

- [ ] Main loop needs no `messages_mut` and no provider-specific cache knobs
- [ ] **If shape A:** demoting a volatile `tool_result` keeps the same id and
  ```
  block kind; `push/rotate_volatile_tool_result` exist and are used
  ```
- [ ] **If shape B:** every accepted `play_move` gets a stable thin
  ```
  `push_tool_result` before the next fat user; only
  `push/rotate_volatile_user` are used for the board
  ```
- [ ] Decision recorded (comment or short note in indium) so the unused shape
  ```
  remains documented here rather than rediscovered
  ```

---



## Reference: measured outcome (prior breakpoint experiment)

Same pattern, prior API (`cache_breakpoint_from_end: Some(1)`), 30-move games.
POC used **shape A** (fat-in-`tool_result`); numbers should transfer to shape B
for cache hit rate, with a small extra stable prefix from thin acks.


|                                  | cache served | `cache_rd` at move 30 | uncached / turn    |
| -------------------------------- | ------------ | --------------------- | ------------------ |
| top-level marker only + demotion | 1–6%         | 0                     | grows to ~21k      |
| breakpoint before volatile tail  | **~65%**     | 4332, growing         | **flat ~900–1300** |


The volatile-slot design targets the same wire behavior without exposing that
knob to consumers.