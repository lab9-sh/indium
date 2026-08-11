# Implementation plan: volatile fat blocks

Status: **ready to implement**  
Depends on: [PROPOSAL-volatile-fat-blocks.md](./PROPOSAL-volatile-fat-blocks.md)  
Repos: hydrogen (library) + indium (consumer port)  
Non-breakage targets: carbon, silicon (append-only; must compile and behave unchanged)

---

## Decision record

### Selected approach

| Layer | Choice | Rationale |
| --- | --- | --- |
| **Indium loop shape** | **Shape B** — thin stable `play_move` ack + volatile **user** fat | Matches the narrow-use-case lean in the proposal `**` note; smaller hydrogen surface; tool_results stay append-only forever |
| **Hydrogen volatile API** | **User-only first** | `push_volatile_user`, `demote_volatile`, `rotate_volatile_user`, `volatile_index`. Defer `*_volatile_tool_result` until a second consumer needs Shape A |
| **Caching** | **Automatic from the mark** | Anthropic adapter places block-level breakpoint on the last content block of `messages[volatile_index - 1]`; omit top-level `cache_control` when a volatile is set. OpenAI/xAI unchanged (`prompt_cache_key`) |
| **Out of scope for this pass** | Reasoning strip, collapse-retries, multi-volatile, mid-transcript edits, LCP fingerprinting | Need free rewrite or a separate design; proposal already excludes them |

### Why Shape B over Shape A (validation of the `**` note)

The proposal `**` note argues for the simplest API for the narrowest game loop:

> The point of the API feature isn't "edit last message", it's "replace FAT user message behind last model response." … Safe to assume the consumer wants to replace FAT block with thin as soon as possible.

That maps cleanly onto Shape B:

1. **Fat is always a user text message** — demotion is content-only text rewrite; no kind-preserving tool_result path on the hot path.
2. **Demotion is never “edit last”** — after `push_response`, fat sits under the assistant (and any subsequent acks / error results). The load-bearing primitive is a **marked index**, not `messages.last_mut()`.
3. **Tool loop hygiene stays append-only** — accepted moves get a thin stable `push_tool_result` *before* the next fat user is rotated in. Anthropic pairing stays valid without rewriting prior tool_results.
4. **API surface matches intent** — hydrogen need not ship `push_volatile_tool_result` for indium to work. Shape A’s larger surface is justified only if we keep “fat rides in the pending tool_result.”

Shape A (current broken indium) is coherent and denser by one message per move, but it forces:

- kind-preserving demotion of `tool_result` every turn,
- dual volatile push paths,
- and the historical failure mode (demoting a tool_result into plain text orphans `tool_use`).

Caching does **not** prefer A: measured ~65% cache serve / flat ~900–1300 uncached per turn came from **breakpoint-before-tail**, not from fat kind. Shape B adds a thin ack into the stable prefix (cheap once, then cache reads).

### Nuance on “always second-to-last”

On the happy path after one `send` + `push_response`, fat is second-to-last. Under **rejection retries** inside `take_turn`, the transcript can be:

```
[…, FAT, asst(bad), error, asst(bad2), error, asst(ok)]
```

So demotion must use **`volatile_index`**, not “message at `len-2`.” The mark generalizes the mental model without widening the API. Multiple *accepted* environment steps stacked above one fat without demoting remains out of scope (uneconomical; demote ASAP at next `deliver_state`).

### Deferral: “agent wants more state / actions”

Agree with the proposal: if a future game needs mid-turn queries or extra actions against a non-demoted fat, either:

- run those as a **separate send with no volatile mark**, or
- design a follow-up hydrogen feature once a real consumer exists.

Do not generalize the volatile slot for that now.

---

## Current ground truth

### hydrogen (`main`)

- `Conversation` is **append-only**: `push_user`, `push_tool_result` (coalescing), `push_response`, `messages()`, pin + `cache_key`.
- **No** `messages_mut`, `push_message`, `cache_breakpoint_from_end`.
- Anthropic adapter always sets **top-level** `cache_control: ephemeral` and has **no** block-level placement.
- OpenAI/xAI send `prompt_cache_key`; usage telemetry for cache read/create already works.
- Carbon / silicon use only append APIs → **safe if volatile is opt-in additive**.

### indium (broken vs current hydrogen)

Compiles against a **reverted** surface. Failures include:

| Call site | Missing API |
| --- | --- |
| `RequestOptions { cache_breakpoint_from_end: Some(1), … }` | field removed |
| `conv.messages_mut()` (strip, demote, collapse) | never on main |
| `conv.push_message(...)` | never on main |
| `TextBlock::new` / `ToolResultBlock::new` in agent + tests | public field structs; no `new` on main |

Runtime shape today is **Shape A** (fat in pending `play_move` tool_result, `fat_carrier` index, local `demote_in_place`).

### carbon / silicon

- Pure append + tools; no demotion.
- Must keep compiling with **zero** required API changes.
- Anthropic always-on top-level cache must remain the default when `volatile_index` is `None`.

---

## Target design (locked)

### Hydrogen public surface (this ship)

```rust
impl Conversation {
    pub fn push_volatile_user(&mut self, text: impl Into<String>) -> Result<(), Error>;
    pub fn demote_volatile(&mut self, stub: impl Into<String>) -> Result<(), Error>;
    pub fn rotate_volatile_user(&mut self, stub: impl Into<String>, fat: impl Into<String>);
    pub fn volatile_index(&self) -> Option<usize>;
}
```

Internal:

```rust
struct Conversation {
    messages: Vec<Message>,
    provider: Option<ProviderKind>,
    cache_key: String,
    volatile: Option<usize>,  // not serialized as consumer-editable; see serde note
}
```

**Invariants**

1. At most one volatile; bare `push_volatile_user` while mark set → `Err(InvalidRequest)`.
2. Only volatile APIs set the mark; `push_user` / `push_tool_result` / `push_response` never do.
3. `demote_volatile` rewrites **content only** (text → short text); clears mark; never deletes; never changes `messages.len()` or roles.
4. `rotate_volatile_user` = demote-if-present + `push_volatile_user` (infallible sugar: demote missing mark is a no-op or only-push path — implement as demote-if-any then push, panicking only on push double-mark which cannot happen after clear).
5. If index is ever invalid (future truncation), clear mark rather than rewrite a random message.

**Serde:** prefer `#[serde(default, skip_serializing)]` on `volatile` so on-disk carbon sessions stay compatible. After load, mark is absent (correct: resumed chat is append-only). Do **not** persist a rewrite-capable index across process boundaries unless a later design needs it.

**Errors:** reuse `Error::InvalidRequest` for double-push / demote-without-mark (if demote is fallible). Keep `rotate_*` ergonomic for the game loop.

**Not shipping now:** `push_volatile_tool_result`, `rotate_volatile_tool_result`, public `messages_mut`, any `RequestOptions` cache field.

### Anthropic placement

On every `build_request` / send:

| Condition | Behavior |
| --- | --- |
| `volatile_index() = Some(i)` and `i > 0` | Attach `cache_control: { "type": "ephemeral" }` on the **last content block** of message `i - 1`. **Omit** top-level request `cache_control` (or make it optional/`skip_serializing_if`). |
| `volatile_index() = Some(0)` | No prior stable message; no block breakpoint; optional top-level or none — first turn edge case. |
| `volatile_index() = None` | Today’s always-on top-level `cache_control` (carbon/silicon path). |

Wire change: `MessagesRequest.cache_control` becomes `Option` (or always skip when placing block-level). `block_to_wire` / message encoding gains a path to inject one ephemeral marker on a chosen block without polluting portable `ContentBlock` types (adapter-local only).

### Indium Shape B loop

```
deliver_state:
  if pending_tool_id:
      push_tool_result(id, thin_ack)     // stable forever
  if had previous fat (pending_stub_for):
      rotate_volatile_user(stub, fat)
  else:
      push_volatile_user(fat)

take_turn:
  deliver_state → send → push_response
  for each tool_use:
    update_notes → push_tool_result("notes updated")   // stable
    play_move illegal → push_tool_result(Error(...))   // stable, retry
    play_move ok → record pending_tool_id + pending_ack; do NOT push result yet
  return move to environment

finish:
  if pending_tool_id: push_tool_result("game over")
  // optional: demote_volatile final stub so saved transcript is thin
```

Remove: `fat_carrier`, local `demote_in_place` as the primary path, `cache_breakpoint_from_end`, reliance on `messages_mut` for the main loop.

**Stub text:** keep `thin_stub(number, own, reply)` as today.

**Ack text:** e.g. `ok: Q16` / `ok: pass` from the accepted move (store on accept as `pending_ack: Option<String>`).

### Features gated / deferred in indium

| Feature | This ship | Notes |
| --- | --- | --- |
| Fat demotion + cache | **yes** | core |
| `update_notes` | **yes** | ordinary `push_tool_result` |
| Rejection retries | **yes** | append error results; fat stays marked until next deliver |
| `--collapse-retries` | **removed** | needed mid-history drain |
| `--keep-reasoning N` | **removed** | needed rewrite of prior assistants; froze cache under breakpoints |

Strip/collapse remain non-goals until a separate hydrogen design.

---

## Work packages

### WP0 — Prep (indium decision, docs)

- [x] Select Shape B + user-only hydrogen API (this document).
- [ ] Add a short pointer at the top of `PROPOSAL-volatile-fat-blocks.md`: “Indium commits to Shape B; see `PLAN-volatile-fat-blocks.md`.”
- [ ] Update indium `README.md` “What is being validated” table after the port (do not leave references to `messages_mut` / `cache_breakpoint_from_end` as current truth).

### WP1 — Hydrogen: volatile slot (unit-tested)

**Files:** `hydrogen/src/types/conversation.rs`, `hydrogen/src/error.rs` (if needed), `hydrogen/src/lib.rs` re-exports (no change if methods are on `Conversation`).

1. Add `volatile: Option<usize>` with serde skip/default.
2. Implement `push_volatile_user` — same message shape as `push_user`, set mark to `messages.len() - 1`, error if mark already set.
3. Implement `demote_volatile(stub)`:
   - require mark;
   - require message is user text (for user-only ship: if somehow wrong kind, `InvalidRequest` rather than silently corrupt);
   - replace content with single `TextBlock { text: stub, extras: None }`;
   - clear mark.
4. Implement `rotate_volatile_user(stub, fat)`:
   - if mark: demote with stub; else ignore stub (or require stub only when mark present — prefer demote-if-any);
   - `push_volatile_user(fat).expect(...)` or map error.
5. `volatile_index()` getter.
6. **Tests:**
   - single-slot enforcement (second push without demote fails);
   - demote is content-only (len, roles unchanged);
   - rotate after `push_response` rewrites the marked user under the assistant, not the assistant;
   - after rotate, new mark is the new tail;
   - `push_user` does not set mark;
   - cache_key unchanged by demote/rotate.

### WP2 — Hydrogen: Anthropic automatic placement

**Files:** `hydrogen/src/anthropic/adapter.rs`, `hydrogen/src/anthropic/wire.rs`, adapter unit tests.

1. Make top-level `cache_control` optional on the wire request.
2. When building messages, if `conv.volatile_index()` is `Some(i)` and `i > 0`, clone/map messages to wire and attach ephemeral `cache_control` to the last JSON content object of message `i - 1`.
3. When volatile is set, do **not** send top-level `cache_control`.
4. When volatile is unset, keep current top-level always-on behavior.
5. **Tests (wire-level, no network):**
   - no volatile → top-level present, no block-level markers;
   - volatile at last index → marker on previous message’s last block, no top-level;
   - volatile at 0 → no panic; defined edge behavior;
   - marker never attached to the volatile message itself.

### WP3 — Hydrogen: OpenAI / xAI smoke

- [ ] Confirm `prompt_cache_key` still sent from `conv.cache_key()` (no code change expected).
- [ ] Existing unit tests still pass; add nothing unless a regression appears.

### WP4 — Indium port to Shape B

**Files:** `indium/src/agent.rs`, `indium/src/main.rs` (stats banner), tests, README.

1. Drop `cache_breakpoint_from_end` from `RequestOptions`.
2. Replace `fat_carrier` + `demote_in_place` + `push_message` with:
   - `pending_tool_id`, `pending_ack`, `pending_stub_for`;
   - `deliver_state` as in target design (ack first, then `rotate_volatile_user` / `push_volatile_user`).
3. On accepted `play_move`: set `pending_tool_id`, `pending_ack` (point string), `pending_stub_for`; do not push tool_result until next deliver / finish.
4. `finish()`: answer pending with `"game over"`; optionally demote last fat for clean logs.
5. ~~Remove `strip_old_reasoning` / `collapse` and CLI flags~~ **done** (`--keep-reasoning`, `--collapse-retries` removed).
6. **Unit tests:**
   - rewrite demotion tests against `Conversation` volatile APIs (user-text path);
   - drop tool_result-kind demotion as a required indium test (hydrogen may add it later with Shape A);
   - optional: pure-logic test that `deliver_state` ordering is ack-then-fat (mock Conversation via public API).
7. Update main.rs cache banner text to “hydrogen automatic stable-prefix caching (volatile mark)” instead of `cache_breakpoint_from_end=1`.

### WP5 — Live validation

| Check | Pass criteria |
| --- | --- |
| Anthropic 30-move bot game | `cache_read` grows over the run; uncached/turn roughly flat (~900–1300 band from prior exp); no API rejections for orphaned tool_use |
| xAI short smoke | completes; no new errors; cache telemetry still populated if provider reports it |
| carbon / silicon | `cargo check` / existing tests green against path dep or published hydrogen revision |
| indium unit tests | all green without private hydrogen APIs |

Optional A/B (not blocking): 10–15 move Shape B only is enough; dual Shape A smoke only if reconsidering the decision.

### WP6 — Docs cleanup

- [ ] indium README: rewrite findings table for post-port world; keep historical measurements as history.
- [ ] hydrogen README (short): mention opt-in volatile slot + automatic Anthropic placement for environment loops.
- [ ] Proposal status line → “accepted / implementing” or link to this plan.

---

## Suggested PR sequence

Implement as a **Graphite-style stack** (or sequential PRs) so carbon/silicon never see a half-broken intermediate:

| PR | Repo | Contents | Risk |
| --- | --- | --- | --- |
| **H1** | hydrogen | WP1 volatile slot + tests | Additive; zero behavior change for existing callers |
| **H2** | hydrogen | WP2 Anthropic placement + WP3 confirm | Behavior change only when mark set; default path must match today |
| **I1** | indium | WP4 port + unit tests | Unblocks compile; needs H1+H2 |
| **I2** | indium | WP5 live run notes + WP6 docs | After green live check |

Ship H1+H2 together before depending on them from indium main. Do **not** land demotion-capable indium against top-level-only Anthropic caching (cost footgun from the proposal).

---

## Explicit non-goals (this plan)

- Public `messages_mut` / free transcript rewrite
- `RequestOptions::cache_breakpoint_from_end` (or any consumer-facing cache placement)
- Shape A / `push_volatile_tool_result`
- Automatic reasoning strip or retry collapse in hydrogen
- LCP fingerprint cache placement
- Changing carbon session format beyond harmless ignored serde fields
- Making `ReasoningBlock` constructible

---

## Acceptance checklist (copy into PR)

### Hydrogen

- [ ] At most one volatile; second `push_volatile_user` without demote fails
- [ ] `demote_volatile` content-only; clears mark
- [ ] Anthropic: volatile set ⇒ block breakpoint on message before volatile, no top-level marker
- [ ] Anthropic: no volatile ⇒ top-level always-on unchanged
- [ ] OpenAI/xAI: `prompt_cache_key` unchanged
- [ ] carbon + silicon compile against new hydrogen without source changes

### Indium

- [ ] Compiles and unit tests pass on path-dep hydrogen
- [ ] No `messages_mut` / `cache_breakpoint_*` / local fat kind demotion in the main loop
- [ ] Shape B: every accepted move eventually answered with thin stable tool_result before next fat user
- [ ] Live 30-move: cache read grows; uncached/turn flat; zero orphaned tool_use errors
- [ ] Decision (Shape B) recorded here and linked from the proposal

---

## Open questions (non-blocking)

1. **`rotate_volatile_user` signature:** infallible (ignore demote-without-mark) vs `Result` — prefer infallible sugar for the game loop.
2. **Final demote on `finish`:** nice for logs; not required for API correctness.
3. **When to add Shape A surface:** only if a second environment consumer proves fat-in-tool_result is load-bearing for model behavior (investigation checklist in the proposal).

---

## Summary

**Validate the lean in the `**` note:** yes — implement the narrow demotion API (“replace marked fat user under later assistant turns”), demote ASAP at the next environment step, and do not design for multi-step agent work stacked on an undemoted fat.

**Indium:** Shape B.  
**Hydrogen:** user-only volatile slot + automatic Anthropic breakpoint before that mark.  
**carbon / silicon:** untouched call paths; default caching unchanged.
