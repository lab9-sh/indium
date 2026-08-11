//! Indium — a human-vs-LLM Go POC over [`hydrogen`].
//!
//! The point of the crate is to exercise hydrogen's agent-loop surface under a
//! long environment loop: transcript rewriting (the tail state block) and
//! explicit cache breakpoints. The Go rules are authoritative and live entirely
//! on this side; the model only proposes coordinates.

pub mod agent;
pub mod goban;
pub mod prompt;
pub mod record;
