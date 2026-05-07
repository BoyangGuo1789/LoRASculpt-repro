# Agent Work

This directory stores durable coordination state for Codex and other agentic coding sessions.

- `TASK_STATE.md` records the current state machine and next transition.
- `logs/` contains one handoff log per repository-working turn.
- `plans/` contains detailed plans for large work.
- `handoffs/` contains worker, reviewer, or external-oracle handoff documents.
- `templates/` contains reusable Markdown templates.

The purpose is continuity: a new model should be able to continue from these files without depending on a long chat thread.
