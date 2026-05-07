# AGENTS.md

# Repository Agent Operating Model

This repository is the durable source of truth for agent work on LoRASculpt. Do not rely on a separate local `Agent.md` for project state, plans, or handoff history.

## 1. Mandatory Repository Artifacts

All durable agent coordination artifacts must live under `docs/agent_work/`:

- `TASK_STATE.md`: current task state machine, active task, blockers, latest evidence, and next transition.
- `logs/`: one handoff log per Codex repository-working turn.
- `plans/`: detailed Markdown plans for large work before execution starts.
- `handoffs/`: worker, reviewer, or external-oracle handoffs.
- `templates/`: reusable templates for logs and plans.

Do not write turn-by-turn logs into this file. `AGENTS.md` only defines the workflow.

## 2. Every-Turn Handoff Log

For every turn that inspects, edits, runs, or tests this repository, create or update a dated log file before the final user response:

```text
docs/agent_work/logs/YYYY-MM-DD-<short-task-slug>.md
```

Each handoff log must include:

- goal and user request,
- starting state and relevant context,
- actions taken,
- files changed or intentionally left untouched,
- commands run and observable results,
- failures or blocked attempts,
- current task state,
- remaining risks,
- recommended next action.

Logs must be concise but enough for a new model to continue without reading the whole conversation. Never record secrets, credentials, API keys, tokens, `.env` values, private keys, or unrelated personal data.

## 3. Detailed Plans For Large Work

Before any large task, write a detailed plan in:

```text
docs/agent_work/plans/YYYY-MM-DD-<short-task-slug>.md
```

A task is large if it touches multiple subsystems, changes training/evaluation behavior, may require long runs, involves architecture choices, or is likely to outlive one conversation.

The plan must include:

- goal and definition of done,
- assumptions and constraints,
- relevant files and systems,
- proposed state transitions,
- execution steps,
- verification strategy,
- expected commands or experiments,
- risks and rollback path,
- ownership split if a worker is used.

Keep the plan updated as execution progresses. A stalled or compressed conversation should be recoverable from the latest plan plus `TASK_STATE.md` and logs.

## 4. Controller / Worker Split For Large Work

For large work, keep the main conversation as the controller:

- The controller owns architecture, plan, state transitions, review, and final summary.
- Execution should happen in a worker or subagent when available and allowed by current tool policy.
- The worker receives the plan, scoped file ownership, verification requirements, and the instruction to write a handoff under `docs/agent_work/handoffs/` or update the task log.
- The main conversation should retain only phase summaries, key decisions, evidence, and final review.

If subagents are unavailable, emulate the split explicitly: keep a planning phase, an execution phase, a verification phase, and a final review phase, and write the same repository artifacts.

## 5. Task State Machine

Manage project progress as state transitions, not as chat history. Keep `docs/agent_work/TASK_STATE.md` current.

Default states:

```text
BACKLOG -> PLANNING -> READY_FOR_WORKER -> EXECUTING -> VERIFYING -> REVIEW -> DONE
                                      \-> BLOCKED
```

Update `TASK_STATE.md` after meaningful changes in:

- active goal,
- owner/controller/worker split,
- current state,
- latest evidence,
- blockers,
- next action,
- verification status.

The latest state file should answer: what is true now, what changed last, what must happen next, and what should not be touched.

## 6. Recovery From Stalled Conversations

If a conversation freezes, compacts badly, or cannot continue:

1. Start a new conversation.
2. Provide the stalled conversation id and the last few useful messages if available.
3. Tell the new model to read `AGENTS.md`, `docs/agent_work/TASK_STATE.md`, the latest plan, and the latest log.
4. Continue from the latest verified state, not from memory.

If the UI supports branching from an earlier message, branch before the failure point and rerun from the latest repository state.

## 7. Verification Standard

Do not claim completion without evidence. Use the smallest relevant verification first.

For this ML repository, prefer this order before expensive runs:

- import/config parse checks,
- dataloader smoke checks,
- tensor shape and dtype checks,
- single forward pass,
- loss/gradient finite checks,
- one-batch or tiny-run smoke test,
- only then longer training or evaluation.

Record verification commands and results in the handoff log.

## 8. Version Commit / GitHub Traceability

After each completed and verified version-level code change, create one Git commit so the project has a clear rollback and traceability point.

Rules:

- A version-level code change means source-code behavior changed in a coherent, reviewable unit.
- Parameter-only changes do not trigger this mandatory commit rule. Examples include changing shell-script arguments, config values, dataset paths, seeds, hyperparameters, or experiment names without changing executable source behavior.
- Before committing, run the smallest relevant verification and record the evidence in the session log.
- Stage only files that belong to the completed version. Do not stage unrelated user changes or dirty files from another task.
- Include the related `docs/agent_work/` plan, state, and log updates in the same commit when they document that version.
- Use a concise commit message that names the behavior changed, not just the files touched.
- If pushing to GitHub is explicitly authorized for the current task, push the commit after creating it. If push authorization is absent, leave the commit local and report that the GitHub push is pending approval.

This rule is for recovery and auditability: future agents should be able to inspect the commit history to find each completed code version and roll back cleanly when needed.

## 9. Safety And Patch Discipline

- Inspect relevant files before editing.
- Keep diffs focused.
- Do not overwrite unrelated user changes.
- Do not run destructive commands, long/expensive training, deployment, credential, billing, or account actions without explicit approval.
- Do not expose secrets in logs, plans, prompts, or handoffs.
- Prefer deterministic CLI commands and repository evidence over confidence.
