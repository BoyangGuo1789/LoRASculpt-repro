# Baseline-First Research Constraint

## Goal

The user rejected the task-aware LoRA-off routing story as too low-innovation for paper submission and requested a durable agent rule: return to the LoRASculpt baseline training paradigm, analyze baseline limitations, and design module/component innovations to improve performance.

## Starting State

The previous best recorded result was `task_aware_lora_gate`, with `Avg=73.6825`, but it is a task-aware inference policy rather than a static or baseline-compatible training-method improvement.

## Actions Taken

- Added `## 0. Baseline-First Research Constraint` to repository `AGENTS.md`.
- Updated local anchor `AGENTS.md` with the same research-direction constraint.
- Updated `docs/agent_work/TASK_STATE.md` to make this the latest durable repository state.

## Files Changed

- `AGENTS.md`
- `docs/agent_work/TASK_STATE.md`
- `docs/agent_work/logs/2026-05-08-baseline-first-research-constraint.md`

## Verification

- Grep confirmed the rule is present in AGENTS.md, TASK_STATE.md, and the session log.
- git diff --check passed with no whitespace errors.
- git status --short showed only this documentation/instruction update before commit.

## Current Task State

DONE after verification and push.

## Remaining Risks

The existing `docs/paper/README.md` still records the task-aware gate result as a result artifact. Future paper writing should treat it as diagnostic evidence, not the final contribution.

## Recommended Next Action

Start a baseline limitation review and propose a concrete baseline-compatible method: module/component/objective/mask/scoring/regularization/training-loop innovation, with smoke tests before full runs.
