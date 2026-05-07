# Plan: Bootstrap Repository-Local Agent Operating Model

Date: 2026-05-06
Status: Done

## Goal

Create repository-local instructions and durable coordination documents so future Codex sessions write handoff logs, prepare detailed plans for large work, split controller/worker responsibilities, and manage progress as task state transitions.

## Definition Of Done

- Root `AGENTS.md` exists and defines the new workflow.
- `docs/agent_work/` exists with log, plan, handoff, state, and template locations.
- A current task state file exists.
- This setup turn has its own handoff log.
- Verification confirms only documentation/coordination files were added or changed.

## Assumptions

- The remote repository root is the correct place for project instructions.
- Existing training code modifications are unrelated user work and must not be touched.
- The workflow should be lightweight enough to use every turn.

## Steps

1. Inspect the repository for existing `AGENTS.md` or agent-state docs.
2. Add root `AGENTS.md` with the new repository-local operating model.
3. Add `docs/agent_work/` structure and templates.
4. Add initial `TASK_STATE.md` and this plan.
5. Add this turn's handoff log.
6. Verify with `git status --short` and a scoped diff/stat review.

## Verification Strategy

- Confirm the repository sees the new docs as untracked/modified documentation files.
- Confirm no existing training source files are edited by this setup.
- Review `AGENTS.md` content for the requested rules: every-turn logs, large-task plans, controller/worker split, recovery, and state transitions.

## Risks

- Future agents may ignore the workflow unless they read `AGENTS.md`; mitigate by keeping it at repository root.
- Too much process could slow small tasks; mitigate by requiring detailed plans only for large work while still keeping concise handoff logs.


## Completion Notes

- Added root `AGENTS.md` with the requested workflow rules.
- Added `docs/agent_work/` coordination structure, templates, state file, and session log.
- Existing modified training files were observed and intentionally left untouched.
