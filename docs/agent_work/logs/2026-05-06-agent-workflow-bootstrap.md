# Session Log: Agent Workflow Bootstrap

Date: 2026-05-06
State: DONE

## Goal

Replace the old external/local agent-instruction habit with repository-local operating rules: every repository-working turn writes a handoff log, large work gets a detailed Markdown plan, large execution should use a controller/worker split, and project progress should be managed as state transitions.

## Starting Context

- Remote repository: `/data/guoboyang/LoRa-Projects/LoRASculpt-repro/LoRASculpt` on `guoboyang3090`.
- No repository-root `AGENTS.md`, `Agent.md`, or `agent.md` was found in the remote code path.
- Existing dirty code files were present before this task and were treated as unrelated user work.

## Actions Taken

- Added root `AGENTS.md` as the repository-level workflow entry point.
- Added `docs/agent_work/README.md` to explain durable agent artifacts.
- Added `docs/agent_work/plans/2026-05-06-agent-workflow-bootstrap-plan.md` and marked it done.
- Added reusable plan and session-log templates.
- Added `docs/agent_work/TASK_STATE.md` with the current state machine and next action.
- Added `.gitkeep` files for empty coordination directories.

## Files Changed

- `AGENTS.md`: new operating model for logs, plans, controller/worker split, recovery, state machine, verification, and safety.
- `docs/agent_work/README.md`: directory guide.
- `docs/agent_work/TASK_STATE.md`: current project-agent state.
- `docs/agent_work/plans/2026-05-06-agent-workflow-bootstrap-plan.md`: setup plan and completion notes.
- `docs/agent_work/templates/session_log_template.md`: reusable handoff log template.
- `docs/agent_work/templates/plan_template.md`: reusable large-task plan template.
- `docs/agent_work/logs/2026-05-06-agent-workflow-bootstrap.md`: this handoff log.
- `docs/agent_work/logs/.gitkeep`, `docs/agent_work/plans/.gitkeep`, `docs/agent_work/handoffs/.gitkeep`: keep coordination directories visible to git.
- `.gitignore`: added a narrow exception so `docs/agent_work/logs/**` is trackable while normal project logs remain ignored.

## Commands And Evidence

- Command: `find .. -maxdepth 3 \( -name AGENTS.md -o -name Agent.md -o -name agent.md \) -print`
- Result: no existing agent instruction file was found near the remote code path.

- Command: `git status --short`
- Result: pre-existing modifications were visible in trainer/train scripts; after this setup, new `AGENTS.md` and `docs/agent_work/` were added.

- Command: `rg -n "Every-Turn|Detailed Plans|Controller / Worker|Task State Machine|Recovery" ...`
- Result: failed because remote `rg` is not installed.

- Command: `grep -nE "Every-Turn|Detailed Plans|Controller / Worker|Task State Machine|Recovery" AGENTS.md docs/agent_work/plans/2026-05-06-agent-workflow-bootstrap-plan.md`
- Result: confirmed all requested sections exist in `AGENTS.md`.

- Command: `wc -l AGENTS.md docs/agent_work/README.md docs/agent_work/plans/2026-05-06-agent-workflow-bootstrap-plan.md docs/agent_work/templates/session_log_template.md docs/agent_work/templates/plan_template.md`
- Result: core files are present and non-empty.


- Command: `git ls-files --others --exclude-standard -- AGENTS.md docs/agent_work .gitignore | sort`
- Result: confirmed `AGENTS.md`, `docs/agent_work/TASK_STATE.md`, the plan, templates, `.gitkeep` files, and the session log are trackable as untracked files.

- Command: `grep -nE "docs/agent_work/logs|Every-Turn|Detailed Plans|Controller / Worker|Task State Machine|Recovery|Current state: DONE" .gitignore AGENTS.md docs/agent_work/TASK_STATE.md`
- Result: confirmed the `.gitignore` exception, requested `AGENTS.md` sections, and DONE task state.

## Failures Or Blockers

- First remote write attempt failed due shell quote parsing before reaching the remote repository; no files were changed by that failed attempt.
- `rg` is unavailable on the remote server, so `grep` was used.
- The existing `logs/` ignore rule initially hid `docs/agent_work/logs/`; this was fixed with a scoped `.gitignore` exception.

## Current State

DONE. The operating model and supporting repository artifacts are in place.

## Remaining Risks

- The files are currently untracked until staged/committed.
- Future agents must read `AGENTS.md` for the workflow to take effect.

## Next Action

For the next substantial LoRASculpt task, create a detailed plan under `docs/agent_work/plans/`, update `TASK_STATE.md`, and use a worker/subagent for execution when available and appropriate.
