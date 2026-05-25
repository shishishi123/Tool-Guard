# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tool Guard** is a reusable implementation of a Split-Replan isolation defense against
Tool Description Poisoning (TDP) attacks on LLM agents. It is built on top of
[AgentDojo](https://github.com/ethz-spylab/agentdojo) and packages the attacks,
defenses, and adaptive evaluations referenced in the paper.

## Evaluation Tracks

1. **Task 1 — Tool Guard vs models** (`examples/evaluate_split_replan_defense.py`):
   Cross-model evaluation of the Split-Replan defense against authority-injection TDP.
2. **Task 2 — Multi-defense comparison** (`examples/evaluate_multi_defense_tokens.py`):
   Compares Tool Guard against DRIFT, ProGent, tool_filter, repeat_prompt, and the
   simple system-prompt baseline. Reuses `examples/evaluate_progent.py`.
3. **Task 3 — Adaptive attacks**:
   * Optimisation-based: PAIR / TAP via
     `src/agentdojo/attacks/adaptive_description_optimizers.py`.
   * Bypass-targeted: alignment / suspicion / combined in
     `src/agentdojo/attacks/tool_description_poisoning.py`.
4. **Task 4 — Results curation**: see `results/{models,multi_defense,
   adaptive_pair_tap,adaptive_alignment_suspicion}`.

## Key Commands

### Environment

```bash
conda create -n tool_guard python=3.10 -y && conda activate tool_guard
cd tool_guard && pip install -e .
```

### Smoke test

```bash
python examples/evaluate_split_replan_defense.py \
    --provider openai --model gpt-4o-mini \
    --suite banking --num-tasks 1 --debug
```

### Multi-defense smoke test

```bash
python examples/evaluate_multi_defense_tokens.py \
    --defense split_replan --suite banking --num-tasks 1 --debug
```

### Linting / typing (optional)

```bash
uv run ruff check --fix .
uv run pyright
```

## Code Architecture

### Agent pipeline (`src/agentdojo/agent_pipeline/`)

* `split_replan_defense.py` – Tool Guard core (validator + split replan + decision matrix).
* `tool_description_defense.py` – Pattern-based sanitization defense (TDP defense).
* `tool_execution.py`, `agent_pipeline.py`, `basic_elements.py` – AgentDojo primitives.
* `token_tracking.py` – Wraps OpenAI/Anthropic/Google clients to track token usage.
* `llms/` – Per-provider LLM elements (OpenAI, Anthropic, Google, Cohere, etc.).

### Attacks (`src/agentdojo/attacks/`)

* `tool_description_poisoning.py` – Baseline TDP strategies (append, prepend, subtle,
  authority) plus the adaptive bypass strategies (alignment, suspicion, combined) and
  optimisation wrappers (PAIR, TAP).
* `adaptive_description_optimizers.py` – Implementation of PAIR-style iterative and
  TAP-style tree-of-attack optimisations driven by `OpenAI` calls.
* `baseline_attacks.py`, `important_instructions_attacks.py`, `dos_attacks.py` –
  Original AgentDojo attack zoo (retained for completeness, not required by the 4 tasks).

### External defenses

* `src/drift/` – DRIFT defense (dynamic validation + injection isolation). Used by
  `create_drift_pipeline` in `evaluate_multi_defense_tokens.py`.
* `secagent/` – ProGent / SecAgent defense (policy generation + JSON-schema validation).
  Used via `evaluate_progent.py`.

### Default suites

`src/agentdojo/default_suites/` ships banking, workspace, slack, travel, and webbase
suites identical to upstream AgentDojo.

## Working notes for agents

* The Split-Replan defense and TDP attack logic must NOT be modified — both are the
  artefacts under test. When fixing bugs, prefer touching the harness/evaluation glue.
* If `tool_guard/src/agentdojo` and the reference `agentdojo` diverge, treat
  `agentdojo` (in the workspace root) as the source of truth.
* `evaluate_progent.py` and `evaluate_drift_tdp.py` only exist to provide module-level
  symbols for the multi-defense script; do not delete them when pruning.

## Code Style

* Python 3.10+ typing (`list[str]`, `int | None`).
* Module-level imports only.
* Standard import order (stdlib, third-party, local) with `ruff` running `I`/`F`.
