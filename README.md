# Tool Guard

This is the implementation of the ICML 2026 Paper **"Think Twice Before You
Act: Protecting LLM Agents Against Tool Description Poisoning via Isolated
Planning"**. This implementation is built on top of
[AgentDojo](https://github.com/ethz-spylab/agentdojo).

Throughout this README, **TDP** refers to **Tool Description Poisoning** —
the attack class where an adversary tampers with tool descriptions visible
to the LLM agent so that the agent is tricked into invoking malicious tool
calls. The baseline TDP attack used everywhere is the authority-injection
strategy in `src/agentdojo/attacks/tool_description_poisoning.py`.

The repository is organised around three reproducible experiments:

| # | Experiment                                                  | Driver script                            |
|---|-------------------------------------------------------------|------------------------------------------|
| 1 | Tool Guard vs TDP, **across LLMs**                          | `examples/evaluate_split_replan_defense.py` |
| 2 | Tool Guard vs **other defenses**, fixed on gpt-4o           | `examples/evaluate_multi_defense_tokens.py` |
| 3 | Tool Guard vs **adaptive attacks** (alignment / suspicion / combined / PAIR / TAP) | `examples/evaluate_split_replan_defense.py --adaptive-type ...` |

Reference numbers live under `results/`, split into the same three tracks
(`results/models/`, `results/multi_defense/`, `results/adaptive_pair_tap/`,
`results/adaptive_alignment_suspicion/`).

---

## 1. Setup

### 1.1 System requirements

* Python 3.10 or newer (3.12 is what we used).
* `pip` (or `uv`, equivalent).
* Outbound network access to the OpenAI / Anthropic / Google APIs you intend
  to use.

### 1.2 Create an environment

```bash
# Conda
conda create -n tool_guard python=3.10 -y
conda activate tool_guard

# OR uv (faster)
# uv venv .venv --python 3.10 && source .venv/bin/activate
```

### 1.3 Install the package in editable mode

```bash
cd tool_guard
pip install -e .
```

This pulls in OpenAI / Anthropic / Google SDKs (`openai`, `anthropic`,
`google-genai`), `pydantic`, `langchain`, `tenacity`, `pyyaml`, `click`, etc.

DRIFT (used by Task 2) additionally depends on PyTorch. If you plan to run
the `drift` defense, install:

```bash
pip install torch
```

If you skip torch, every other defense in Task 2 still works — `drift` will
just be reported as unavailable.

### 1.4 API keys

Export the keys for the providers you want to evaluate:

```bash
export OPENAI_API_KEY="sk-..."          # always needed (validator, attacks, defaults)
export ANTHROPIC_API_KEY="..."         # only for --provider anthropic
export GOOGLE_API_KEY="..."            # only for --provider google
```

You can put them in a local `.env` and `source` it before launching.

---

## 2. Folder architecture

```
tool_guard/
├── examples/
│   ├── evaluate_split_replan_defense.py     # Task 1 + Task 3 driver
│   ├── evaluate_multi_defense_tokens.py     # Task 2 driver
│   ├── evaluate_progent.py                  # ProGent helpers imported by Task 2
│   ├── evaluate_drift_tdp.py                # Standalone DRIFT evaluator
│   └── counter_benchmark/                   # Built-in injection / utility benchmark
├── src/
│   ├── agentdojo/                           # Core AgentDojo fork
│   │   ├── attacks/
│   │   │   ├── tool_description_poisoning.py    # Standard + adaptive TDP attacks
│   │   │   └── adaptive_description_optimizers.py # PAIR / TAP optimisers
│   │   └── agent_pipeline/
│   │       ├── split_replan_defense.py      # Tool Guard core
│   │       ├── tool_description_defense.py  # Pattern-based sanitization
│   │       └── token_tracking.py            # Token tracker used by Task 2
│   └── drift/                                # DRIFT defense (Task 2)
├── secagent/                                  # ProGent / SecAgent defense (Task 2)
└── results/
    ├── models/                                # Task 1 reference numbers
    ├── multi_defense/                         # Task 2 reference numbers
    ├── adaptive_pair_tap/                     # Task 3 — PAIR / TAP attack results
    └── adaptive_alignment_suspicion/          # Task 3 — alignment / suspicion / combined attack results
```

---

## 3. Reproducing the experiments

All three experiments share the same TDP attack
(`authority_injection_strategy` in
`src/agentdojo/attacks/tool_description_poisoning.py`) and the same
Split-Replan / Tool Guard defense
(`src/agentdojo/agent_pipeline/split_replan_defense.py`). Only the model,
the defense, or the attack strategy changes between experiments.

### 3.1 Task 1 — Defense effectiveness under different models

Driver: `examples/evaluate_split_replan_defense.py`.

Each run does **four passes** per task: benign (no defense), benign +
defense, attack (no defense), attack + defense. The script reports utility,
ASR with and without defense, defense effectiveness, false-positive rate,
and per-pass latency.

Example sweep over `banking / workspace / slack / travel` with
`gpt-4o-mini`:

```bash
for suite in banking workspace slack travel; do
  python examples/evaluate_split_replan_defense.py \
      --provider openai --model gpt-4o-mini \
      --suite $suite --all-tasks \
      --output results/models/${suite}_gpt4o-mini.json \
      2>&1 | tee results/models/${suite}_gpt4o-mini.log
done
```

To reproduce the cross-model numbers we report (`gpt-4o`, `gpt-4o-mini`,
`claude-3-5-haiku-20241022`, `gemini-2.5-flash`, `gemini-2.5-pro`), swap
the `--provider` / `--model` flags:

```bash
# OpenAI gpt-4o
python examples/evaluate_split_replan_defense.py \
    --provider openai --model gpt-4o --suite banking --all-tasks \
    --output results/models/banking_gpt4o.json

# Anthropic Claude 3.5 Haiku
python examples/evaluate_split_replan_defense.py \
    --provider anthropic --model claude-3-5-haiku-20241022 --suite banking --all-tasks \
    --output results/models/banking_claude35haiku.json

# Google Gemini 2.5 Flash
python examples/evaluate_split_replan_defense.py \
    --provider google --model gemini-2.5-flash --suite banking --all-tasks \
    --output results/models/banking_gemini25flash.json
```

Reference JSON / LOG files for every model are already in `results/models/`.

### 3.2 Task 2 — Different defenses, fixed model

Driver: `examples/evaluate_multi_defense_tokens.py`.

Each run does **two passes** per task — `benign + defense` and
`attack + defense` — and additionally records prompt / completion / total
tokens and latency for every pass. Use the same driver for every defense
listed below, varying `--defense`:

| `--defense`     | Defense name                                     |
|-----------------|--------------------------------------------------|
| `none`          | System-prompt baseline (a.k.a. **system prompt** / **simple**) |
| `tool_filter`   | Built-in AgentDojo tool filter                   |
| `repeat_prompt` | Repeat the user query before every LLM call      |
| `drift`         | DRIFT defense (`src/drift/`, needs torch)        |
| `progent`       | ProGent / SecAgent policy validation (`secagent/`) |
| `split_replan`  | Tool Guard (this work, for comparison)           |

Full sweep on `gpt-4o`:

```bash
for defense in none tool_filter repeat_prompt drift progent split_replan; do
  for suite in banking workspace slack travel; do
    python examples/evaluate_multi_defense_tokens.py \
        --defense $defense --model gpt-4o --suite $suite --all-tasks \
        --output results/multi_defense/${defense}_${suite}_gpt4o.json \
        2>&1 | tee results/multi_defense/${defense}_${suite}_gpt4o.log
  done
done
```

The reference numbers we shipped under `results/multi_defense/` are
produced with `gpt-4o` on every defense, so the command above reproduces
them directly. The output JSON for every defense + suite combination
includes per-task token counts and latencies, so the **overhead** for each
defense is recorded directly in the result files.

### 3.3 Task 3 — Adaptive attacks against Tool Guard

Driver: still `examples/evaluate_split_replan_defense.py`, but with
`--adaptive-type`. Supported attack strategies:

| `--adaptive-type` | Strategy                                                | Implementation                                                                   |
|-------------------|---------------------------------------------------------|----------------------------------------------------------------------------------|
| *unset*           | Standard authority injection (baseline TDP attack)      | `authority_injection_strategy`                                                   |
| `alignment`       | Bypass alignment check only                             | `alignment_adaptive_strategy` (`tool_description_poisoning.py`)                  |
| `suspicion`       | Bypass suspicion check only                             | `suspicion_adaptive_strategy`                                                    |
| `combined`        | Bypass both checks                                      | `combined_adaptive_strategy`                                                     |
| `pair`            | PAIR-style optimisation against validator prompts       | `pair_adaptive_strategy` → `adaptive_description_optimizers.pair_optimize_description` |
| `tap`             | TAP-style tree-of-attack optimisation                   | `tap_adaptive_strategy` → `adaptive_description_optimizers.tap_optimize_description`   |

Reproduce the full results in `results/adaptive_alignment_suspicion/` and
`results/adaptive_pair_tap/`:

```bash
# Alignment / suspicion / combined adaptive attacks
for atk in alignment suspicion combined; do
  for suite in banking workspace slack travel; do
    python examples/evaluate_split_replan_defense.py \
        --provider openai --model gpt-4o-mini \
        --suite $suite --all-tasks --adaptive-type $atk \
        --output results/adaptive_alignment_suspicion/${suite}_split_replan_${atk}.json \
        2>&1 | tee results/adaptive_alignment_suspicion/${suite}_${atk}.log
  done
done

# PAIR and TAP optimisation-based attacks (write per-iteration logs)
export ADAPTIVE_OPT_MAX_ITERS=20      # default; lower if you want a quick run
export ADAPTIVE_OPT_RECORD_PATH=results/adaptive_pair_tap/adaptive_optimization_records.jsonl
for atk in pair tap; do
  for suite in banking workspace slack travel; do
    python examples/evaluate_split_replan_defense.py \
        --provider openai --model gpt-4o-mini \
        --suite $suite --all-tasks --adaptive-type $atk \
        --output results/adaptive_pair_tap/split_replan_${atk}_${suite}_gpt4omini.json \
        2>&1 | tee results/adaptive_pair_tap/split_replan_${atk}_${suite}_gpt4omini.log
  done
done
```

Useful env vars for the optimisation-based attacks (defined in
`src/agentdojo/attacks/adaptive_description_optimizers.py`):

| Variable                       | Default                                              | Meaning                              |
|--------------------------------|------------------------------------------------------|--------------------------------------|
| `ADAPTIVE_OPT_MODEL`           | `gpt-4o-mini`                                        | Model that drives the optimiser      |
| `ADAPTIVE_OPT_MAX_ITERS`       | `20`                                                 | Max optimisation iterations          |
| `ADAPTIVE_TAP_WIDTH`           | `3`                                                  | TAP frontier width                   |
| `ADAPTIVE_TAP_BRANCHING`       | `2`                                                  | TAP branching factor                 |
| `ADAPTIVE_OPT_RECORD_PATH`     | `test_results/adaptive_optimization_records.jsonl`   | Per-iteration before/after JSONL log |

The optimiser writes before/after descriptions to the JSONL file pointed
to by `ADAPTIVE_OPT_RECORD_PATH`, which is what
`results/adaptive_pair_tap/adaptive_optimization_records.jsonl` captures
for the reference runs.

---

## 4. Command reference

### `examples/evaluate_split_replan_defense.py`

| Option              | Description |
|---------------------|-------------|
| `--provider`, `-p`  | `openai`, `anthropic`, `google` (default `openai`) |
| `--model`, `-m`     | Model identifier (e.g. `gpt-4o-mini`, `claude-3-5-haiku-20241022`, `gemini-2.5-flash`) |
| `--suite`           | One of `banking`, `workspace`, `slack`, `travel` |
| `--num-tasks N`     | Evaluate the first N user tasks |
| `--all-tasks`       | Evaluate every user task in the suite |
| `--tasks ID [ID …]` | Evaluate specific user-task ids |
| `--debug`           | Verbose defense / validator logging |
| `--quiet`           | Minimal output |
| `--output PATH`     | Save aggregated results as JSON |
| `--adaptive-type`   | `pair`, `tap`, `alignment`, `suspicion`, `combined` |

### `examples/evaluate_multi_defense_tokens.py`

| Option            | Description |
|-------------------|-------------|
| `--defense`       | `none`, `tool_filter`, `repeat_prompt`, `drift`, `progent`, `split_replan`, `tdp_defense` |
| `--model`         | Model identifier (currently OpenAI; e.g. `gpt-4o-mini`, `gpt-4o`) |
| `--suite`         | `banking`, `workspace`, `slack`, `travel` |
| `--num-tasks N`   | First N tasks |
| `--all-tasks`     | Every task |
| `--tasks ID …`    | Specific task ids |
| `--debug`         | Verbose internal logging |
| `--output PATH`   | Save JSON results (includes per-task tokens + latency) |

---

## 5. Acknowledgements

This codebase builds on
[AgentDojo](https://github.com/ethz-spylab/agentdojo) and reuses the DRIFT
(`src/drift/`) and ProGent / SecAgent (`secagent/`) defenses from their
published implementations.
