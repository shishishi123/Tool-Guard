# Tool Guard

This is the implementation of the ICML 2026 Paper **"Think Twice Before You
Act: Protecting LLM Agents Against Tool Description Poisoning via Isolated
Planning"**. This implementation is built on top of
[AgentDojo](https://github.com/ethz-spylab/agentdojo).

Throughout this README, **TDP** refers to **Cross-Tool Description Poisoning** —
where an adversary tampers with tool descriptions visible to the LLM agent so that the agent is tricked into invoking malicious tool
calls. The baseline TDP attack used everywhere is in `src/agentdojo/attacks/tool_description_poisoning.py`.

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

DRIFT (used by Experiment 2) additionally depends on PyTorch. If you plan to run
the `drift` defense, install:

```bash
pip install torch
```

If you skip torch, every other defense in Experiment 2 still works — `drift` will
just be reported as unavailable.

### 1.4 API keys

Export the keys for the providers you want to evaluate:

```bash
export OPENAI_API_KEY="sk-..."          # always needed (validator, attacks, defaults)
export ANTHROPIC_API_KEY="..."         # only for --provider anthropic
export GOOGLE_API_KEY="..."            # only for --provider google
```

You can put them in a local `.env` and `source` it before launching.

### 1.5 Quick smoke test

Once setup is complete, you can verify the pipeline end-to-end with a tiny
`banking` suite run on `gpt-4o-mini` (only the first user task):

```bash
python examples/evaluate_split_replan_defense.py \
    --provider openai --model gpt-4o-mini \
    --suite banking --num-tasks 1 \
    --output /tmp/smoke_banking.json
```

A successful run does all four passes (benign / benign+defense /
attack / attack+defense), prints utility, ASR with and without defense, and
per-pass latency, and writes the full record (including token counts and
latency) to `/tmp/smoke_banking.json`. The whole run typically completes in
a couple of minutes and costs only a few cents.

---

## 2. Folder architecture

```
tool_guard/
├── examples/
│   ├── evaluate_split_replan_defense.py     # Experiment 1 + Experiment 3 driver
│   ├── evaluate_multi_defense_tokens.py     # Experiment 2 driver
│   ├── evaluate_progent.py                  # ProGent helpers imported by Experiment 2
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
│   │       └── token_tracking.py            # Token tracker used by Experiment 2
│   └── drift/                                # DRIFT defense (Experiment 2)
├── secagent/                                  # ProGent / SecAgent defense (Experiment 2)
└── results/
    ├── models/                                # Experiment 1 reference numbers
    ├── multi_defense/                         # Experiment 2 reference numbers
    ├── adaptive_pair_tap/                     # Experiment 3 — PAIR / TAP attack results
    └── adaptive_alignment_suspicion/          # Experiment 3 — alignment / suspicion / combined attack results
```

---

## 3. Reproducing the experiments

All three experiments share the same TDP attack
(`src/agentdojo/attacks/tool_description_poisoning.py`) and the same
Split-Replan / Tool Guard defense
(`src/agentdojo/agent_pipeline/split_replan_defense.py`). Only the model,
the defense, or the attack strategy changes between experiments.

> **Latency & overhead are recorded automatically.** Every run from any of
> the three driver scripts writes per-task / per-pass latency and prompt /
> completion / total token counts into the `--output` JSON file (and, for
> Experiment 3 with PAIR / TAP, per-iteration optimiser records into the
> JSONL file pointed to by `ADAPTIVE_OPT_RECORD_PATH`). You don't need to
> add any extra flag — defense overhead is captured for every command in
> this section.

### 3.1 Experiment 1 — Defense effectiveness under different models

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

### 3.2 Experiment 2 — Different defenses, fixed model

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

### 3.3 Experiment 3 — Adaptive attacks against Tool Guard

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

> **We strongly recommend running Experiment 3 under `nohup`.** PAIR and
> TAP call the optimiser model many times per user task, so a full sweep
> over all four suites can take **several hours**. Wrapping each sweep in
> `nohup ... &` lets it survive shell disconnects and writes a single
> log file you can `tail -f`. Per-pass latency, token counts, and (for
> PAIR / TAP) per-iteration optimiser records are still saved to the
> output JSON / JSONL files as the run progresses.

Reproduce the full results in `results/adaptive_alignment_suspicion/` and
`results/adaptive_pair_tap/`:

```bash
# Alignment / suspicion / combined adaptive attacks (run in background)
nohup bash -c '
for atk in alignment suspicion combined; do
  for suite in banking workspace slack travel; do
    python examples/evaluate_split_replan_defense.py \
        --provider openai --model gpt-4o-mini \
        --suite $suite --all-tasks --adaptive-type $atk \
        --output results/adaptive_alignment_suspicion/${suite}_split_replan_${atk}.json
  done
done
' > results/adaptive_alignment_suspicion/run.log 2>&1 &
echo "alignment/suspicion/combined PID=$!"

# PAIR and TAP optimisation-based attacks (run in background)
nohup bash -c '
export ADAPTIVE_OPT_MAX_ITERS=20    # default; lower (e.g. 5) for a quick check
export ADAPTIVE_OPT_RECORD_PATH=results/adaptive_pair_tap/adaptive_optimization_records.jsonl
for atk in pair tap; do
  for suite in banking workspace slack travel; do
    python examples/evaluate_split_replan_defense.py \
        --provider openai --model gpt-4o-mini \
        --suite $suite --all-tasks --adaptive-type $atk \
        --output results/adaptive_pair_tap/split_replan_${atk}_${suite}_gpt4omini.json
  done
done
' > results/adaptive_pair_tap/run.log 2>&1 &
echo "PAIR/TAP PID=$!"
```

Monitor with:

```bash
tail -f results/adaptive_pair_tap/run.log
# or in another shell
tail -f results/adaptive_alignment_suspicion/run.log
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

## 4. Estimated reproduction cost

Costs depend heavily on (a) the model's per-token price at the time you
run, (b) the suite size (banking has ~17 user tasks, slack ~21, travel
~20, workspace ~40, each multiplied by every relevant injection task in
the attack passes), and (c) for PAIR / TAP, the optimisation budget set
through `ADAPTIVE_OPT_MAX_ITERS`. The numbers below are **rough
order-of-magnitude estimates in USD** based on late-2025 OpenAI /
Anthropic / Google pricing and the token counts we logged in `results/`
during our reference runs. Use them for planning, not invoicing.

| Step                                                                              | Model                                  | Approx. cost (USD) |
|-----------------------------------------------------------------------------------|----------------------------------------|--------------------|
| Quick smoke test (Section 1.5)                                                    | `gpt-4o-mini`                          | < $0.10            |
| **Experiment 1** — single suite, single model                                     | `gpt-4o-mini`                          | $1 – $3            |
| **Experiment 1** — all four suites, single model                                  | `gpt-4o-mini`                          | $5 – $10           |
| **Experiment 1** — all four suites, single model                                  | `gpt-4o`                               | $30 – $80          |
| **Experiment 1** — all four suites, single model                                  | `claude-3-5-haiku`, `gemini-2.5-flash` | $5 – $20 each      |
| **Experiment 1** — all four suites, single model                                  | `gemini-2.5-pro`                       | $30 – $60          |
| **Experiment 1** — full cross-model table (all five models)                       | mixed                                  | $100 – $250        |
| **Experiment 2** — one defense, all four suites                                   | `gpt-4o`                               | $20 – $50          |
| **Experiment 2** — all six defenses, all four suites                              | `gpt-4o`                               | $150 – $300        |
| **Experiment 3** — alignment / suspicion / combined, all four suites              | `gpt-4o-mini`                          | $5 – $15           |
| **Experiment 3** — PAIR *or* TAP, all four suites (20 optimisation iters)         | `gpt-4o-mini`                          | $30 – $100 each    |

If you want representative numbers on a tighter budget:

* Pass `--num-tasks 5` instead of `--all-tasks` to run only the first few
  user tasks per suite.
* For Experiment 3, lower `ADAPTIVE_OPT_MAX_ITERS` (e.g. `5`) to cap the
  PAIR / TAP optimiser early.
* Drop the `--provider anthropic` / `--provider google` rows of
  Experiment 1 if you only need the OpenAI numbers.

---

## 5. Command reference

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

## 6. Acknowledgements

This codebase builds on
[AgentDojo](https://github.com/ethz-spylab/agentdojo) and reuses the DRIFT
(`src/drift/`) and ProGent / SecAgent (`secagent/`) defenses from their
published implementations.
