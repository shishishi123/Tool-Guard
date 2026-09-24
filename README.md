# Tool Guard

Implementation of **Think Twice Before You Act: Protecting LLM Agents Against
Tool Description Poisoning via Isolated Planning** (ICML 2026). Built on
[AgentDojo](https://github.com/ethz-spylab/agentdojo).

**TDP** (Cross-Tool Description Poisoning) tampers with tool descriptions so the
agent invokes malicious tools. The attack is
`src/agentdojo/attacks/tool_description_poisoning.py`. The defense (Tool Guard)
is `src/agentdojo/agent_pipeline/split_replan_defense.py`.

This is the paper artifact: the three experiments, the Tool Guard defense, and
the shipped reference results.

| # | Experiment | Script |
|---|---|---|
| 1 | Tool Guard vs TDP, across LLMs | `experiments/evaluate_split_replan_defense.py` |
| 2 | Tool Guard vs other defenses (gpt-4o) | `experiments/evaluate_multi_defense_tokens.py` |
| 3 | Adaptive attacks (alignment / suspicion / combined / PAIR / TAP) | same as Exp 1, with `--adaptive-type` |

Reference numbers are under `results/models/`, `results/multi_defense/`,
`results/adaptive_pair_tap/`, and `results/adaptive_alignment_suspicion/`.

## Setup

Python 3.10+ (3.12 used here), then:

```bash
conda create -n tool_guard python=3.10 -y && conda activate tool_guard
pip install -e .
export OPENAI_API_KEY="sk-..."          # required (validator, attacks, defaults)
export ANTHROPIC_API_KEY="..."          # only for --provider anthropic
export GOOGLE_API_KEY="..."             # only for --provider google
```

Experiment 2's `drift` defense also needs `pip install torch`. Skip torch and
every other defense still runs.

## Smoke test

```bash
python experiments/evaluate_split_replan_defense.py \
    --provider openai --model gpt-4o-mini \
    --suite banking --num-tasks 1 \
    --output /tmp/smoke_banking.json
```

Four passes (benign / benign+defense / attack / attack+defense), a few minutes,
typically a few cents.

## Layout

```
.
├── experiments/
│   ├── evaluate_split_replan_defense.py   # Exp 1 + Exp 3
│   ├── evaluate_multi_defense_tokens.py   # Exp 2
│   └── evaluate_progent.py                # imported by Exp 2
├── src/
│   ├── agentdojo/                         # AgentDojo runtime + Tool Guard
│   │   ├── agent_pipeline/split_replan_defense.py
│   │   └── attacks/tool_description_poisoning.py
│   └── drift/                             # Exp 2 baseline
├── secagent/                              # Exp 2 ProGent / SecAgent baseline
└── results/                               # shipped reference JSON
```

## Reproduce

Latency, tokens, and overhead are written into `--output` automatically.

### Experiment 1 — across models

```bash
python experiments/evaluate_split_replan_defense.py \
    --provider openai --model gpt-4o-mini \
    --suite banking --all-tasks \
    --output results/models/banking_gpt4o-mini.json
```

Suites: `banking`, `workspace`, `slack`, `travel`.
Models we report, with `--provider`:

- `openai`: `gpt-4o`, `gpt-4o-mini`
- `anthropic`: `claude-3-5-haiku-20241022`
- `google`: `gemini-2.5-flash`, `gemini-2.5-pro`

### Experiment 2 — other defenses, gpt-4o

`--defense`: `none`, `tool_filter`, `repeat_prompt`, `drift`, `progent`, `split_replan`.

```bash
python experiments/evaluate_multi_defense_tokens.py \
    --defense split_replan --model gpt-4o --suite banking --all-tasks \
    --output results/multi_defense/split_replan_banking_gpt4o.json
```

### Experiment 3 — adaptive attacks (gpt-4o-mini)

`--adaptive-type`: `alignment`, `suspicion`, `combined`, `pair`, `tap`.

```bash
python experiments/evaluate_split_replan_defense.py \
    --provider openai --model gpt-4o-mini \
    --suite banking --all-tasks --adaptive-type alignment \
    --output results/adaptive_alignment_suspicion/banking_split_replan_alignment.json
```

PAIR / TAP extras: `ADAPTIVE_OPT_MAX_ITERS` (default 20),
`ADAPTIVE_OPT_RECORD_PATH` (per-iteration JSONL). Reference files are already
in `results/adaptive_*`; skip this experiment if you only need the numbers.

## Cost (order of magnitude)

| Step | Model | USD |
|---|---|---|
| Smoke test | gpt-4o-mini | < $0.10 |
| Exp 1, four suites | gpt-4o-mini | $5 – $10 |
| Exp 1, four suites | gpt-4o | $30 – $80 |
| Exp 2, all defenses × four suites | gpt-4o | $150 – $300 |
| Exp 3, alignment / suspicion / combined | gpt-4o-mini | $1 – $3 |
| Exp 3, PAIR / TAP | gpt-4o-mini | $2 – $15 |

Use `--num-tasks 5` or lower `ADAPTIVE_OPT_MAX_ITERS` to cut cost.

## Acknowledgements

Builds on [AgentDojo](https://github.com/ethz-spylab/agentdojo). Experiment 2
reuses DRIFT (`src/drift/`) and ProGent / SecAgent (`secagent/`).
