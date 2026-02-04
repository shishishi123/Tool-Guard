# Split-Replan Defense Evaluation

Evaluation of Split-Replan Isolation Defense against Tool Description Poisoning (TDP) attacks. Built on top of [AgentDojo](https://github.com/ethz-spylab/agentdojo).

## Setup

### 1. Create Conda Environment

```bash
conda create -n tool_guard python=3.10 -y
conda activate tool_guard
```

### 2. Install Dependencies

```bash
cd tool_guard
pip install -e .
```

### 3. Set API Keys

Set your LLM provider API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google
export GOOGLE_API_KEY="your-google-key"
```

## Running Evaluations

### Quick Test (1 task, 1 suite)

Debug mode with a single task on the banking suite:

```bash
python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite banking --num-tasks 1 --debug
```

### Run All 4 Suites (Sequential)

```bash
python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite banking --all-tasks --output results/split_replan_banking.json
python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite workspace --all-tasks --output results/split_replan_workspace.json
python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite slack --all-tasks --output results/split_replan_slack.json
python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite travel --all-tasks --output results/split_replan_travel.json
```

### Run All 4 Suites (Background with nohup)

For long-running experiments that should continue after terminal disconnect:

```bash
nohup python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite banking --all-tasks --output results/split_replan_banking.json > results/split_replan_banking.log 2>&1 &
nohup python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite workspace --all-tasks --output results/split_replan_workspace.json > results/split_replan_workspace.log 2>&1 &
nohup python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite slack --all-tasks --output results/split_replan_slack.json > results/split_replan_slack.log 2>&1 &
nohup python examples/evaluate_split_replan_defense.py --model gpt-4o-mini --suite travel --all-tasks --output results/split_replan_travel.json > results/split_replan_travel.log 2>&1 &
```

## Command Options

| Option | Description |
|--------|-------------|
| `--provider`, `-p` | LLM provider: `openai`, `anthropic`, `google` (default: `openai`) |
| `--model`, `-m` | Model name (e.g., `gpt-4o-mini`, `claude-3-5-sonnet-20241022`, `gemini-1.5-flash`) |
| `--suite` | Task suite: `banking`, `workspace`, `slack`, `travel` |
| `--num-tasks` | Number of tasks to evaluate |
| `--all-tasks` | Evaluate all tasks in the suite |
| `--tasks` | Specific task IDs to evaluate |
| `--debug` | Enable verbose debug output |
| `--output` | Path to save JSON results |
| `--adaptive-type` | Adaptive attack type: `alignment`, `suspicion`, `combined` |

### Example with Different Models

```bash
# OpenAI GPT-4o
python examples/evaluate_split_replan_defense.py --provider openai --model gpt-4o --suite banking --all-tasks

# Anthropic Claude
python examples/evaluate_split_replan_defense.py --provider anthropic --model claude-3-5-sonnet-20241022 --suite banking --all-tasks

# Google Gemini
python examples/evaluate_split_replan_defense.py --provider google --model gemini-1.5-flash --suite banking --all-tasks
```

## Acknowledgements

This implementation is based on [AgentDojo](https://github.com/ethz-spylab/agentdojo).

