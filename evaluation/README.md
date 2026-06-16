# PiSAs — Privacy in Shared Agentic Systems

An evaluation harness for measuring **contextual-integrity (CI) violations** in multi-agent LLM
systems. It runs workplace scenarios through three agent topologies and four memory regimes, and
scores each on appropriateness violations, visibility violations, completeness, and task utility.

Two entry points share the same core (`agents.py` + `judges.py`):

- **`app.py`** — an interactive Streamlit demo for stepping through one scenario.
- **`run_pipeline.py` + `run_evaluation.py`** — a headless CLI for batch experiments.

---

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here   # only for native OpenAI models (gpt-5, gpt-5.5, o4-mini)
export HF_TOKEN=your_hf_token_here    # only to download the benchmark from Hugging Face
```

Model calls go through [OpenRouter](https://openrouter.ai) by default (via litellm), so any
OpenRouter model id works for agents, judges, and verifiers. A small registry routes a few models
elsewhere: `gpt-5`, `gpt-5.5`, and `o4-mini` call the OpenAI API directly (using `OPENAI_API_KEY`),
and `gpt-oss-120b` runs against a local server. Run all commands from the repository root.

The benchmark scenarios are hosted as a Hugging Face dataset and downloaded automatically (see
[Run the CLI](#run-the-cli)); `HF_TOKEN` is needed only for that download — not for local runs.
Get a read token at <https://huggingface.co/settings/tokens> (or run `huggingface-cli login`).

---

## How a run works

Every system solves the executor's task in **two stages**: an *information-gathering* stage that
produces a gathered-info summary, and a *decision* stage that uses only that summary. Violations are
measured at the gathering and agent-to-agent surfaces — never at the final output.

### Topologies

- **Single** — one agent sees every artifact; no inter-agent communication. The baseline, where
  privacy rests entirely on the model's own judgment.
- **Centralized** — one agent per user plus a fixed **Coordinator** that routes all communication.
  Data is partitioned by visibility, so the executor reaches its peers through the Coordinator.
- **Decentralized** — one agent per user, **token-passing**: the active agent forwards the
  conversation to the peer it judges most relevant. Begins and ends with the executor, no hub.

### Memory  *(Centralized & Decentralized only)*

| Mode | Demo toggles | CLI flags |
|---|---|---|
| None | — | *(none)* |
| Private | Private | `--private-memory` |
| Shared | Shared | `--shared-memory` |
| Hybrid | Private + Shared | `--private-memory --shared-memory` |

A preloading phase distills each agent's artifacts into structured notes — private memory
*replaces* the raw artifact text, shared memory *augments* it. Shared writes are open to `all`
agents, or (Centralized only) restricted to the Coordinator with `--shared-memory-writer executor`;
reads are always universal. Memory is scoped to a single run.

---

## Run the demo

```bash
streamlit run app.py                          # pick a scenario from the dropdown
streamlit run app.py -- -d <scenario_folder>  # or pin one by path
```

Opens at <http://localhost:8501>. Choose a topology, models, memory mode, and privacy level in the
sidebar, then hit **▶ Run Agent Task**. The API key auto-fills from `$OPENROUTER_API_KEY`.

---

## Run the CLI

`run_pipeline.py` orchestrates a run → `pipeline_<scenario>.json`; `run_evaluation.py` judges it →
`evaluation_<scenario>.json` beside it.

**By default — with no `-d` or `--scenarios-folder` — the runner downloads the PiSAs benchmark from
Hugging Face** (cached locally; needs `HF_TOKEN`) and runs one task, selected by `--task`
(default `JIRA_allocation`; also `meeting_allocation`, `severity_classification`), writing under
`results/PiSAs/<task>/<scenario>/`:

```bash
python run_pipeline.py -s centralized --agent-llm anthropic/claude-sonnet-4-6 --private-memory --shared-memory
python run_pipeline.py -s centralized --agent-llm anthropic/claude-sonnet-4-6 --task meeting_allocation
```

To run **local** scenarios instead, pass an explicit path with `-d` (single) or `--scenarios-folder`
(batch):

```bash
# Single local scenario
python run_pipeline.py -s centralized --agent-llm anthropic/claude-sonnet-4-6 \
    --private-memory --shared-memory -d data/<scenario> -o results/run1
python run_evaluation.py -d data/<scenario> -o results/run1 --agent-audit --memory-audit

# Batch — every scenario folder, K runs each
python run_pipeline.py -s centralized --agent-llm <model> \
    --scenarios-folder data --results-path results/run1 --run-index 0,1,2
python run_evaluation.py --scenarios-folder data --results-path results/run1 \
    --agent-audit --memory-audit
```

Batch mode forks one isolated subprocess per `(scenario, run-index)` over every subfolder that holds
a `scenario.json`. `aggregate_results.py` then rolls the per-scenario JSONs into summary tables, and
`launch_pipeline.sh` / `launch_evaluation.sh` sweep systems × privacy × memory × runs (and can spin
up a local SGLang server for open models).

**Key flags** (run either script with `-h` for the rest):

- `run_pipeline.py` — `-s {single,centralized,decentralized}`, `--agent-llm` (required),
  `--privacy-level {None,Low,Medium,High}` (default `High`), the memory flags above. Eval reads the
  rest of the config back from the pipeline JSON.
- `run_evaluation.py` — `--judge-llm` (default `gemini-2.5-pro`) plus the audit switches that select
  which surfaces to score: `--output-audit` (V_G, C, U — *on*), `--agent-audit` (V_A — off),
  `--memory-audit` (V_PMem, V_SMem — off).

---

## Metrics

Violations use a two-stage **extract → verify** pipeline: a lenient judge flags candidate values
*without* the ground truth, then a 3-model verifier committee (each given the ground truth) votes,
and a majority confirms each violation. Completeness uses a single ground-truth-aware judge with a
verbatim string-match guard against ground-truth leakage.

**Appropriateness** — inappropriate attributes that get disclosed:

> **V_G** (gathered-info summary) · **V_A2A** (agent-to-agent messages) · **V_appr = V_G ∪ V_A2A**

**Visibility** — attributes that reach an agent who should not access them:

> **V_A** (agent context) · **V_PMem** (private memory) · **V_SMem** (shared memory)
> · **V_vis = V_A ∪ V_PMem ∪ V_SMem**

**Completeness (C)** — fraction of *appropriate* attributes present in the gathered summary.
**Utility (U)** — binary: does the implied decision match the oracle?

Rates are normalized by the inappropriate-attribute count (appropriateness) or the count of
attributes hidden from ≥1 agent (visibility). Across K runs, violation rates take the **any-K
union** while completeness and utility take the **per-run mean**.

---

## Data & outputs

The full benchmark (all task families and their scenarios) lives in the PiSAs Hugging Face dataset
and is downloaded on demand. The local `data/` folder keeps a single sample scenario so the demo and
local runs work out of the box. Either way, each scenario folder holds four files:

| File | Contents |
|---|---|
| `scenario.json` | Cast, org structure, task, and the timeline of artifacts (each `visible_to` a set of users). |
| `appropriateness.json` | Per-attribute `appropriate` / `inappropriate` label. |
| `visibility.json` | Per-attribute `visible_to` / `hidden_from` user sets. |
| `utility.json` | Oracle answer, allowed answers, decision rule, ground-truth attribute values. |

Each run writes to `<output-dir>/<scenario_id>/`: a `pipeline_<id>.json` trace (config, gathered
summary, a2a log, per-agent contents, memory snapshots, timing — enough to re-judge without
re-running) and an `evaluation_<id>.json` with the scored metrics above.

---

## File overview

```
app.py                 # Streamlit demo (Single / Decentralized / Centralized)
agents.py              # Agent classes, topology runners, prompt templates, memory
judges.py              # LLM judges + verifier committee
run_pipeline.py        # CLI: orchestrate a run  → pipeline_*.json
run_evaluation.py      # CLI: judge a run        → evaluation_*.json
aggregate_results.py   # Roll per-scenario results into summary tables
eval_utils.py          # Shared metric-summary rendering (used by the demo)
launch_pipeline.sh     # Outer-loop sweep wrappers
launch_evaluation.sh   #   (systems × privacy × memory × runs)
data/                  # Scenario folders (one per scenario)
```
