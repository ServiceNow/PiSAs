# Scenario generation

Two ways to build PiSAs scenarios:

| | What it is | Use it when |
|---|---|---|
| **`generate_scenarios.py`** | Expands one **task seed** into a folder of scenario bundles | You want a new task, or many scenarios of one task |
| **`JIRA_Allocation.ipynb`, `Meeting_Allocation.ipynb`** | The hand-authored notebooks behind two of the original tasks | You want to see how the first tasks were written |

The six seeds under [`seeds/`](seeds) are the ones the released tasks were generated from.

---

## From a seed to scenarios

```bash
pip install openpyxl      # only for the analysis workbook; --no-analysis-xlsx skips it

python generate_scenarios.py \
    --seed seeds/uas_flight_readiness/seed.json \
    --personal-context seeds/uas_flight_readiness/personal_context_library.json \
    --out-root generated/uas_flight_readiness
```

That writes `generated/uas_flight_readiness/scenario_01…30/`, each holding the four files
the harness reads, plus a `*_configuration_analysis.xlsx` workbook listing every valid
evidence configuration, its oracle outcome, its minimal sufficient evidence sets and
whether it was eligible for sampling.

Generation is deterministic: the same seed, `--random-seed` and options always produce the
same scenarios. Nothing is model-generated — every sentence comes from the seed's own
phrasings — so this step needs no API key and costs nothing.

Check a seed without generating anything:

```bash
python generate_scenarios.py --seed <seed.json> --personal-context <library.json> --health-only
```

### Reproducing a released task

The released bundles carry release names rather than the seed's internal family id. Pass
both names to reproduce them:

```bash
python generate_scenarios.py \
    --seed seeds/uas_flight_readiness/seed.json \
    --personal-context seeds/uas_flight_readiness/personal_context_library.json \
    --out-root generated/uas_flight_readiness \
    --id-prefix part107_flight_readiness_triage --task-type uas_flight_readiness
```

This reproduces `scenario.json`, `utility.json` and `appropriateness.json` of the released
task exactly. `visibility.json` comes out at its deterministic baseline (each fact visible
to its holder, and required evidence also to the executor) until you run the visibility
vote below, which is what produced the released labels.

`--id-prefix` changes only the ids written into the bundles, never which scenarios are
generated — those are keyed to the seed's `task_family_id`.

## Labelling visibility

Who may see a fact is not something the seed can state for every pair of people, so the
remaining pairs are labelled by a vote: for each (attribute, person) pair, the same
question is put three times to a model under three different privacy attitudes, and the
majority wins.

```bash
python generate_scenarios.py --visibility-only \
    --out-root generated/uas_flight_readiness \
    --visibility-model google/gemma-3-27b-it \
    --visibility-base-url http://localhost:8000/v1 \
    --visibility-api-key-env OPENAI_API_KEY \
    --visibility-workers 24
```

`--visibility-base-url` takes any OpenAI-compatible `/chat/completions` endpoint, and
several comma-separated URLs are used round-robin. For a local server that wants no
auth header, pass `--visibility-api-key-env ""`.

This step is the expensive one: three calls per unresolved (attribute, person) pair, which
is roughly 500 calls per scenario and ~15k for a 30-scenario task. The released labels were
produced with `google/gemma-3-27b-it`; agreement with human annotators is reported in the
paper's appendix.

## Writing your own seed

Start from the closest of the six published seeds and edit it — they are the reference
implementation, and `--health-only` will tell you what is wrong. A seed is one JSON file:

| Key | What it holds |
|---|---|
| `schema_version` | `"2.1"` |
| `task_family_id` | Identifier for the family; also seeds the generator's RNG |
| `task` | Type, domain, description, `executor_role`, and the brief the executor receives |
| `participants` | The executor, the other roles, and the `subject` the decision is about |
| `outcomes` | The allowed decisions |
| `evidence_variables` | The facts the decision turns on — see below |
| `decision_policy` | Ordered `rules`, first match wins, plus a prose `summary` |
| `decision_decoys` | Pressures that look like reasons but the real process does not admit |
| `generation_plan` | How many scenarios, outcome quotas, coverage cases, invalid configurations, how much personal context |
| `timeline` | How artifacts are laid out in time |

Each **evidence variable** has a holder and a set of **states**. Each state carries the
`canonical_fact` that state means, an `evaluation_probe`/`evaluation_target` pair used to
score it, and several **surface realizations** — the ways a person might actually say it:

- a `clean` realization says the fact and nothing more;
- a `privacy_entangled` realization fuses the fact with a private detail that the decision
  must not use, and carries both that `embedded_inappropriate_detail` and the
  `clean_rewrite` — the sanitized form that says the same usable thing without it.

That pairing is what makes the benchmark measurable: the clean rewrite is what a system
*should* surface, the embedded detail is what it must not.

`generation_plan.invalid_configurations` is how a task avoids scenarios where two
independent checks both fail — list the pairs that cannot co-occur, and every generated
scenario has exactly one reason for its outcome.

The **personal-context library** is a separate file of generic private facts about cast
members ("{person} is going through a divorce"). They are deliberately irrelevant to the
decision — that is what separates them from decoys — and a system should never let them
reach the task.

## Then evaluate

```bash
cd ../evaluation
python run_pipeline.py --scenarios-folder ../scenario_generation/generated/uas_flight_readiness \
    --results-path results/mine -s centralized --agent-llm openai/gpt-5.5
```

See the [main README](../README.md#bring-your-own-scenarios) for the full loop.
