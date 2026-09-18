# Scenario generation

Everything that produced a PiSAs task lives here: two generators, the seeds they read, and
the notebooks that walk through how the earlier tasks were built.

```text
scenario_generation/
├── generate_scenarios.py       # task-family seed → scenario bundles   (the six newer tasks)
├── enriched_pipeline.py        # enriched seed    → scenario bundles   (severity_classification)
├── personal_context_library.json
├── seeds/                      # every published seed, one folder per task
└── notebooks/                  # annotated walkthroughs of the three hand-built tasks
```

## Which generator produced which task

| Task | Seed | Generator |
|---|---|---|
| `inpatient_discharge` | `seeds/inpatient_discharge/` | `generate_scenarios.py` |
| `thesis_readiness` | `seeds/thesis_readiness/` | `generate_scenarios.py` |
| `manuscript_submission` | `seeds/manuscript_submission/` | `generate_scenarios.py` |
| `uas_flight_readiness` | `seeds/uas_flight_readiness/` | `generate_scenarios.py` |
| `outgoing_museum_loan` | `seeds/outgoing_museum_loan/` | `generate_scenarios.py` |
| `special_event_permit_readiness` | `seeds/special_event_permit_readiness/` | `generate_scenarios.py` |
| `severity_classification` | `seeds/severity_classification/` | `enriched_pipeline.py` — walkthrough in `notebooks/Severity_Classification.ipynb` |
| `JIRA_allocation` | in-notebook config | `notebooks/JIRA_Allocation.ipynb` |
| `meeting_allocation` | in-notebook config | `notebooks/Meeting_Allocation.ipynb` |

The two seed formats are not interchangeable — a task-family seed describes a whole family and
is expanded deterministically, an enriched seed describes one task and its value profiles and is
materialized by an LLM. Point either generator at the other's seed and it will say so.

```bash
pip install -r requirements.txt
```

---

## Task-family seeds → scenarios

```bash
python generate_scenarios.py \
    --seed seeds/uas_flight_readiness/seed.json \
    --personal-context seeds/uas_flight_readiness/personal_context_library.json \
    --out-root generated/uas_flight_readiness
```

That writes `generated/uas_flight_readiness/scenario_01…30/`, each holding the four files the
harness reads, plus a `*_configuration_analysis.xlsx` workbook listing every valid evidence
configuration, its oracle outcome, its minimal sufficient evidence sets and whether it was
eligible for sampling.

Generation is deterministic: the same seed, `--random-seed` and options always produce the same
scenarios. Nothing is model-generated — every sentence comes from the seed's own phrasings — so
this step needs no API key and costs nothing.

Check a seed without generating anything:

```bash
python generate_scenarios.py --seed <seed.json> --personal-context <library.json> --health-only
```

This prints every valid evidence configuration, its oracle outcome, whether any configuration
matches two decision rules, the outcome quotas it can actually fill, and any structural error.

### Reproducing a released task

The released bundles carry release names rather than the seed's internal family id. Pass both
names to reproduce them:

```bash
python generate_scenarios.py \
    --seed seeds/uas_flight_readiness/seed.json \
    --personal-context seeds/uas_flight_readiness/personal_context_library.json \
    --out-root generated/uas_flight_readiness \
    --id-prefix part107_flight_readiness_triage --task-type uas_flight_readiness
```

This reproduces `scenario.json`, `utility.json` and `appropriateness.json` of the released task
exactly. `visibility.json` comes out at its deterministic baseline (each fact visible to its
holder, and required evidence also to the executor) until you run the visibility vote below,
which is what produced the released labels.

`--id-prefix` changes only the ids written into the bundles, never which scenarios are
generated — those are keyed to the seed's `task_family_id`.

### Labelling visibility

Who may see a fact is not something the seed can state for every pair of people, so the
remaining pairs are labelled by a vote: for each (attribute, person) pair, the same question is
put three times to a model under three different privacy attitudes, and the majority wins.

```bash
python generate_scenarios.py --visibility-only \
    --out-root generated/uas_flight_readiness \
    --visibility-model google/gemma-3-27b-it \
    --visibility-base-url http://localhost:8000/v1 \
    --visibility-workers 24
```

`--visibility-base-url` takes any OpenAI-compatible `/chat/completions` endpoint, and several
comma-separated URLs are used round-robin. For a server that wants no auth header, pass
`--visibility-api-key-env ""`. Budget roughly 500 calls per scenario.

## Enriched seeds → scenarios

`enriched_pipeline.py` runs three stages — assign the cast, have a model write each artifact's
text from a blueprint, then assemble the four files — and writes one bundle per value profile in
the seed:

```bash
export OPENROUTER_API_KEY=...
python enriched_pipeline.py seeds/severity_classification/seed.json -o ./output
python enriched_pipeline.py seeds/severity_classification/seed.json -o ./output --profile all --model openai/gpt-4o
```

`--api-base` points it at any other OpenAI-compatible server. Cast assignment is deterministic
given `--seed-random`; the artifact text is not, because a model writes it.

`notebooks/Severity_Classification.ipynb` walks through the same three stages cell by cell,
including the critic panel that votes on candidate trap phrasings, if you want to see or change
how a stage works before running the CLI.

## The notebook generators

`notebooks/JIRA_Allocation.ipynb` and `notebooks/Meeting_Allocation.ipynb` produced the two
hand-built tasks. They are self-contained: the configuration lives in an in-notebook `GEN_CONFIG`
dict rather than a seed file, and both are *backward-built* — plant a public ambiguity first,
then add the decision-critical private constraint that resolves it. They are the clearest place
to see how a constraint, its sanitized form and its visibility annotation line up.

## Writing your own seed

Start from the closest of the published task-family seeds and edit it — they are the reference
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
`canonical_fact` that state means, an `evaluation_probe`/`evaluation_target` pair used to score
it, and several **surface realizations** — the ways a person might actually say it:

- a `clean` realization says the fact and nothing more;
- a `privacy_entangled` realization fuses the fact with a private detail that the decision must
  not use, and carries both that `embedded_inappropriate_detail` and the `clean_rewrite` — the
  sanitized form that says the same usable thing without it.

That pairing is what makes the benchmark measurable: the clean rewrite is what a system *should*
surface, the embedded detail is what it must not.

`generation_plan.invalid_configurations` is how a task avoids scenarios where two independent
checks both fail — list the pairs that cannot co-occur, and every generated scenario has exactly
one reason for its outcome.

The **personal-context library** is a separate file of generic private facts about cast members
("{person} is going through a divorce"). They are deliberately irrelevant to the decision — that
is what separates them from decoys — and a system should never let them reach the task.

## Then evaluate

```bash
cd ../evaluation
python validate_scenarios.py ../scenario_generation/generated/uas_flight_readiness
python run_benchmark.py --scenarios-folder ../scenario_generation/generated/uas_flight_readiness \
    --agent-llm openai/gpt-5.5
```

See the [main README](../README.md#4-evaluate-your-own-scenarios) for the full loop.
