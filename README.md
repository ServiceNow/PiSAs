# PiSAs
Benchmarking Contextual Integrity in Multi-User Agentic Systems.

PiSAs (Privacy in Shared Agentic systems) is a benchmark for evaluating unintentional privacy leakage in multi-user agentic systems. Each scenario carries dual contextual-integrity annotations — whether an attribute is appropriate for the task, and which users may legitimately access it — making cross-user spillage measurable across outputs, inter-agent communication, and memory.

- **[`scenario_generation/`](scenario_generation/)** — notebooks that generate the benchmark scenarios (`JIRA_Allocation.ipynb`, `Meeting_Allocation.ipynb`).
- **[`evaluation/`](evaluation/)** — harness that runs the scenarios through the agent systems and scores privacy, completeness, and utility. See [`evaluation/README.md`](evaluation/README.md) for details.
