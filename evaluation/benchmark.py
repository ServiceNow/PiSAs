"""
PiSAs benchmark resolution — download the dataset from Hugging Face and locate tasks.

The dataset is public, so no token and no login are needed: the first call downloads
it into the standard Hugging Face cache (``$HF_HOME``/``~/.cache/huggingface``) and
every later call reuses that copy.

Layout of the dataset repo::

    <root>/<task_name>/scenario_NN/{scenario,utility,visibility,appropriateness}.json

Task names are read from the downloaded snapshot rather than hard-coded here, so a
task added to the dataset is usable without changing this code.
"""

import os
import sys
from pathlib import Path

# Hugging Face dataset holding the full PiSAs benchmark (public, no token required).
PISAS_HF_REPO = os.environ.get("PISAS_HF_REPO", "ServiceNow/PiSAs")

_HF_DOWNLOAD_HELP = (
    "Could not download the PiSAs benchmark from Hugging Face "
    f"({PISAS_HF_REPO}).\n"
    "The dataset is public, so no token is needed — this is usually a network or proxy issue.\n"
    "  • check connectivity to https://huggingface.co\n"
    "  • behind a proxy, set HTTPS_PROXY\n"
    "  • offline, point the harness at a local copy instead: --scenarios-folder <path>\n"
    "  • a token is only needed if you mirrored the dataset into a private repo; then set\n"
    "    HF_TOKEN=hf_xxx (or run `hf auth login`) and PISAS_HF_REPO=<your/repo>"
)


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def download_benchmark(repo_id: str = None, revision: str = None, quiet: bool = False) -> Path:
    """Download (or reuse the cached copy of) the PiSAs dataset; return its root path."""
    repo_id = repo_id or PISAS_HF_REPO
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _fail("huggingface_hub is not installed — pip install -r requirements.txt")
    if not quiet:
        print(f"  Downloading the PiSAs benchmark from Hugging Face ({repo_id})…")
    try:
        # token=False would *forbid* a token; leaving it unset lets an existing token be
        # used when the repo is a private mirror, without ever requiring one.
        return Path(snapshot_download(repo_id, repo_type="dataset", revision=revision))
    except Exception as e:
        _fail(f"{_HF_DOWNLOAD_HELP}\n  Underlying error: {e}")


def list_tasks(root: Path) -> list:
    """Task names available under a benchmark root (directories holding scenario folders)."""
    root = Path(root)
    tasks = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if any((s / "scenario.json").exists() for s in d.iterdir() if s.is_dir()):
            tasks.append(d.name)
    return tasks


def resolve_task(task: str, repo_id: str = None, revision: str = None, quiet: bool = False) -> Path:
    """Return the local folder holding every scenario of ``task``, downloading if needed."""
    root = download_benchmark(repo_id=repo_id, revision=revision, quiet=quiet)
    task_dir = root / task
    if not task_dir.is_dir():
        available = list_tasks(root)
        _fail(f"task {task!r} is not in the benchmark at {root}.\n"
              f"  Available tasks: {', '.join(available) if available else '(none found)'}")
    return task_dir


def print_tasks(repo_id: str = None, revision: str = None) -> None:
    """`--list-tasks` helper: print the tasks in the dataset with their scenario counts."""
    root = download_benchmark(repo_id=repo_id, revision=revision)
    print(f"\n  PiSAs benchmark at {root}\n")
    for name in list_tasks(root):
        n = sum(1 for s in (root / name).iterdir() if s.is_dir() and (s / "scenario.json").exists())
        print(f"    {name:<32} {n:>3} scenarios")
    print()
