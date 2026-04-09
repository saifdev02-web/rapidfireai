#!/usr/bin/env python3
"""Build local-Jupyter copies from Colab-oriented notebooks.

- Reads rf-colab-*.ipynb sources (unchanged on disk — those are the Colab originals).
- Writes rf-jupyter-*.ipynb next to them with Colab-specific APIs replaced.

Optional: pass --snapshot to also copy each Colab file under colab_originals/ (frozen duplicate).

Re-run after editing Colab notebooks:
  python tutorial_notebooks/_build_jupyter_local_notebooks.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PAIRS: list[tuple[str, str]] = [
    ("fine-tuning/rf-colab-tensorboard-tutorial.ipynb", "fine-tuning/rf-jupyter-tensorboard-tutorial.ipynb"),
    ("rag-contexteng/rf-colab-rag-fiqa-tutorial.ipynb", "rag-contexteng/rf-jupyter-rag-fiqa-tutorial.ipynb"),
    ("rag-contexteng/trackio/rf-colab-tutorial-rag-fiqa-trackio.ipynb", "rag-contexteng/trackio/rf-jupyter-tutorial-rag-fiqa-trackio.ipynb"),
    ("fine-tuning/trackio/rf-colab-tutorial-sft-trackio.ipynb", "fine-tuning/trackio/rf-jupyter-tutorial-sft-trackio.ipynb"),
]

RAY_DASHBOARD_CELL = """# Ray dashboard (local Jupyter). If the iframe is blank, open http://127.0.0.1:8855 in your browser.
from IPython.display import IFrame, display

display(IFrame(src="http://127.0.0.1:8855", width=950, height=600))
"""

END_EXPERIMENT_CELL = """import ipywidgets as widgets
from IPython.display import display


def _on_end_clicked(_b):
    experiment.end()
    print("Done!")
    end_btn.disabled = True
    end_btn.description = "Experiment ended"


end_btn = widgets.Button(description="Click to End Experiment", button_style="info")
end_btn.on_click(_on_end_clicked)
display(end_btn)
"""


def _split_lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return [line + "\n" for line in s.strip("\n").split("\n")]


def _cell_text(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, str):
        return src
    return "".join(src)


def _set_cell_source(cell: dict, text: str) -> None:
    lines = _split_lines(text)
    cell["source"] = lines


def _transform_cell(cell: dict) -> None:
    if cell.get("cell_type") != "code":
        return
    t = _cell_text(cell)
    if "from google.colab import output" not in t:
        return
    if "serve_kernel_port_as_iframe" in t:
        _set_cell_source(cell, RAY_DASHBOARD_CELL)
        return
    if "eval_js" in t and "experiment.end()" in t:
        _set_cell_source(cell, END_EXPERIMENT_CELL)
        return
    # Fallback: strip unreachable colab import if any other pattern appears
    t = t.replace("from google.colab import output\n", "")
    _set_cell_source(cell, t)


def _transform_markdown(cell: dict) -> None:
    if cell.get("cell_type") != "markdown":
        return
    src = _cell_text(cell)
    if "Open In Colab" in src and "colab.research.google.com" in src:
        _set_cell_source(
            cell,
            "### Local Jupyter\n\n"
            "This notebook is adapted to run in **Jupyter Lab / Notebook** on your machine. "
            "The Colab-oriented source is `rf-colab-*.ipynb` in the same folder (unchanged).\n\n"
            "Use a Python 3.12+ environment with RapidFire installed (`pip install -e .` from the repo root, or `pip install rapidfireai`). "
            "Start services with `./setup/fit/start_dev.sh start` when the tutorial expects the dispatcher / MLflow / UI.\n",
        )
        return

    replacements = [
        (
            "👉 <b>Note:</b> This Colab notebook illustrates simplified usage",
            "👉 <b>Note:</b> This Jupyter notebook illustrates simplified usage",
        ),
        ("This Colab notebook", "This notebook"),
        ("on Google Colab", "locally in Jupyter"),
        ("on Colab", "in this notebook"),
        ("**Runtime → Restart runtime**", "**Kernel → Restart kernel**"),
        ("Runtime → Restart runtime", "Kernel → Restart kernel"),
        ("the Colab notebook", "this notebook"),
        ("in the Colab notebook", "below"),
        ("Do not let the Colab notebook tab stay idle", "Avoid leaving the Jupyter server idle for long sessions"),
        ("Colab will disconnect", "your session may time out"),
        ("Colab’s GPU", "your GPU"),
        ("all-local Colab", "all-local (this machine)"),
        ("Starting them here would log to the wrong server", "duplicate services can confuse tracking"),
        ("run `rapidfireai start` on Colab", "run `rapidfireai start` in a terminal"),
        ("Interactive Controller panel UI for Colab", "Interactive Controller panel"),
        ("RapidFire AI also provides an Interactive Controller panel UI for Colab that lets you manage",
            "RapidFire AI also provides an Interactive Controller panel that lets you manage",
        ),
        ("# Display the Ray dashboard in the Colab notebook", "# Display the Ray dashboard"),
        ("# Display the Trackio dashboard in the Colab notebook", "# Display the Trackio dashboard"),
        ("from Colab, just rerun", "from protobuf issues, just rerun"),
        ("from Colab", "in this environment"),
    ]
    for old, new in replacements:
        src = src.replace(old, new)
    _set_cell_source(cell, src)


def _strip_colab_metadata(nb: dict) -> None:
    meta = nb.get("metadata", {})
    meta.pop("colab", None)
    if meta.get("accelerator"):
        meta.pop("accelerator", None)
    for cell in nb.get("cells", []):
        cm = cell.get("metadata")
        if isinstance(cm, dict):
            cm.pop("colab", None)


# Applied to every cell after markdown/code-specific transforms.
COLAB_SCRUBS: list[tuple[str, str]] = [
    ("### On your Mac (before Colab)\n", "### On your Mac (only if using ngrok / remote MLflow)\n"),
    ("MLflow on Colab", "MLflow on this machine"),
    ("services on Colab", "services in this notebook session"),
    ("Skipping rapidfire start on Colab", "Skipping rapidfire start in this notebook"),
    ("local Colab dispatcher", "local dispatcher"),
    ("from Colab, just rerun", "if you see protobuf issues, rerun"),
    ("within Colab's memory", "within available memory"),
    ("memory constraints in Colab", "memory constraints in this environment"),
    ("in Colab GPU memory", "in GPU memory"),
    ("keep this Colab run fast", "keep this tutorial run fast"),
    ("Colab-friendly protobuf", "notebook-friendly protobuf"),
    ("fit within Colab's", "fit within"),
    ("After you upload this notebook to Colab:", "Before running install:"),
    (
        "The Controller uses ipywidgets and is compatible with both Colab (ipywidgets 7.x) and Jupyter (ipywidgets 8.x).\n",
        "The Controller uses ipywidgets and works in Jupyter Lab / Notebook (ipywidgets 8.x); Google Colab is also supported (ipywidgets 7.x).\n",
    ),
    ("# Display the Trackio dashboard in the Colab notebook\n", "# Display the Trackio dashboard\n"),
    ("In this beginner-friendly Colab,", "In this beginner-friendly tutorial,"),
    ("### Option 2: Install in Google Colab\n", "### Option 2: Install from this notebook\n"),
    (
        "For simplicity, you can run this notebook on Google Colab. This notebook is configured to run end-to-end on Colab with no local installation required.",
        "Use the cells below in order; the same `pip install` flow works in Jupyter Lab or Notebook on your machine.",
    ),
    ("rather than Google Colab.", "for day-to-day work."),
    ('experiment_name="exp1-fiqa-rag-colab"', 'experiment_name="exp1-fiqa-rag-jupyter"'),
    ('run_name="colab_minimal_ping"', 'run_name="jupyter_minimal_ping"'),
    ("run 'colab_minimal_ping'", "run 'jupyter_minimal_ping'"),
]


def _scrub_colab_mentions(cell: dict) -> None:
    if cell.get("cell_type") not in ("markdown", "code", "raw"):
        return
    src = _cell_text(cell)
    for old, new in COLAB_SCRUBS:
        src = src.replace(old, new)
    _set_cell_source(cell, src)


def build_one(colab_rel: str, jupyter_rel: str, *, snapshot: bool) -> None:
    colab_path = ROOT / colab_rel
    jupyter_path = ROOT / jupyter_rel
    if snapshot:
        backup_root = ROOT / "colab_originals"
        backup_path = backup_root / colab_rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(colab_path, backup_path)
        print(f"Snapshot: {backup_path.relative_to(ROOT.parent)}")

    with open(colab_path, encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        _transform_markdown(cell)
        _transform_cell(cell)
        _scrub_colab_mentions(cell)

    _strip_colab_metadata(nb)

    jupyter_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jupyter_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {jupyter_path.relative_to(ROOT.parent)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--snapshot",
        action="store_true",
        help="Also copy rf-colab-*.ipynb into tutorial_notebooks/colab_originals/ (optional duplicate).",
    )
    args = ap.parse_args()
    for colab_rel, jupyter_rel in PAIRS:
        build_one(colab_rel, jupyter_rel, snapshot=args.snapshot)
    print("Done.")


if __name__ == "__main__":
    main()
