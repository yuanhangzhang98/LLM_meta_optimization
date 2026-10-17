# Reproduce the revised paper

These tools check existing measurements and regenerate Figure 3. They do not
call an LLM or run solver experiments. The generic framework and original
solver/objective files remain at the repository root.

## Get the recorded data

Download `supporting_data.zip` from the
[paper-2026-10-17 release](https://github.com/yuanhangzhang98/LLM_meta_optimization/releases/tag/paper-2026-10-17)
and save it in `reproduce/artifacts/`. From the repository root, run:

```bash
python -m zipfile -e reproduce/artifacts/supporting_data.zip reproduce/artifacts
python reproduce/artifacts/data/verify.py --checksums
```

The verifier uses only the Python standard library. It checks the source and
data checksums, 35 pooled benchmark medians from 17,600 per-instance records,
and the eight additional search runs. It also verifies availability of the
414 original solver sources and 42 objective functions.

The release assets are:

| Asset | Contents |
|---|---|
| `supporting_data.zip` | Campaign records, follow-up runs, raw benchmarks, figure inputs, and verification scripts |
| `audit_logs.zip` | Optional historical decision logs and replay tools |
| `core.zip` | Optional frozen, minimal framework snapshot for historical reproduction |
| `supplements.zip` | All three components above, with `core/`, `data/`, and `logs/` as siblings |
| `manuscript.pdf`, `manuscript_source.zip` | Corresponding public manuscript and complete LaTeX source |
| `SHA256SUMS` | SHA-256 hashes of the release assets |

To inspect historical decision logs or reconstruct an experiment workspace,
extract `supplements.zip` into a separate directory and follow its component
READMEs. The frozen `core/` is an optional reproduction snapshot; use the
repository root for the maintained framework.

## Recompute the objective analyses

Use Python 3.9 or later and install the offline dependencies:

```bash
python -m pip install -r reproduce/requirements.txt
python reproduce/analyze_campaign.py --output reproduce/output/consensus.json
python reproduce/analyze_objective0.py --output reproduce/output/objective0.json
```

The first command evaluates the 42 archived objectives on all 414 designs using
the repository's consensus implementation. It reports raw consensus and the
final recorded meta-agent multipliers. The second checks the fixed expert
objective's cached scores and decomposes its extrapolation behavior, including
design 247's small-size fit and its rank relative to design 340.

## Regenerate Figure 3

Install Graphviz so that `dot` is on your PATH, then run:

```bash
python reproduce/restore_design_figure.py --output reproduce/output/figure3.pdf --summary reproduce/output/figure3.json
```

The figure uses original experimental and genealogy records. Design 340 has
one fit to its 16 original and three fresh measurements; only the fresh points
have error bars. The fitted exponents are approximately 2.51 for the original
baseline and 1.46 for design 340. These lines are reference fits; the largest
sizes bend above them. Graphviz versions can change node placement.

All tools locate their default data relative to this directory. Use
`--campaign`, `--data-root`, or `--data-dir`, respectively, to read an extraction
elsewhere. Outputs go to the paths you specify. The original plotting scripts
under `scripts/` remain available for their original campaign workflow.

## Keep solver identities tied to their runs

Solver IDs are local to each search run:

| Solver discussed in the paper | Path inside the extracted `data/` |
|---|---|
| Consensus design 340 | `campaign/solvers/solver_340.py` |
| Fixed-expert design 89 | `runs/expert_a/solvers/solver_89.py` |
| Tuned baseline 9001 | `benchmarks/hebo_baseline/solvers_work/solver_9001.py` |

The existing repository file `solvers/solver_89.py` is design 89 from the
original campaign. It is a different solver from the fixed-expert design 89.
Keep the run directories separate when inspecting or executing these files.

## Environments and provenance

`requirements.txt` here contains the small offline dependency set. The root
`requirements.txt` is for discovery runs. `requirements-historical.txt` records
the retained Python 3.9.13 experiment environment, including PyTorch 2.6.0;
select the appropriate CPU/CUDA wheel separately when reconstructing it.
Installing analysis dependencies does not require PyTorch, HEBO, an API key,
or a GPU.

The offline commands were checked in a fresh Python 3.11 environment. Discovery
dependencies were resolved separately for Python 3.10; a fresh GPU runtime
installation and end-to-end discovery run were not performed for this release.

`provenance.json` records the source of these reproduction tools. The downloaded
data include per-file provenance, checksums, and the original instance-generation
settings. Large generated instance tensors are omitted, but their generator
and content hashes are retained. No new scientific experiments were run to
prepare this release.

The original public code is tagged `paper-original` at commit `6ee37ae`.
The revised paper and evidence are tagged `paper-2026-10-17`. Public framework
files retain the repository's named MIT attribution; third-party licenses are
preserved in the supporting data.
