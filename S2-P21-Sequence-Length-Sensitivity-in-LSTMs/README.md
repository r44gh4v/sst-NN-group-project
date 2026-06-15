# S2-P21 - Sequence Length Sensitivity in LSTMs

Section 2 / Task 2 of the Scaler DL-I final project. The combined report (PDF)
and slide deck covering this task **and** S1-P11 live under the `S1-P11` folder,
per the project guide.

## Task

> Train an LSTM on sequences of length 10, 50, and 200. Measure accuracy and
> training time. Show how performance changes with sequence length.

## What's here

| File | Purpose |
|---|---|
| `s2-p21.ipynb` | The deliverable notebook. Runs top-to-bottom. **Executed locally on CPU** (Python 3.12, torch 2.12+cpu) with all outputs/figures saved. |
| `build_notebook.py` | Regenerates `s2-p21.ipynb` *unexecuted* and deterministically (stdlib only - `python build_notebook.py`). Edit the experiment here, then re-run. **Note: running it overwrites the executed notebook**, so re-execute afterwards. |
| `requirements.txt` | Runtime deps for the notebook. |

## Approach (in one paragraph)

A synthetic **delayed-signal recall** task isolates *memory over distance* from
everything else: a clean `+1`/`-1` class marker sits at one known timestep, the
rest of the sequence is `N(0,1)` noise, and the model classifies from the final
timestep. Four controlled experiments: (1) **length sweep**
`[10,20,30,50,100,200]` × 5 seeds - accuracy & training time vs length;
(2) **distance control** at fixed `L=200` - marker at start vs end; (3) **LSTM
vs vanilla RNN** across the sweep; (4) **forget-gate bias** - re-run the LSTM
with forget bias `+2` to test whether the horizon is a capacity limit or an
initialization artifact. Single `CONFIG` dict, `set_seed`, multi-seed error
bars, pandas tables, labelled plots.

## Headline findings

- Accuracy shows a **memory horizon at L≈30** (≈1.0 up to L=20, knee at L=30 with
  a wide error bar, chance by L≥50 at the fixed 30-epoch budget).
- Training time scales **~linearly** with length: ~5.1 s (L=10) → ~39 s (L=200),
  ≈7.6× on CPU.
- **Distance, not length, is the cause** - at L=200, marker-at-end = 1.000 vs
  marker-at-start = 0.502 (chance). Same length, same compute. The key insight.
- **The *default* LSTM is beaten by a vanilla RNN** past the knee - because
  PyTorch inits the forget-gate bias to 0 (cell state halves every step).
- **One bias fixes it:** forget-gate bias `+2` gives **1.000 at every length up
  to 200, zero variance** - the "horizon" was an initialization artifact, not a
  capacity limit.

## How to run

The notebook already ships **executed** (CPU, ~30 min total). To re-run:

**Local (how it was produced):**
```
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe numpy pandas matplotlib seaborn jupyterlab ipykernel
.venv/Scripts/jupyter-lab.exe   # open s2-p21.ipynb, Run All
```

**Colab:** open `s2-p21.ipynb`, select a kernel (GPU optional - timing is cleaner
on CPU; device is auto-detected and reported), Runtime → **Run all**, save with
outputs.
