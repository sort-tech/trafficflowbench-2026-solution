# TrafficFlowBench 2026: team sortdev code package

This package rebuilds both final submissions of team sortdev from the public Kaggle release.
It uses no external data, no pretrained models and no paid services.

- Kaggle team: sortdev (1 member, Jiwon Park)
- Report: `report/sortdev_report.pdf`
- Licence: MIT (`LICENSE`). `trafficflowbench-public/` is the organisers' repository, unchanged, under its own MIT licence.

## Result

| | Public LB | Private LB |
|---|---|---|
| Final A | 0.88376 | 0.86748 |
| Final B | 0.88324 | 0.86725 |

Final standing: 130th of 802 teams. [`POSTMORTEM.md`](POSTMORTEM.md) compares all 56 submissions on both leaderboards and lists how this public copy differs from the package sent to the organisers.

## 1. Quick start

```bash
PYTHON312=/path/to/python3.12 ./setup_env.sh
TFB_REL=/path/to/kaggle_public final/run_all.sh out
```

`TFB_REL` is the unpacked Kaggle data folder: the one that holds `submission_key.csv` and the `train/`, `validation/` and `private/` panels.
The run writes these files:

| File | What it is |
|---|---|
| `out/submission_A.csv` | final choice A |
| `out/submission_B.csv` | final choice B |
| `out/task1/`, `out/task2_A/`, `out/task2_B/`, `out/task4/` | per-task files |
| `out/logs/` | per-step logs with wall time and peak memory, and md5 of every output (`run_all.log`) |

A and B share Task 1 and Task 4 and differ only in Task 2. The configuration is set at the top of `final/run_all.sh`.

## 2. What runs

| Order | Step | Command inside `run_all.sh` | Wall time | Peak memory |
|---|---|---|---|---|
| 1 | Task 4 ODME: NNLS with each split's own counts (`nnls_own`) | `task4/build.py` | 3–6 min | 1 GB |
| 2 | Task 2 LightGBM models (every model trained from the release in an empty folder) | `task2/repro_g5.sh` | 40–45 min | 8 GB |
| 3 | Task 2 network blends for A and B (inference with the shipped weights, run once per candidate) | `task2_nn/repro_final.sh` | 5–10 min | 2.5 GB |
| 4 | Task 1 state (Task 3 is scored on these rows): two-stage LightGBM, gap specialists, self-supervised correction | `task1/build.py` | 6.5 h (M4 Pro, 10 threads) to 13.5 h (Linux aarch64, 16 threads, shared machine) | 21 GB |
| 5 | merge + format check | `trafficflowbench-public/src/merge_submissions.py`, `submit/check_submission.py` | 1 min | 6 GB |

We recommend 32 GB RAM and about 60 GB of free disk for caches and datasets.
Task 1 uses `TFB_THREADS` threads (default 8).

### Final configurations

| | Task 1 | Task 2 | Task 4 | Public LB |
|---|---|---|---|---|
| A | v7 | i5 ongoing probabilities mixed 0.5 with a 3-seed CNN, with a climatology gate (`cpu3g_go50`) | nnls_own | 0.88376 |
| B | v7 | the same blend with network weight 0.7 (`cpu3g_go70`) | nnls_own | 0.88324 |

i5 (`lgb_onthr_sq31_pao1535`, LightGBM only, 0.88019) is the base of both blends; `task2/repro_g5.sh` writes it to `out/task2_build/candidates/`.

Expected md5 of the Task 2 files: A `d7233f01b47c9f71bc36ed3dfd942ef5`, B `08dbae5412257a7e0ed30748bbe9e9c5`.
Task 2 and Task 4 are byte-identical across macOS and Linux (Task 4 differs in the last float digit on Linux).
Task 1 agrees with the submitted file at score level, not byte level: its submitted models were trained on two machine types, and LightGBM paths differ slightly between macOS and Linux. See the next section.

## 3. Reproducibility

- Every seed is fixed in code:
  - Task 1 stage-2 seeds are 0, 11 and 23.
  - Task 2 LightGBM seeds are set in `task2/model.py`.
  - The network seeds are 20261101–20261103.
  - LightGBM runs with `deterministic=True, force_row_wise=True`, and boosting rounds are fixed, not early-stopped.
- Rule check for Task 2:
  - Every Task 2 input uses only data stamped at or before the window's forecast origin T. Train-only tables (climatology, thresholds, models) are fitted on train.
  - We tested this for all 160 validation and private windows. For each window we deleted every released row stamped after T (337.8 M rows in all) and predicted again.
  - The LightGBM part (`task2/trunc_test.py`) changed 0 of 174,000 cells.
  - For the network blends (`task2_nn/build_mix.py --trunc`, `task2_nn/trunc_final.sh` for A and B), the network input arrays rebuilt from the truncated data were equal to the arrays used for the submission, and 0 cells changed.
  - Functions that read split-wide statistics are set to fail if called during the test.
- Network weights:
  - `task2_nn/weights/cpu_r{1,2,3}` were trained on CPU in deterministic mode (`task2_nn/train.py`).
  - `NN_TRAIN=1 final/run_all.sh out` retrains them, taking about 4 h 45 min.
  - Retraining reproduces the weights bit for bit only on aarch64 Linux with torch 2.14.1. On other machines training diverges from step ~500, and the result agrees at cell level only.
- Runs from code-only copies:
  - We ran the whole pipeline on macOS arm64 (M4 Pro) and Linux aarch64 (20 cores). Task 2 and Task 4 were also checked on Windows x86, and Task 2 on macOS arm64 (M4).
  - `final/REPRODUCTION_LOG.md` gives the md5 values and cell-level differences.
  - On the other machine types, Task 1 differed from the submitted file by a mean of 0.002–0.095 km/h and 0.2–7.3 veh/h. That is far inside the 1 % reproduction rule.

## 4. Folder map

| Folder | Content |
|---|---|
| `final/` | `run_all.sh` (single entry point) and the reproduction log |
| `task1/` | Task 1 builder `build.py` and its stages (cache, datasets, stage 1/2 LightGBM, gap specialists, self-supervised correction) |
| `task2/` | Task 2 window tables, datasets, LightGBM families, mixing and set selection (`build.py`); `repro_g5.sh` runs all of it in an empty folder |
| `task2_nn/` | spatio-temporal CNN (`train.py`), release inference (`rel_predict.py`), gated blend (`build_mix.py`), shipped weights |
| `task4/` | regularised NNLS path-flow solver |
| `submit/check_submission.py` | format check of the upload file |
| `trafficflowbench-public/` | organiser code (MIT): Task 3 evaluator used for metrics, `merge_submissions.py` |


## 5. External resources

- Data: only the competition release.
- Software: open-source Python packages (pinned in `requirements.txt` and `requirements_nn.txt`).
- We used no pretrained models, external datasets or paid APIs.
- We used an AI coding assistant (Claude Code, Anthropic) during development. It is freely or cheaply available, as the rules allow. The pipeline itself contains no language model; see report Section 6.3.
