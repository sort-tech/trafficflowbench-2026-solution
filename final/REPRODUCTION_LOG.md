# Final package (team sortdev)

Rebuilds the selected final submission from the public release only.

```bash
TFB_REL=/path/to/kaggle_public final/run_all.sh [out_dir]
```

- Python 3.12, numpy 2.x, pandas 3.x, pyarrow, scipy, scikit-learn, LightGBM 4.7 (CPU), polars, duckdb. No external data, no pretrained models, no paid APIs.
- Seeds are fixed inside each builder; LightGBM runs with `deterministic=True`, `force_row_wise=True`.
- Selected configuration is at the top of `run_all.sh` (Task 4 `nnls_own`, Task 2 variant, Task 1 config).

## Reproduction log

Clean copy (code only, no caches, models or candidates) in `KAGGLE/repro/`, run from scratch on macOS 15 / Apple M4 (10 cores, 24 GB), shared with other jobs. (10/31: the rebuilt outputs under `repro/` were cleared to free disk; the logs and run scripts stay, and the md5 results are recorded below.)

| Task | Command | Wall time | CPU time | Peak memory | Result vs candidate |
|---|---|---|---|---|---|
| 4 | `build.py --out final` | 6 min 03 s | 13 min 17 s | 0.9 GB | md5 identical (`31c6dff6…`) |
| 2 | `build.py --split all --fresh --onset-variant aug --ongoing-variant aug` | 21 min 27 s | 33 min | 8.0 GB RSS (14.3 GB footprint) | md5 identical (`471aea49…`) |
| 1 | `TFB_THREADS=8 OMP_NUM_THREADS=8 build.py --stage2 --retrain --split validation,private` | 3 h 53 min (8 threads) | 21 h | 10.5 GB RSS | not byte-identical to the 16-thread GB10 candidate (early stopping differs); see notes |

Notes
- The package must ship `trafficflowbench-public/` (organiser code, MIT) next to `task1/`: Task 1 imports `task3/score_task3.py` from it for its metrics, and `merge_submissions.py` builds the upload file. Found on 10/26 when the first clean-copy Task 1 run stopped after 4 minutes without it.
- Task 1 from a code-only copy failed after 4 h 20 min (10/27 00:25, Mac 8 threads, peak 11.6 GB RSS) because `stage2.py` writes to `preds/` without creating it. Fix requested in task1 (create every output directory). Rerun with the directories pre-created: finished 10/27 04:19, 3 h 53 min wall, 21 h CPU, peak 10.5 GB RSS. Versus the GB10 candidate gbm_v8s2 (16 threads): speed diff RMS 0.34 km/h (max 19.8), flow diff RMS 39 vph (max 5054); similar in size to the gbm_v8s2 → x3g change (≈0.0007 on the proxy). LB check: r1_repro8t_lgb1_nnls vs c8.
- Thread count changes Task 1 early stopping (stage-2 speed model: 3348 rounds on 8 threads vs 3983 on 16). Final package should use fixed round counts (requested 10/27) so the result depends on thread count only at floating-point level; the 8-thread early-stopping rebuild will be scored on the LB once to measure the effect directly.
- Task 2 needs about 16 GB RAM when building its datasets from scratch; training alone stays under 4 GB.
- x86 check (10/26, Windows 10 x64, i7-10870H, Python 3.13.2 with numpy 2.5.4 / pandas 3.1.0 / scipy 1.18.1): Task 4 rebuilt from scratch. Bytes differ (CRLF line endings, and last-digit floating-point differences between ARM and x86), values agree on all 70,708 paths: max abs diff 2.0e-11, relative L1 3.6e-15. Task 2 (lgb_aug, `--fresh`, data downloaded on the laptop): 174,000 rows, 53 cells (0.03 %) differ from the Mac file, positives 10,781 vs 10,782 — threshold-edge cells flipped by ARM/x86 floating point; estimated effect ≤ 0.001 on S_total, an order of magnitude inside the 1 % rule. Task 1 not yet checked on x86 (≈12 GB RAM).
- `run_all.sh` picks `/usr/bin/time -l` on macOS and `-v` on Linux.

## 10-31 final-config run on GB10 (Linux aarch64, 20 cores, bash, 16 threads)

`final/run_all.sh` (v7 + h5 + nnls_own) from a code-only copy in `~/work/tfb/final_v7h5`, same library versions as the Mac.
- Task 4 nnls_own: 3 min. md5 differs from the Mac file (d9749a4b… vs 31c6dff6…) only in float last digits: max |Δ path_flow| 1.6e-11.
- Task 2: 30 min, g5 md5 1aa5a72e… and h5 md5 68941838… — byte-identical to the Mac candidates. (First try stopped at `v1_thr.py fit` because `models/cv/` did not exist in an empty folder; `repro_g5.sh` now creates it.)
- Task 1 v7: 13 h 37 min (19:07 → 08:43), peak 20.2 GB RSS, exit 0. md5 48537a3a… vs the submitted v7 c0a8ba04…: 127,876 of 6,735,795 cells differ (1.9 %, gap specialists retrained on Linux instead of the Mac files), mean |Δspeed| 0.0020 km/h (p99 0.06, max 2.6), mean |Δflow| 0.19 vph (p99 5.5, max 4009). That is about 1 % of the v7 − ss3f difference (0.174 km/h / 25.3 vph), which moved the public LB by +0.00029, so the score gap is far inside the 1 % rule.
- Merged file md5 36b6fbd8…, format check OK (6,980,503 rows). Whole run 14 h 10 min. Outputs copied to `repro/final_v7h5_gb10/`.

## 10-30 Task 1 v7 run on the Mac mini (macOS arm64, M4 Pro, 10 threads)

Same v7 code and settings in `~/tfb/task1_v7` (resumed twice in the first 20 min, 18:38 → 01:02, about 6 h 25 min, peak 21.1 GB RSS). Fetched on 11-02 to `repro/v7_macmini/`.
- md5 12db9f5b… vs the submitted v7 c0a8ba04…. Already the stage-1 model files differ (e.g. `s1_ds13/p0_y_k.txt` 8f8261c1… vs d4b6266a…), so LightGBM on macOS takes a slightly different path from the first stage on; the GB10 run did not.
- 99 % of cells differ, but by little: mean |Δspeed| 0.094 km/h (p99 0.65, max 18.9), mean |Δflow| 7.2 vph (p99 66, max 3811). Same size in R1/R2/R3. Against the GB10 repro the numbers are the same (0.095 km/h, 7.3 vph).
- That is about half of the v7 − ss3f difference (0.174 km/h / 25.3 vph, public LB +0.00029), so the expected score gap is of order 0.0002–0.0003, far inside the 1 % rule (about 0.0088). Byte-level reproduction holds on Linux (GB10); on macOS it is score-level only.

## 11-02 two final configurations in one run

`final/run_all.sh` now writes both final choices from a single run, since they share Task 1 (v7) and Task 4 (nnls_own) and differ only in Task 2:
- `submission_A.csv`: Task 2 = i5 (`lgb_onthr_sq31_pao1535`, 2b9a602f…).
- `submission_B.csv`: Task 2 = i5 + split climatology for onset (`lgb_i5_sclim75p`, c5de3558…). Validation rows equal A; 7 private onset windows differ.
- Merge step tested on the submitted per-task files: submission_A md5 d08c0db0… equals the submitted i5 file; submission_B 45476368….
- `task2/repro_g5.sh` builds g5, h5, i5 and sclim75p from an empty folder. Mac mini (macOS) run on 11-02: all four md5 equal the submitted files (37.7 min).


## 11-03 GB10 run of both final configurations (code-only copy `~/work/tfb/final_ab`)
Started 11/2 12:09 (16 threads, external SSH), done 11/3 02:30: Task 4 3 min, Task 2 44 min, Task 1 13 h 34 min, merge 1 min.
- Task 2 A `lgb_onthr_sq31_pao1535`: 2b9a602f… = submitted (byte-identical). Task 2 B `lgb_i5_sclim75p`: c5de3558… = submitted (byte-identical).
- Task 4 nnls_own: d9749a4b… (same as the 10/30 GB10 run); vs the Mac file 31c6dff6… max |Δ path_flow| 1.6e-11.
- Task 1 v7: 48537a3a… vs submitted c0a8ba04…: 1.88 % of speed cells and 1.90 % of flow cells differ, mean |Δ| 0.0020 km/h and 0.19 vph (p99 0.06 km/h, 5.5 vph). Smaller than the Mac mini difference by about 50×; score effect negligible.
- submission_A df6aed73…, submission_B 7c99a3bd… (both pass check_submission.py).
- Copies of the GB10 Task 1 and Task 4 files: `repro/gb10_final_ab/`.
- Mac mini run of the same configuration (`~/tfb/final_ab`, 11/2 13:06–20:52, collected 11/4): Task 2 A 2b9a602f… and B c5de3558… = submitted; Task 4 31c6dff6… = the Mac file. Task 1 0787dee7…: 99 % of cells differ, mean |Δ| 0.095 km/h and 7.3 vph (p99 0.65 km/h, 66 vph), the same size as the 10-30 Mac mini run. The format check stopped because polars is missing in the Mac mini venv, so submission_B was not merged there. Copies: `repro/macmini_final_ab/`.
- If the final A becomes the gated network blend (cpu3g_go50, LB 0.88376 on 11/3), run_all.sh needs the network step from `task2_nn/repro_nn.sh` (B20).

## 11-03 run_all.sh: A = gated network blend, B = i5
- `T2_CANDS=(A:nn/cpu3g_go50 B:lgb_onthr_sq31_pao1535)`. After `task2/repro_g5.sh`, `task2_nn/repro_final.sh` builds the network candidate from the same task2 folder (prep 81 s, inference with the shipped CPU-deterministic weights `task2_nn/weights/cpu_r{1,2,3}` 3 × 5 s, gated mix 266 s; peak RSS 2.3 GB). Options NN_W_GO / NN_GATE pick the variant (go70: NN_W_GO=0.7; c00: NN_GATE=0.0); NN_TRAIN=1 retrains the networks (bit-identical only on GB10, about 4 h 45 min).
- Extra environment: a torch venv `.nn_venv` (python3.12, torch 2.14.1, numpy 2.5.4, CPU only), passed as PY_NN.
- Check on this Mac with the Task 2 step only (the empty-folder task2 build in `repro/nn_chain`): task2_A d7233f01… (= submitted cpu3g_go50), task2_B 2b9a602f… (= submitted i5). Network inference differs from GB10 by at most 1.8e-5 in probability; the candidate files are identical.
- Training the networks again on macOS (M4, same torch, 2 threads deterministic) diverges from step ~500, so the weights are bit-reproducible on GB10 only; the Mac retrain result is compared at cell level in task2_nn/RESULTS.md.

## 11-05 final B = go70; package runs on GB10 and on the Mac (M4, 24 GB)
- Final pair changed: A = cpu3g_go50 (d7233f01…, public 0.88376), B = cpu3g_go70 (08dbae54…, public 0.88324). B was i5 before; i5 stays the LightGBM base of both. `T2_CANDS=(A:nn/cpu3g_go50:0.5 B:nn/cpu3g_go70:0.7)`; `repro_final.sh` reuses trained networks for the second candidate when NN_TRAIN=1.
- `task2_nn/trunc_final.sh`: both blends rebuilt with every array cut at T: 0 of 174,000 cells differ for go50 and go70.
- Bug found by the package runs: the format check read `ROOT/data/kaggle_public/sample_submission.csv`. `check_submission.py` now follows TFB_REL and `run_all.sh` passes `--template`.
- GB10 (Linux aarch64, 16 threads, package zip in an empty folder): Task 4 2.5 min, Task 2 44 min + 5 min per blend, A and B byte-identical to the submitted files; Task 1 13 h 41 min, peak 20.1 GB, md5 48537a3a… (same as the earlier GB10 rebuild; 1.9 % of cells differ from the submitted c0a8ba04…, mean |Δ| 0.0020 km/h, 0.19 vph). The rebuilt A scored 0.88376 on the public leaderboard, equal to the submitted A.
- Mac M4 (macOS arm64, 24 GB, 8 threads, package zip in an empty folder, 11-04 22:06 → 11-05 23:06): Task 4 31c6dff6… (= submitted), Task 2 A d7233f01…, B 08dbae54… (= submitted); Task 1 24 h 11 min (heavy swapping; macOS peak footprint 9.0 GB), md5 abfa6946…, mean |Δ| 0.092 km/h and 7.1 vph from the submitted file (p99 0.62 km/h, 65 vph), the same size as the M4 Pro run. Merged A 6d03aa2f… and B a93ed4f7… pass the format check.
