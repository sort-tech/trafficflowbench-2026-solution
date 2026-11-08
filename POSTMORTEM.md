# Post-competition notes

Written after the private leaderboard was released. The report in `report/` was finished before that, so it does not contain these results. Everything below comes from the two leaderboard scores of each submission. We did not analyse private labels.

## Final standing

- 130th of 802 teams, private score 0.86748 (final A). Final B scored 0.86725. On the public leaderboard the team was 77th.
- 10th place scored 0.89721, so the gap to the award review pool was about 0.030.
- The package rebuild of final A (`r_pkg_gb10_A`) also scored 0.86748 on the private leaderboard, the same as the submitted A to all five digits.

## Scores that matter

| Submission | What it is | Public | Private |
|---|---|---|---|
| `m_v7_cpu3ggo50_nnls` | **Final A**: i5 + network 0.5, gated | 0.88376 | 0.86748 |
| `p_v7_cpu3ggo70_nnls` | **Final B**: i5 + network 0.7, gated | 0.88324 | 0.86725 |
| `m_v7_cpu3ggo30_nnls` | i5 + network 0.3, gated | 0.88263 | 0.86899 |
| `i5_v7_pao1535_nnls` | i5: LightGBM only, the base of both finals | 0.88019 | 0.86907 |
| `k_v7_cpu3go50_nnls` | i5 + network 0.5, no gate | 0.87876 | 0.86571 |
| `m1_v7_sclim75p_nnls` | i5 + split climatology on onset windows | 0.88019 | 0.87322 |
| `n_v7_sca75_nnls` | i5 + split climatology everywhere, λ 0.75 | 0.87366 | 0.87157 |
| `c1_gbm4c_lgb1_nnls` | early anchor: older state model, `lgb_v1` queue | 0.86066 | 0.85696 |
| `c3_gbm4c_notod35_nnls` | same state model, queue model without time-of-day features | 0.85836 | 0.87259 |

## What the private month showed

**1. Near the top, the public score said little about the private score.** Across the 31 of our submissions with a public score of at least 0.87, the correlation between the two leaderboards was 0.17 (Pearson) and 0.33 (Spearman). Across the 45 with at least 0.80 it was 0.84 and 0.58, because that range still includes large real improvements. The rebuild `r_pkg_gb10_A` is left out of these counts because it duplicates final A.

**2. The network blend did not transfer.** Compared with i5, the blend gained on public and lost on private, and the loss grew with the network weight.

| Network weight (gated) | Public vs i5 | Private vs i5 |
|---|---|---|
| 0.3 | +0.00244 | −0.00008 |
| 0.5 (final A) | +0.00357 | −0.00159 |
| 0.7 (final B) | +0.00305 | −0.00182 |

The climatology gate itself did transfer. Against the same blends without the gate it added +0.00103, +0.00177 and +0.00252 on private at weights 0.3, 0.5 and 0.7.

**3. Reducing dependence on train-month climatology is what held up.** This matches finding 2 in the report, more strongly than we expected.

- Replacing train climatology with climatology measured early in the same split, on onset windows only (`m1_v7_sclim75p_nnls`), changes nothing on public by construction and added +0.00415 on private. It is our best private score.
- Applying split climatology everywhere (`n_v7_sca75_nnls`, `n_v7_sca5_nnls`) lost 0.00653 and 0.00734 on public but gained 0.00250 and 0.00233 on private, against i5.
- The early queue model without time-of-day features (`c3_gbm4c_notod35_nnls`) lost 0.00230 on public against its anchor `c1_gbm4c_lgb1_nnls` and gained 0.01563 on private. That is about +0.052 in S_queue, and it is our second-best private score out of 56, even with an older state model.

A caution on the first bullet: +0.00415 on the leaderboard is +0.0138 in S_queue. Each of the 80 scored windows carries 1/80 of S_queue, so this equals about 1.1 windows of IoU, and only 7 windows changed. It points the same way as the other two results, but on its own it is a small sample.

## What we would do differently

- **Choose the two finals to test different hypotheses.** Our rule for the second final required a public score within 0.002 of final A. That removed i5 (−0.0036), and it would equally have removed `sclim75p`, which has the same public score as i5. The rule left the go70 blend, which differs from final A in only 95 private cells across 24 windows, so the pair hedged almost nothing. With i5 as the second final the team score would have been 0.86907. With `sclim75p` it would have been 0.87322, about 100th place. Neither would have reached the top 10.
- **Treat public gains of a few thousandths as noise when they come from one month of 80 windows.** The network was first dropped and later brought back because of a +0.0036 public gain.
- **Give more weight to models that do not use calendar features.** The no-time-of-day model was tested on the first day of combined submissions and set aside because its public score was lower.

## Differences from the package sent to the organisers

- `report/sortdev_report.pdf`: the contact e-mail was removed from the author line. The rest of the text is unchanged.
- `task2/sclim_all_eval.py` and `task2/trunc_test.py`: temporary paths from the author's machine were replaced by `/tmp/`. `final/run_all.sh` does not call either script.
- Added: this file, the Result section of `README.md`, and `.gitignore`.

## All 56 submissions

Dates are UTC, as recorded by Kaggle.

| # | Date | Submission | Public | Private | Private − Public |
|---|---|---|---|---|---|
| 1 | 2026-10-25 | `b0_baseline` | 0.55343 | 0.54859 | -0.00484 |
| 2 | 2026-10-25 | `q1_lgb_v1` | 0.69083 | 0.68229 | -0.00854 |
| 3 | 2026-10-25 | `o1_nnls_own` | 0.58501 | 0.58176 | -0.00325 |
| 4 | 2026-10-25 | `o2_nnls_w75` | 0.56381 | 0.55852 | -0.00529 |
| 5 | 2026-10-25 | `q2_diag_onset` | 0.66649 | 0.65029 | -0.01620 |
| 6 | 2026-10-26 | `c1_gbm4c_lgb1_nnls` | 0.86066 | 0.85696 | -0.00370 |
| 7 | 2026-10-26 | `c8_gbm8s2_lgb1_nnls` | 0.86414 | 0.86048 | -0.00366 |
| 8 | 2026-10-26 | `c3_gbm4c_notod35_nnls` | 0.85836 | 0.87259 | +0.01423 |
| 9 | 2026-10-26 | `c4_gbm4c_robust_nnls` | 0.85454 | 0.86562 | +0.01108 |
| 10 | 2026-10-26 | `c9_diag_persT` | 0.84604 | 0.83617 | -0.00987 |
| 11 | 2026-10-27 | `d1_diag_sensor` | 0.78677 | 0.76796 | -0.01881 |
| 12 | 2026-10-27 | `d3_diag_famA` | 0.77700 | 0.78699 | +0.00999 |
| 13 | 2026-10-27 | `d4_diag_famB` | 0.79194 | 0.78020 | -0.01174 |
| 14 | 2026-10-27 | `d8_gbm8s2ss_lgb1_nnls` | 0.86552 | 0.86195 | -0.00357 |
| 15 | 2026-10-27 | `d7_gbm8s2_lgb2_nnls` | 0.86067 | 0.86360 | +0.00293 |
| 16 | 2026-10-28 | `d9_diag_zeros` | 0.63274 | 0.63215 | -0.00059 |
| 17 | 2026-10-28 | `d5_diag_famC` | 0.78545 | 0.78548 | +0.00003 |
| 18 | 2026-10-28 | `d10_diag_onsetdil` | 0.81607 | 0.81991 | +0.00384 |
| 19 | 2026-10-28 | `e1_x3gfix_lgb1_nnls` | 0.86478 | 0.86117 | -0.00361 |
| 20 | 2026-10-28 | `e2_x3gssfix_lgb1_nnls` | 0.86616 | 0.86263 | -0.00353 |
| 21 | 2026-10-29 | `f1_x3gss3f_lgb1_nnls` | 0.86631 | 0.86279 | -0.00352 |
| 22 | 2026-10-29 | `f2_ss3f_br03_nnls` | 0.86559 | 0.86564 | +0.00005 |
| 23 | 2026-10-29 | `f3_ss3f_todsnap_nnls` | 0.86381 | 0.86173 | -0.00208 |
| 24 | 2026-10-29 | `f3_ss3f_ongphys_nnls` | 0.87189 | 0.86722 | -0.00467 |
| 25 | 2026-10-29 | `f4_ss3f_ongphys03_nnls` | 0.87240 | 0.86726 | -0.00514 |
| 26 | 2026-10-30 | `g1_ss3f_ongphys0_nnls` | 0.87003 | 0.86671 | -0.00332 |
| 27 | 2026-10-30 | `g2_ss3f_onaug_nnls` | 0.86765 | 0.86980 | +0.00215 |
| 28 | 2026-10-30 | `g3_ss3f_ongphysaug_nnls` | 0.87419 | 0.86523 | -0.00896 |
| 29 | 2026-10-30 | `g4_ss3f_ongpa44_nnls` | 0.87505 | 0.86521 | -0.00984 |
| 30 | 2026-10-30 | `g5_ss3f_ongpa26_nnls` | 0.87518 | 0.86430 | -0.01088 |
| 31 | 2026-10-31 | `h1_ss3f_ongpm26_nnls` | 0.87514 | 0.86562 | -0.00952 |
| 32 | 2026-10-31 | `h2_ss3f_sq31_nnls` | 0.87701 | 0.86529 | -0.01172 |
| 33 | 2026-10-31 | `h3_ss3f_ongpo26_nnls` | 0.87622 | 0.86788 | -0.00834 |
| 34 | 2026-10-31 | `h4_ss3f_onthr_nnls` | 0.87738 | 0.86616 | -0.01122 |
| 35 | 2026-10-31 | `h5_ss3f_onthr_sq31_nnls` | 0.87922 | 0.86714 | -0.01208 |
| 36 | 2026-11-01 | `i1_v7_onthr_sq31_nnls` | 0.87951 | 0.86751 | -0.01200 |
| 37 | 2026-11-01 | `i2_v7_pao33_nnls` | 0.87943 | 0.86880 | -0.01063 |
| 38 | 2026-11-01 | `i3_v7_athr_nnls` | 0.87906 | 0.86780 | -0.01126 |
| 39 | 2026-11-01 | `i4_v7_hmix_nnls` | 0.88006 | 0.86894 | -0.01112 |
| 40 | 2026-11-01 | `i5_v7_pao1535_nnls` | 0.88019 | 0.86907 | -0.01112 |
| 41 | 2026-11-02 | `k_v7_cpu3go50_nnls` | 0.87876 | 0.86571 | -0.01305 |
| 42 | 2026-11-02 | `j1_rf7_pao1535_nnls` | 0.88019 | 0.86908 | -0.01111 |
| 43 | 2026-11-02 | `k_v7_cpu3go70_nnls` | 0.87778 | 0.86473 | -0.01305 |
| 44 | 2026-11-02 | `k_v7_cpu3go30_nnls` | 0.87905 | 0.86796 | -0.01109 |
| 45 | 2026-11-02 | `j2_v7_pao1535hmix_nnls` | 0.88013 | 0.86927 | -0.01086 |
| 46 | 2026-11-03 | `n_v7_scr75_nnls` | 0.87527 | 0.86808 | -0.00719 |
| 47 | 2026-11-03 | `n_v7_sca75_nnls` | 0.87366 | 0.87157 | -0.00209 |
| 48 | 2026-11-03 | `n_v7_sca5_nnls` | 0.87285 | 0.87140 | -0.00145 |
| 49 | 2026-11-03 | `m_v7_cpu3ggo50_nnls` | 0.88376 | 0.86748 | -0.01628 |
| 50 | 2026-11-03 | `m1_v7_sclim75p_nnls` | 0.88019 | 0.87322 | -0.00697 |
| 51 | 2026-11-04 | `p_v7_cpu3ggo70_nnls` | 0.88324 | 0.86725 | -0.01599 |
| 52 | 2026-11-04 | `p_v7_cpu3gc00_nnls` | 0.88215 | 0.86756 | -0.01459 |
| 53 | 2026-11-04 | `m_v7_cpu3ggo30_nnls` | 0.88263 | 0.86899 | -0.01364 |
| 54 | 2026-11-04 | `p_v7_cpu3gc07_nnls` | 0.88055 | 0.86629 | -0.01426 |
| 55 | 2026-11-04 | `q_v7_cpu3gago50_nnls` | 0.88315 | 0.86852 | -0.01463 |
| 56 | 2026-11-05 | `r_pkg_gb10_A` | 0.88376 | 0.86748 | -0.01628 |
