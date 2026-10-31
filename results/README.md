# Saved results

To check a number in the paper, open the **[figure and table evidence map](../docs/EVIDENCE_MAP.md)**. It gives the phase, collecting code, exact JSON field, and evaluation conditions. These files can be read without running the [experiments](../experiments/README.md).

| Filename pattern | How to read it |
|---|---|
| `*_v2.json` | The seven rerun archives. Both integer recipes may appear in one file: e.g. the [C4 anchor](phase_c4_state_int_quantisation_v2.json) retains the old `full` row alongside LAB's `full_A`. Historical verdict flags about `full` do not score LAB. |
| No version suffix | For a phase with a matching rerun, this is the earlier archive. Other unsuffixed files support different measurements: C3b is continuous noise; C16/C16b already test the range-aware rule as `half_slope_rg`; C10b records pre-clip trajectories. |
| `*_partial_*.json` | Per-model snapshots from C16/C16b. Use the combined files for the paper's factorisation comparison; do not count the snapshots as additional replicates. |

Two easy-to-confuse files: the [C7 rerun](phase_c7_long_context_validation%20_v2.json) compares uniform and LAB on **two** windows (the filename contains a space before `_v2`); the [C9 rerun](phase_c9_fp16_bf16_clip_ablation_v2.json) compares precision and clipping on **four** windows and has no uniform row. Keep ratios within one evaluation.
