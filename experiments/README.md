# Experiment scripts

Start with the **[figure and table evidence map](../docs/EVIDENCE_MAP.md)**. It pairs each measurement with its script, saved file, and result key. No script needs to run to inspect the reported numbers.

| What was measured | Phase scripts (in the same order) |
|---|---|
| Estimate the static and empirical transition, output coupling, and train/validation agreement | [C1](phase_c1_ssm_spectral_baseline.py), [C1b](phase_c1b_mamba1_empirical.py), [C1c](phase_c1c_gamma_extraction.py), [C1c-bis](phase_c1c_bis_calibration_robustness.py) |
| Initial parameter-perturbation probes; direct continuous state-noise tests and amplitude sweep | [C2](phase_c2_bit_allocator_ppl.py), [C2b](phase_c2b_bit_allocator_ppl_amplified.py), [C2c](phase_c2c_full_R_allocator_ppl.py), [C3](phase_c3_state_step_quantisation.py), [C3b](phase_c3b_sigma_rel_sweep.py) |
| Integer anchor; multi-size and cross-model transfer | [C4](phase_c4_state_int_quantisation.py), [C8a](phase_c8a_mamba_scale_sweep.py), [C5b](phase_c5b_ssm_quant_in_hybrid.py), [C8c](phase_c8c_falcon_mamba_7b.py), [C8d](phase_c8d_jamba_mini.py) |
| Hybrid model setup; joint attention/state composition and budget sweep | [C5a](phase_c5a_hybrid_load_substrate.py), [C5c](phase_c5c_hybrid_composition.py), [C5d](phase_c5d_composition_b0_sweep.py) |
| Historical packing, slope tests, targeted-mean comparisons and range/score ablations | [C6](phase_c6_real_packing_compression.py), [C12](phase_c12_strict_optimum_full.py), [C16](phase_c16_matched_mean_sweep.py), [C16b](phase_c16b_lyapunov_vs_range_factorisation.py) |
| Long-context allocator comparison, precision/clipping controls, and trajectory logging | [C7](phase_c7_long_context_validation.py), [C9](phase_c9_fp16_bf16_clip_ablation.py), [C10](phase_c10_state_norm_trajectory.py), [C10b](phase_c10b_state_norm_pre_clip.py) |

The seven integer/long-context rerun scripts (C4, C5b, C7, C8a, C8c, C8d, C9) are the later versions of those phases. Many Colab scripts write an **unsuffixed** filename under `/content/`; consult the [result-file guide](../results/README.md) for the archived name.

[Shared allocation helper](_lab_alloc.py) describes the named rules; the seven rerun scripts contain inline implementations. [C10/C10b/C12 analysis helper](analyze_c10_c10b_c12_results.py) reads existing outputs. For the figures actually in the paper, use the links in the evidence map.
