# Lyapunov-Aware Bit Allocation for State-Space Model Quantisation

**If a model has six bits per state value on average, which values should get the extra precision?** A Mamba language model carries a numerical state from one token to the next. Two equally large state values can matter very differently. A rounding error in one can fade after a few updates, while the same error in another can persist and affect later predictions.

LAB estimates how long each error may persist, how strongly it reaches the output, and how wide a range each group of values needs. On five Mamba-1 sizes, allocating bits this way kept the perplexity increase below 4% at a nominal six bits. Giving every group exactly six bits raised perplexity by 18–23%.

Kentaro Sato · 2 November 2026 · [Paper (PDF)](LAB.pdf) · [Manuscript](LAB.md) · [How LAB works](#from-a-state-update-to-a-bit-width) · [日本語の概要](#日本語の概要) · [Code and saved results](#check-a-result)

## What six bits buys on five models

The integer reruns compared each model with its own FP16 baseline on WikiText-2. Perplexity measures how well the model predicts the next token. A smaller increase means more of the original prediction quality survived quantisation.

| Allocation at nominal six bits | Achieved mean bits per value | Estimated state-cache compression vs. FP16 | Measured perplexity increase |
|---|---:|---:|---:|
| Six bits for every group | 6.00 | 2.58× | +18.18% to +23.09% |
| LAB | 5.84–6.35 | 2.44–2.65× | **+3.35% to +3.98%** |

The cache estimate counts 20 metadata bits per group for the range and bit width. If 100 separately stored states share this metadata, the overhead is 0.2 bits per value, giving a compression ratio of $16/(\bar b+0.2)$ at achieved mean width $\bar b$. A single state overwritten at every token still counts as one stored state. The experiments measured prediction quality by simulating quantisation during inference. End-to-end GPU memory and throughput require a packed implementation and benchmark. [Per-model results and storage assumptions](docs/RESULTS.md#quality-and-storage-across-five-sizes)

## From a state update to a bit width

**1. Follow a rounding error for one more token.** Let $e_t$ be the difference between the quantised and reference states at step $t$. The transition $\bar A_t$ carries part of the previous error forward. Rounding and clipping the current update introduce a local error $\eta_t$. For the two paths through a layer with the same input and initial state,

$$
e_t = \bar A_t e_{t-1} + \eta_t.
$$

**2. Measure what differs between groups.** LAB runs one calibration pass over 1,024 WikiText-2 training tokens. In Mamba-1, each group $g$ pairs one channel with one state unit.

| Measurement | How LAB uses it |
|---|---|
| Persistence $\rho_g$, the mean magnitude of the state transition | A value closer to one indicates that earlier errors can survive more updates. |
| Output coupling $\Gamma_g$, the mean squared readout coefficient | A larger coefficient gives the same state error more influence on the output. |
| Half-range $r_g = 7\,\mathrm{RMS}(h_g)$, from the root mean square of the group's state values | A wider range gives coarser rounding at the same bit width. |

**3. Spend the bits where finer rounding helps most.** LAB combines persistence and coupling into a score $R_g$. Under a fine-resolution approximation, the group's contribution to squared readout error scales with $R_g r_g^2 4^{-b_g}$. Minimising the sum under a fixed budget gives the continuous allocation

$$
R_g=\frac{\Gamma_g}{1-\rho_g}, \qquad
b_g^\star=c+\tfrac12\log_2 R_g+\log_2 r_g.
$$

The constant $c$ sets the budget. At the same range, a fourfold score calls for one extra bit. At the same score, doubling the range also calls for one extra bit. The integer rule computes extra bits from the score and range, recentres the widths around the target, then rounds and clamps them to 2–16 bits. This explains the spread in achieved widths in the table. During Mamba-1 inference, the quantiser rounds and clips each updated value to its group's signed integer grid. The dequantised value feeds both the readout and the next token update. [Derivation and integer rule](LAB.md#3-theory-a-lyapunov-aware-bit-allocator) · [Mamba-2.8B implementation](experiments/phase_c4_state_int_quantisation.py)

The Lyapunov bound motivating the score assumes contraction at each step and zero-mean noise uncorrelated with the propagated error. For allocation, LAB uses an empirical *mean* transition magnitude. The continuous approximation also omits clipping and cross-group error terms. The tests below ask how well that rule works with integer widths.

## Does assigning the right score to each group matter?

In a separate five-window test on Mamba-2.8B, we kept each group's range and shuffled the persistence–coupling scores across groups. At nominal four bits, both allocations achieved **3.96 bits** on average. Perplexity increased by **19.3% with LAB and 90.0% after shuffling**. With the bit cost matched to the reported precision, which group received which score mattered.

![Four- and six-bit comparisons on Mamba-2.8B. At four bits, shuffling the scores across groups raises the perplexity penalty even though the achieved mean width matches LAB's.](figures/fig04_factorisation.png)

*Paper Figure 3. Both panels use the separate five-window test, which adjusts the bit budget by binary search. The six-bit LAB increase here is 4.7%. The five-size reruns above used a different budget adjustment.* [Saved shuffle comparison](results/phase_c16b_lyapunov_vs_range_factorisation.json)

## Where the result transfers

Across eight models, LAB had the lowest measured perplexity among five allocators in **14 of 15 model–budget comparisons**. Seven models were tested at both four and six bits. On Zamba2, evaluated only at six bits with quantisation at chunk boundaries, a simpler persistence rule narrowly led. These rankings describe the evaluated text windows. [Rankings by experiment](docs/RESULTS.md#what-transfers-to-other-models)

Long contexts provide two separate comparisons. On Mamba-2.8B, LAB beat uniform six-bit allocation on the same two PG-19 windows at each of 8K, 16K and 32K tokens. At 32K, their perplexities were **8.82 and 13.55**, respectively. On a separate set of four 32K windows, **clipping alone gave 10.51 and LAB gave 10.97**. Bounding state values can itself improve prediction on long sequences. [Paired windows and clipping results](docs/RESULTS.md#long-context-results-keep-the-windows-paired)

## 日本語の概要

Mambaは、直前までに読んだ情報を数値として次のトークンへ引き継ぎます。その数値を少ないビットで保存すると丸め誤差が生じます。LABの問いは、**同じ平均ビット予算なら、どこを細かく保存すれば予測の質を保てるか**です。短い文章で「誤差の残りやすさ」「出力への効きやすさ」「値が動く幅」を測り、グループごとに精度を配分します。

Mamba-1の5サイズで、目標を平均6ビットにすると、一律6ビットで18–23%だった予測指標（perplexity）の悪化がLABでは約3–4%に収まりました。別の比較では、平均3.96ビットのままグループ間でスコアを入れ替えるだけで、悪化が19.3%から90.0%に広がりました。圧縮率は、100個の保存状態が設定情報を共有する場合の状態キャッシュの見積もりです。[モデル別の結果と計算条件](docs/RESULTS.md)

## Check a result

The initial integer experiments used an earlier score-only rule. In the Mamba-2.8B rerun, the JSON retains that rule as `full` and records range-aware LAB as `full_A`. For example, open the six-bit row and compare the perplexity increases of **6.69%** and **3.98%**, respectively. The five-size summary reuses this 2.8B run. [Anchor rerun JSON](results/phase_c4_state_int_quantisation_v2.json) · [Guide to experiment labels](docs/RESULTS.md#which-experiment-does-each-number-describe)

| To… | Start here |
|---|---|
| Trace a paper figure or table to the script and measured values | [Evidence map](docs/EVIDENCE_MAP.md) |
| Inspect a specific run or read its implementation | [Saved result files](results/README.md) · [Experiment scripts](experiments/README.md) |
| Verify the paper tables from saved JSON | Run `make verify` from the repository root with Python 3.9+ and Make. [Verification script](verification/verify_paper.py) · [Saved summary](verification/summary.json) |

**Cite the paper:** Kentaro Sato (2026), *Lyapunov-Aware Bit Allocation for State-Space Model Quantisation*. [DOI 10.5281/zenodo.20017854](https://doi.org/10.5281/zenodo.20017854) · [Citation metadata](CITATION.cff) · [License](LICENSE) · [Contact](mailto:info@metaclan.jp)
