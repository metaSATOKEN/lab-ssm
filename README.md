# Lyapunov-Aware Bit Allocation for State-Space Model Quantisation

Kentaro Sato · 31 October 2026 · [Paper (PDF)](LAB.pdf) · [Paper (Markdown)](LAB.md)

**If two values in a model’s memory are equally large, should they get the same number of bits?** A Mamba model carries a numerical state from one token to the next. A rounding error in one part of that state can fade by the next token. An error of the same size elsewhere can survive many updates and change a later prediction.

LAB uses a short calibration passage to estimate how long an error in each state group is likely to persist and how strongly that group influences the output. It combines those estimates with the range of values the group takes to assign its bit width. Across five Mamba-1 sizes, this rule kept the increase in perplexity below 4% on WikiText-2 at a nominal six bits. Giving every group six bits raised perplexity by at least 18% on each model.

## How much memory does that save?

We simulated rounding the recurrent state during inference and measured perplexity, a measure of how well the model predicts the next token. Lower is better. The table compares each model with its own FP16 baseline. Six bits was the **target average** for LAB, so the table also shows the width it actually used.

| Allocation on five Mamba-1 sizes | Actual mean bits | Estimated state-cache compression vs. FP16 | Measured perplexity increase |
|---|---:|---:|---:|
| Six bits for every group | 6.00 | 2.58× | +18.18% to +23.09% |
| LAB | 5.84–6.35 | 2.44–2.65× | **+3.35% to +3.98%** |

Both compression estimates include the cost of storing each group’s range and bit width, shared across 100 stored states. They describe the recurrent state cache. End-to-end GPU memory and inference speed remain open engineering questions.

## Does it matter which groups get the bits?

In a separate test on five text windows, we kept each group’s range and shuffled the scores for persistence and output influence across groups. On Mamba-2.8B, LAB and the shuffled rule both averaged **3.96 bits**. Yet the perplexity increase rose from **19.3% to 90.0%** after shuffling. On these windows, the model kept more of its predictive quality when the score went to the group it described.

![On Mamba-2.8B, the left panel compares four-bit allocations. LAB and “Shuffle R” both average 3.96 bits, but shuffling raises the perplexity penalty. The right panel shows the six-bit comparison.](figures/fig04_factorisation.png)

## How far does the result travel?

Across eight models, LAB gave the lowest perplexity of five allocation rules in **14 of 15** model–budget comparisons. The exception was Zamba2 at six bits, where a simpler rule based on persistence narrowly led.

On Mamba-2.8B, LAB also gave lower perplexity than uniform allocation on two matched PG-19 windows at each length from 8K to 32K tokens. On a separate set of four windows at 32K, clipping the state to its calibrated range alone gave slightly lower perplexity than LAB. That comparison shows why clipping also matters when interpreting the long-context result.

The initial integer experiments tested a rule based on persistence and output influence alone. The reruns tested LAB’s revised weighting of that score together with each group’s range and retained the earlier rule for comparison. [Results by model and experiment](docs/RESULTS.md) · [Trace each figure and table to the data](docs/EVIDENCE_MAP.md)

[Experiment scripts](experiments/README.md) · [Saved results](results/README.md) · [License](LICENSE)
