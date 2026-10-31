"""Check LAB's tables against the saved measurements with the Python standard library."""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
PROFILES = ("full_A", "proxy_correct", "proxy_C2", "anti_full", "uniform")
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def read(name):
    return json.loads((DATA / f"phase_{name}.json").read_text())


def table(text, heading):
    section = text.split(heading, 1)[1].split("\n#", 1)[0]
    lines = re.search(r"(?m)(?:^\|[^\n]*\n)+", section).group().splitlines()
    return [[value.strip() for value in row.strip("|").split("|")] for row in lines[2:]]


def paired_rows(table_rows, profile_names):
    if len(table_rows) != len(profile_names):
        raise ValueError(f"Expected {len(profile_names)} profile rows, found {len(table_rows)}")
    return zip(table_rows, profile_names)


def pct(row):
    return row["pct_of_base"] if "pct_of_base" in row else row["pct_of_fp16"]


def ranking(rows):
    return sorted(PROFILES, key=lambda profile: pct(rows[profile]))


def frontier(sweep, budgets):
    points = [(int(b), name, rows[name]["mean_b"], pct(rows[name]))
              for b, rows in sweep.items() if int(b) in budgets for name in rows if name in PROFILES]
    return [[b, name] for b, name, width, loss in points
            if not any(w <= width and p <= loss and (w < width or p < loss) for _, _, w, p in points)]


def family_sweeps(suffix):
    family = read("c8a_mamba_scale_sweep" + suffix)["results"]
    sweeps = {model.split("/")[-1]: row["sweep"] for model, row in family.items()}
    # The family summary repeats the 2.8B anchor with less precision.
    sweeps["mamba-2.8b-hf"] = read("c4_state_int_quantisation" + suffix)["sweep_results"]
    return sweeps


def verify():
    checksum_file = ROOT / "verification/results.sha256"
    expected = {name: digest for digest, name in
                (line.split("  ", 1) for line in checksum_file.read_text().splitlines())}
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DATA.glob("*.json")}
    if expected != actual:
        raise ValueError("Saved result files differ from verification/results.sha256")

    paper = (ROOT / "LAB.md").read_text()
    guide = (ROOT / "docs/RESULTS.md").read_text()
    sections = ("4.5", "4.6", "5.1", "5.2", "5.3", "5.4", "6.2", "6.3", "6.6", "7.1", "7.3")
    tables = {sec: table(paper, f"### {sec} ") for sec in sections}
    sizes_table = table(guide, "## Quality and storage across five sizes")
    context_table = table(guide, "## Long-context results")
    checked = 0

    def check(cells, values):
        nonlocal checked
        numbers = [n for cell in cells for n in NUMBER.findall(cell)]
        if len(numbers) != len(values):
            raise ValueError(f"Table cells {cells} have the wrong number of values for {values}")
        for number, value in zip(numbers, values):
            decimals = len(number.partition(".")[2])
            if abs(float(number) - value) > 0.5 * 10 ** -decimals + 1e-10:
                raise ValueError(f"Printed {number} differs from {value} in {cells}")
            checked += 1

    family = family_sweeps("_v2")
    initial = family_sweeps("")
    anchor = family["mamba-2.8b-hf"]
    budgets = ("3", "4", "5", "6", "8")
    for row, profile in paired_rows(tables["4.6"], ("full_A", "proxy_correct", "proxy_C2", "uniform", "anti_full", "ratio")):
        values = [pct(anchor[b]["uniform"]) / pct(anchor[b]["full_A"]) if profile == "ratio"
                  else pct(anchor[b][profile]) for b in budgets]
        check(row[1:], values)

    ranks = {}
    frontiers = {}
    for i, (model, sweep) in enumerate(family.items()):
        four, six = sweep["4"]["full_A"], sweep["6"]["full_A"]
        bits4, bits6 = four["mean_b"], six["mean_b"]
        change4, change6 = pct(four), pct(six)
        check(tables["5.1"][i][2:], [pct(initial[model]["6"]["full"]), change4,
              pct(sweep["4"]["uniform"]) / change4, change6, pct(sweep["6"]["uniform"]) / change6])
        for j, key in enumerate(("uniform", "proxy_correct", "full_A")):
            result = sweep["6"][key]
            width = result["mean_b"]
            check(tables["6.3"][3*i+j][2:], [width, 16/width, 16/(width+0.2), pct(result)])
        check(tables["6.6"][i][1:], [bits4, 16/bits4, 16/(bits4+0.2), change4])
        check(sizes_table[i][1:], [16/(bits6+0.2), change6, 16/(bits4+0.2), change4])
        ranks[model] = {b: ranking(sweep[b]) for b in ("4", "6")}
        frontiers[model] = {"common_INT4_INT6": frontier(sweep, (4, 6)),
                            "all_saved_budgets": frontier(sweep, tuple(map(int, sweep)))}

    for section, name in [("5.3", "c8c_falcon_mamba_7b"), ("5.4", "c8d_jamba_mini")]:
        sweep = read(name + "_v2")["sweep"]
        ranks[name] = {b: ranking(sweep[b]) for b in ("4", "6")}
        for row, key in paired_rows(tables[section], ("full_A", "proxy_correct", "uniform", "proxy_C2", "anti_full")):
            check(row[1:], [pct(sweep[b][key]) for b in ("4", "6")])
    zamba = {k.removeprefix("profile_"): v for k, v in read("c5b_ssm_quant_in_hybrid_v2")["results"].items()}
    ranks["zamba2"] = {"6": ranking(zamba)}
    for row, key in paired_rows(tables["5.2"], ("proxy_C2", "full_A", "proxy_correct", "uniform", "anti_full")):
        check(row[1:], [pct(zamba[key])])

    noise = read("c3b_sigma_rel_sweep")["sweep_results"]
    for row, key in paired_rows(tables["4.5"], ("full", "proxy_correct", "proxy_C2", "anti_full", "uniform")):
        a, b, c = [pct(noise[s][key]) for s in ("0.05", "0.1", "0.2")]
        exponents = (math.log2(b/a), math.log2(c/b))
        check(row[-1:], [sum(exponents)/2, abs(exponents[1]-exponents[0])/2])

    matched = read("c16_matched_mean_sweep")["results"]["Mamba-2.8B"]["sweep"]
    ablated = read("c16b_lyapunov_vs_range_factorisation")["results"]["Mamba-2.8B"]["sweep"]
    profiles = ("uniform", "half_slope_rg", "half_slope", "range_only", "shuffled_R_plus_r", "R_plus_shuffled_r")
    for row, key in paired_rows(tables["6.2"], profiles):
        for i, budget in enumerate(("4", "6"), 1):
            name = f"b{budget}_{key}"
            source = matched if key in profiles[:3] else ablated
            result = source[name]
            if name in matched and name in ablated:
                for field in ("achieved_mean", "ppl"):
                    if matched[name][field] != ablated[name][field]:
                        raise ValueError(f"C16 and C16b disagree on {name}/{field}")
            check([row[i]], [result["delta_pct"], result["achieved_mean"]])

    paired = read("c7_long_context_validation _v2")["sweep_results"]
    clipped = read("c9_fp16_bf16_clip_ablation_v2")["sweep_results"]
    for i, length in enumerate(("8192", "16384", "32768")):
        lab = paired[length]["full_A"]["ppl"]
        uniform = paired[length]["uniform"]["ppl"]
        for row in (tables["7.1"][i], context_table[i]):
            check(row[1:], [lab, uniform, uniform/lab])
        check(tables["7.3"][i][1:], [clipped[length][k]["ppl"] for k in
              ("FP16_orig", "BF16_orig", "FP16_clip_only", "FP16_full_A")])

    orders = [order for model in ranks.values() for order in model.values()]
    counts = {"model_budget_cells": len(orders), "LAB_first": sum(x[0] == "full_A" for x in orders),
              "continuous_order_reproduced": sum(x == list(PROFILES) for x in orders)}
    if counts != {"model_budget_cells": 15, "LAB_first": 14, "continuous_order_reproduced": 0}:
        raise ValueError(f"Model rankings differ from §5.5: {counts}")
    if [len(v["common_INT4_INT6"]) for v in frontiers.values()] != [2, 2, 4, 6, 6]:
        raise ValueError("Quality–storage frontiers differ from §6.4")
    if checked != 244:
        raise ValueError(f"Expected 244 printed values, checked {checked}")
    return {"status": "PASS", "printed_numeric_values_verified": checked,
            "saved_json_files_verified": len(actual), "rank_counts": counts,
            "ranking": ranks, "frontiers": frontiers}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
