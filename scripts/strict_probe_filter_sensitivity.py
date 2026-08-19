#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
results = pd.read_parquet(ROOT / "results/age_sex_interaction_results.parquet")
problem_file = ROOT / "venv/lib/python3.12/site-packages/methylcheck/data_files/450k_polymorphic_crossRxtve_probes.csv.gz"
problem = pd.read_csv(problem_file)
problem_probes = set(problem["Probe"].astype(str))
strict = results.loc[~results["probe"].isin(problem_probes)].copy()

for p_column in [column for column in strict.columns if column.endswith("_p")]:
    values = strict[p_column].to_numpy(float)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    strict[p_column.replace("_p", "_strict_fdr")] = restored

strict.to_parquet(ROOT / "results/age_sex_interaction_results_strict_filter.parquet", index=False)
strict.nsmallest(1000, "fully_adjusted_hc3_p").to_csv(
    ROOT / "results/top1000_age_sex_interactions_strict_filter.tsv", sep="\t", index=False
)

flagged = problem.loc[problem["Probe"].isin(results.loc[results["fully_adjusted_hc3_fdr"] < 0.05, "probe"])]
flagged.groupby(["Probe", "Reason"])["ShortCitation"].agg(lambda x: ";".join(sorted(set(x)))).reset_index().to_csv(
    ROOT / "results/primary_significant_probe_flags.tsv", sep="\t", index=False
)

print(f"Strictly retained probes: {len(strict)}")
print(f"Strict-filter HC3 FDR < 0.05: {(strict['fully_adjusted_hc3_strict_fdr'] < 0.05).sum()}")
