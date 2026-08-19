#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
meta = pd.read_csv(ROOT / "results/sample_metadata.tsv", sep="\t", keep_default_na=True)
meta.loc[meta["sex"] == "NA", "sex"] = np.nan
meta = meta.loc[meta["age"].notna() & meta["sex"].isin(["F", "M"])].copy()

age = np.where(meta.age <= 20, np.log(meta.age + 1) - np.log(21), (meta.age - 20) / 21)
age = age - age.mean()
male = meta.sex.eq("M").astype(float).to_numpy()
design = pd.DataFrame({"intercept": 1.0, "age": age, "male": male, "age_x_male": age * male}, index=meta.index)
design = pd.concat(
    [
        design,
        pd.get_dummies(meta.sentrix_id.astype(str), prefix="slide", drop_first=True, dtype=float),
        pd.get_dummies(meta.sentrix_position.astype(str), prefix="position", drop_first=True, dtype=float),
        pd.get_dummies(meta.year_of_collection.astype(str), prefix="collection", drop_first=True, dtype=float),
    ], axis=1,
)
x0 = design.to_numpy(float)
keep, rank = [], 0
for j in range(x0.shape[1]):
    new_rank = np.linalg.matrix_rank(x0[:, keep + [j]])
    if new_rank > rank:
        keep.append(j); rank = new_rank
design = design.iloc[:, keep]
x = design.to_numpy(float)
inv = np.linalg.inv(x.T @ x)
projection = inv @ x.T
leverage = np.sum((x @ inv) * x, axis=1)
df = len(meta) - np.linalg.matrix_rank(x)

rows = []
for cell in ["CD8T", "CD4T", "NK", "Bcell", "Mono", "Gran"]:
    y = meta[cell].to_numpy(float)
    coef = projection @ y
    residual = y - x @ coef
    def contrast_result(terms):
        c = np.zeros(x.shape[1])
        for term, value in terms.items(): c[design.columns.get_loc(term)] = value
        estimate = c @ coef
        weights = c @ inv @ x.T
        se = np.sqrt(np.sum((weights * residual / np.maximum(1 - leverage, 1e-8)) ** 2))
        p = 2 * stats.t.sf(abs(estimate / se), df)
        return estimate, se, p
    female = contrast_result({"age": 1})
    male_slope = contrast_result({"age": 1, "age_x_male": 1})
    interaction = contrast_result({"age_x_male": 1})
    rows.append({
        "cell_fraction": cell,
        "female_change_per_10_adult_years": female[0] * 10 / 21,
        "female_hc3_p": female[2],
        "male_change_per_10_adult_years": male_slope[0] * 10 / 21,
        "male_hc3_p": male_slope[2],
        "male_minus_female_change_per_10_adult_years": interaction[0] * 10 / 21,
        "interaction_hc3_p": interaction[2],
    })

out = pd.DataFrame(rows)
out["interaction_fdr"] = np.minimum(out["interaction_hc3_p"] * len(out), 1.0)
out.to_csv(ROOT / "results/leukocyte_age_sex_associations.tsv", sep="\t", index=False)
print(out.to_string(index=False))
