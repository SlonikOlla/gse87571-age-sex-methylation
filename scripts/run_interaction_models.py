#!/usr/bin/env python3
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
P1 = RESULTS / "matrix1_beta.parquet"
P2 = RESULTS / "matrix2_beta.parquet"
Q1 = RESULTS / "matrix1_detection_qc.parquet"
Q2 = RESULTS / "matrix2_detection_qc.parquet"
META = RESULTS / "sample_metadata.tsv"
MANIFEST = ROOT / "downloads" / "HumanMethylation450k_15017482_v3.csv.gz"
QUALITY_MASK = ROOT / "venv/lib/python3.12/site-packages/methylprep/models/qualityMask450.txt.gz"


def horvath_age(age, adult_age=20.0):
    age = np.asarray(age, dtype=float)
    return np.where(
        age <= adult_age,
        np.log(age + 1.0) - np.log(adult_age + 1.0),
        (age - adult_age) / (adult_age + 1.0),
    )


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    good = np.isfinite(p)
    vals = p[good]
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[good] = tmp
    return out


def design_matrix(meta, age_kind="transformed", adjustment="full"):
    age = horvath_age(meta["age"]) if age_kind == "transformed" else meta["age"].to_numpy(float)
    age = age - np.mean(age)
    male = (meta["sex"] == "M").astype(float).to_numpy()
    pieces = [
        pd.DataFrame({"intercept": 1.0, "age": age, "male": male, "age_x_male": age * male}, index=meta.index)
    ]
    if adjustment in {"cell", "full"}:
        # Granulocytes are the omitted fraction because the six estimates are compositional.
        pieces.append(meta[["CD8T", "CD4T", "NK", "Bcell", "Mono"]].astype(float))
    if adjustment == "full":
        pieces.append(pd.get_dummies(meta["sentrix_id"].astype(str), prefix="slide", drop_first=True, dtype=float))
        pieces.append(pd.get_dummies(meta["sentrix_position"].astype(str), prefix="position", drop_first=True, dtype=float))
        pieces.append(pd.get_dummies(meta["year_of_collection"].astype(str), prefix="collection", drop_first=True, dtype=float))
    design = pd.concat(pieces, axis=1)
    # Remove any exactly redundant columns while retaining the prespecified terms.
    x = design.to_numpy(float)
    keep = []
    rank = 0
    for column in range(x.shape[1]):
        candidate = x[:, keep + [column]]
        new_rank = np.linalg.matrix_rank(candidate)
        if new_rank > rank:
            keep.append(column)
            rank = new_rank
    design = design.iloc[:, keep]
    if "age_x_male" not in design.columns:
        raise RuntimeError("Interaction term became rank deficient")
    return design


def fit_interaction(y, design, clusters):
    x = design.to_numpy(float)
    xtx_inv = np.linalg.inv(x.T @ x)
    projector = xtx_inv @ x.T
    coef = projector @ y
    resid = y - x @ coef
    df = x.shape[0] - np.linalg.matrix_rank(x)
    sigma2 = np.sum(resid * resid, axis=0) / df
    j = design.columns.get_loc("age_x_male")
    se = np.sqrt(np.maximum(sigma2 * xtx_inv[j, j], 0))
    tval = coef[j] / se
    pval = 2 * stats.t.sf(np.abs(tval), df)

    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    influence = projector[j, :, None] * resid / np.maximum(1.0 - leverage[:, None], 1e-8)
    hc3_se = np.sqrt(np.sum(influence * influence, axis=0))
    hc3_t = coef[j] / hc3_se
    hc3_p = 2 * stats.t.sf(np.abs(hc3_t), df)

    cluster_codes = pd.Categorical(clusters).codes
    cluster_scores = []
    score = projector[j, :, None] * resid
    for cluster in np.unique(cluster_codes):
        cluster_scores.append(np.sum(score[cluster_codes == cluster, :], axis=0))
    cluster_scores = np.vstack(cluster_scores)
    cluster_count = cluster_scores.shape[0]
    correction = (cluster_count / (cluster_count - 1)) * ((x.shape[0] - 1) / df)
    cluster_se = np.sqrt(correction * np.sum(cluster_scores * cluster_scores, axis=0))
    cluster_t = coef[j] / cluster_se
    cluster_p = 2 * stats.t.sf(np.abs(cluster_t), cluster_count - 1)
    return coef[j], se, pval, hc3_se, hc3_p, cluster_se, cluster_p


metadata = pd.read_csv(META, sep="\t", keep_default_na=True)
metadata.loc[metadata["sex"] == "NA", "sex"] = np.nan
metadata = metadata.loc[metadata["age"].notna() & metadata["sex"].isin(["F", "M"])].copy()
metadata = metadata.set_index("geo_accession", drop=False)

designs = {
    "unadjusted": design_matrix(metadata, "transformed", "none"),
    "cell_adjusted": design_matrix(metadata, "transformed", "cell"),
    "fully_adjusted": design_matrix(metadata, "transformed", "full"),
    "fully_adjusted_linear_age": design_matrix(metadata, "linear", "full"),
}

q1 = pd.read_parquet(Q1)
q2 = pd.read_parquet(Q2)
q1 = q1.rename(columns={c: f"{c}_1" for c in q1.columns if c != "probe"})
q2 = q2.rename(columns={c: f"{c}_2" for c in q2.columns if c != "probe"})
qc = q1.merge(q2, on="probe", validate="one_to_one")
qc["detection_fail_n"] = qc["detection_fail_n_1"] + qc["detection_fail_n_2"]
qc["beta_missing_n"] = qc["beta_missing_n_1"] + qc["beta_missing_n_2"]

manifest = pd.read_csv(MANIFEST, usecols=["IlmnID", "CHR", "MAPINFO", "Genome_Build"])
manifest = manifest.rename(columns={"IlmnID": "probe", "CHR": "chr", "MAPINFO": "position", "Genome_Build": "genome_build"})
qc = qc.merge(manifest, on="probe", how="left", validate="one_to_one")
qc["chr"] = qc["chr"].astype(str)
qc["genome_build"] = qc["genome_build"].astype(str)
quality_mask = set(pd.read_csv(QUALITY_MASK).iloc[:, 0].astype(str))
qc["autosomal"] = qc["chr"].astype(str).isin([str(i) for i in range(1, 23)])
qc["quality_masked"] = qc["probe"].isin(quality_mask)
qc["analysis_eligible"] = (
    qc["autosomal"]
    & ~qc["quality_masked"]
    & (qc["detection_fail_n"] <= int(np.floor(0.01 * len(metadata))))
    & (qc["beta_missing_n"] <= int(np.floor(0.01 * len(metadata))))
)
qc[["probe", "chr", "position", "genome_build", "detection_fail_n", "beta_missing_n", "autosomal", "quality_masked", "analysis_eligible"]].to_parquet(
    RESULTS / "probe_qc.parquet", index=False
)
eligible = dict(zip(qc["probe"], qc["analysis_eligible"]))
annotation = qc.set_index("probe")[["chr", "position", "genome_build"]]

pf1 = pq.ParquetFile(P1)
pf2 = pq.ParquetFile(P2)
if pf1.num_row_groups != pf2.num_row_groups:
    raise RuntimeError("Matrix parts have unequal row-group counts")

writers = {}
try:
    for group in range(pf1.num_row_groups):
        left = pf1.read_row_group(group).to_pandas().set_index("probe")
        right = pf2.read_row_group(group).to_pandas().set_index("probe")
        if not left.index.equals(right.index):
            raise RuntimeError(f"Probe-order mismatch in row group {group}")
        beta = pd.concat([left, right], axis=1)
        keep_probes = [probe for probe in beta.index if eligible.get(probe, False)]
        if not keep_probes:
            continue
        beta = beta.loc[keep_probes, metadata.index].T
        values = beta.to_numpy(float)
        means = np.nanmean(values, axis=0)
        missing = np.where(~np.isfinite(values))
        values[missing] = means[missing[1]]
        clipped = np.clip(values, 1e-6, 1 - 1e-6)
        mvalues = np.log2(clipped / (1 - clipped))

        output = pd.DataFrame({"probe": keep_probes})
        ann = annotation.loc[keep_probes]
        output["chr"] = ann["chr"].astype(str).to_numpy()
        output["position"] = ann["position"].to_numpy()
        for name, design in designs.items():
            coef, se, pval, hc3_se, hc3_p, cluster_se, cluster_p = fit_interaction(mvalues, design, metadata["sentrix_id"])
            beta_coef, _, _, _, _, _, _ = fit_interaction(values, design, metadata["sentrix_id"])
            output[f"{name}_interaction_m"] = coef
            output[f"{name}_se"] = se
            output[f"{name}_p"] = pval
            output[f"{name}_hc3_se"] = hc3_se
            output[f"{name}_hc3_p"] = hc3_p
            output[f"{name}_cluster_se"] = cluster_se
            output[f"{name}_cluster_p"] = cluster_p
            output[f"{name}_interaction_beta"] = beta_coef

        table = pa.Table.from_pandas(output, preserve_index=False)
        if "interaction" not in writers:
            writers["interaction"] = pq.ParquetWriter(RESULTS / "age_sex_interaction_results_unadjusted.parquet", table.schema, compression="zstd")
        writers["interaction"].write_table(table)
finally:
    for writer in writers.values():
        writer.close()

results = pd.read_parquet(RESULTS / "age_sex_interaction_results_unadjusted.parquet")
for name in designs:
    results[f"{name}_fdr"] = bh_adjust(results[f"{name}_p"])
    results[f"{name}_hc3_fdr"] = bh_adjust(results[f"{name}_hc3_p"])
    results[f"{name}_cluster_fdr"] = bh_adjust(results[f"{name}_cluster_p"])
results.to_parquet(RESULTS / "age_sex_interaction_results.parquet", index=False)
results.nsmallest(1000, "fully_adjusted_hc3_p").to_csv(RESULTS / "top1000_age_sex_interactions.tsv", sep="\t", index=False)

summary = {
    "samples": int(len(metadata)),
    "female": int((metadata["sex"] == "F").sum()),
    "male": int((metadata["sex"] == "M").sum()),
    "total_probes": int(len(qc)),
    "autosomal_probes": int(qc["autosomal"].sum()),
    "quality_masked": int(qc["quality_masked"].sum()),
    "detection_fail_filter": int((qc["detection_fail_n"] > int(np.floor(0.01 * len(metadata)))).sum()),
    "eligible_probes": int(qc["analysis_eligible"].sum()),
}
for name in designs:
    summary[f"{name}_fdr_lt_0.05"] = int((results[f"{name}_fdr"] < 0.05).sum())
    summary[f"{name}_hc3_fdr_lt_0.05"] = int((results[f"{name}_hc3_fdr"] < 0.05).sum())
    summary[f"{name}_cluster_fdr_lt_0.05"] = int((results[f"{name}_cluster_fdr"] < 0.05).sum())
with open(RESULTS / "interaction_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
