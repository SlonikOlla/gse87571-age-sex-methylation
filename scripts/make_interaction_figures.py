#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)


def horvath_age(age, adult_age=20.0):
    age = np.asarray(age, dtype=float)
    return np.where(age <= adult_age, np.log(age + 1) - np.log(adult_age + 1), (age - adult_age) / (adult_age + 1))


def full_design(meta):
    transformed = horvath_age(meta["age"])
    age_mean = np.mean(transformed)
    age = transformed - age_mean
    male = (meta["sex"] == "M").astype(float).to_numpy()
    pieces = [pd.DataFrame({"intercept": 1.0, "age": age, "male": male, "age_x_male": age * male}, index=meta.index)]
    pieces.append(meta[["CD8T", "CD4T", "NK", "Bcell", "Mono"]].astype(float))
    pieces.append(pd.get_dummies(meta["sentrix_id"].astype(str), prefix="slide", drop_first=True, dtype=float))
    pieces.append(pd.get_dummies(meta["sentrix_position"].astype(str), prefix="position", drop_first=True, dtype=float))
    pieces.append(pd.get_dummies(meta["year_of_collection"].astype(str), prefix="collection", drop_first=True, dtype=float))
    design = pd.concat(pieces, axis=1)
    x = design.to_numpy(float)
    keep, rank = [], 0
    for column in range(x.shape[1]):
        new_rank = np.linalg.matrix_rank(x[:, keep + [column]])
        if new_rank > rank:
            keep.append(column)
            rank = new_rank
    return design.iloc[:, keep], age_mean


def hc3_contrast(y, design, contrast):
    x = design.to_numpy(float)
    inv = np.linalg.inv(x.T @ x)
    coef = inv @ x.T @ y
    resid = y - x @ coef
    leverage = np.sum((x @ inv) * x, axis=1)
    weights = contrast @ inv @ x.T
    se = np.sqrt(np.sum((weights * resid / np.maximum(1 - leverage, 1e-8)) ** 2))
    estimate = contrast @ coef
    df = len(y) - np.linalg.matrix_rank(x)
    p = 2 * stats.t.sf(abs(estimate / se), df)
    return estimate, se, p, coef


results = pd.read_parquet(RESULTS / "age_sex_interaction_results.parquet")
significant = results.loc[results["fully_adjusted_hc3_fdr"] < 0.05].copy()
significant["passes_slide_clustered_fdr"] = significant["fully_adjusted_cluster_fdr"] < 0.05
significant["passes_cell_adjusted_hc3_fdr"] = significant["cell_adjusted_hc3_fdr"] < 0.05
significant["passes_cell_adjusted_clustered_fdr"] = significant["cell_adjusted_cluster_fdr"] < 0.05
significant["conservative_evidence_tier"] = np.select(
    [
        significant["passes_slide_clustered_fdr"]
        & significant["passes_cell_adjusted_hc3_fdr"]
        & significant["passes_cell_adjusted_clustered_fdr"],
        significant["passes_slide_clustered_fdr"],
    ],
    ["robust_across_full_and_cell_adjusted_models", "robust_in_fully_adjusted_model"],
    default="hc3_only",
)

annotation_columns = [
    "IlmnID", "UCSC_RefGene_Name", "UCSC_RefGene_Group", "UCSC_CpG_Islands_Name",
    "Relation_to_UCSC_CpG_Island", "Enhancer", "DHS",
]
annotation = pd.read_csv(
    ROOT / "idats/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz",
    skiprows=7,
    usecols=annotation_columns,
    low_memory=False,
).rename(columns={"IlmnID": "probe"})
significant = significant.merge(annotation, on="probe", how="left", validate="one_to_one")

current_gene = {
    "cg00167275": "SHLD2 / GLUD1",
    "cg08115371": "TENM4",
    "cg16936953": "VMP1",
    "cg12054453": "VMP1",
    "cg18942579": "VMP1",
    "cg00565090": "NOP53 / SNORD23",
    "cg08280341": "FIGN region",
}
significant["display_gene"] = significant.apply(
    lambda row: current_gene.get(row["probe"], str(row["UCSC_RefGene_Name"]).split(";")[0] if pd.notna(row["UCSC_RefGene_Name"]) else "intergenic"),
    axis=1,
)

meta = pd.read_csv(RESULTS / "sample_metadata.tsv", sep="\t", keep_default_na=True)
meta.loc[meta["sex"] == "NA", "sex"] = np.nan
meta = meta.loc[meta["age"].notna() & meta["sex"].isin(["F", "M"])].copy().set_index("geo_accession", drop=False)
design, transformed_age_mean = full_design(meta)

probe_list = significant["probe"].tolist()
left = pd.read_parquet(RESULTS / "matrix1_beta.parquet", filters=[("probe", "in", probe_list)]).set_index("probe")
right = pd.read_parquet(RESULTS / "matrix2_beta.parquet", filters=[("probe", "in", probe_list)]).set_index("probe")
betas = pd.concat([left, right], axis=1).loc[probe_list, meta.index]

summary_rows = []
prediction_data = {}
for row in significant.itertuples(index=False):
    y = betas.loc[row.probe].to_numpy(float)
    y[~np.isfinite(y)] = np.nanmean(y)
    female_contrast = np.zeros(design.shape[1]); female_contrast[design.columns.get_loc("age")] = 1
    male_contrast = female_contrast.copy(); male_contrast[design.columns.get_loc("age_x_male")] = 1
    female_slope, female_se, female_p, coef = hc3_contrast(y, design, female_contrast)
    male_slope, male_se, male_p, _ = hc3_contrast(y, design, male_contrast)
    summary_rows.append({
        "probe": row.probe,
        "chr": row.chr,
        "position_hg38": row.position,
        "gene": row.display_gene,
        "female_beta_change_per_10_adult_years": female_slope * 10 / 21,
        "female_slope_hc3_p": female_p,
        "male_beta_change_per_10_adult_years": male_slope * 10 / 21,
        "male_slope_hc3_p": male_p,
        "male_minus_female_beta_change_per_10_adult_years": row.fully_adjusted_interaction_beta * 10 / 21,
        "interaction_hc3_p": row.fully_adjusted_hc3_p,
        "interaction_fdr": row.fully_adjusted_hc3_fdr,
        "interaction_slide_clustered_p": row.fully_adjusted_cluster_p,
        "interaction_slide_clustered_fdr": row.fully_adjusted_cluster_fdr,
        "cell_adjusted_interaction_fdr": row.cell_adjusted_hc3_fdr,
        "cell_adjusted_slide_clustered_fdr": row.cell_adjusted_cluster_fdr,
        "unadjusted_interaction_fdr": row.unadjusted_hc3_fdr,
        "linear_age_sensitivity_fdr": row.fully_adjusted_linear_age_hc3_fdr,
        "linear_age_slide_clustered_fdr": row.fully_adjusted_linear_age_cluster_fdr,
        "conservative_evidence_tier": row.conservative_evidence_tier,
        "Illumina_gene_annotation": row.UCSC_RefGene_Name,
        "Illumina_gene_group": row.UCSC_RefGene_Group,
        "CpG_island_relation": row.Relation_to_UCSC_CpG_Island,
    })

    grid = np.linspace(14, 94, 161)
    grid_t = horvath_age(grid) - transformed_age_mean
    nuisance_mean = design.drop(columns=["intercept", "age", "male", "age_x_male"]).mean(axis=0).to_numpy() @ coef[4:]
    female_pred = coef[0] + coef[1] * grid_t + nuisance_mean
    male_pred = coef[0] + coef[2] + (coef[1] + coef[3]) * grid_t + nuisance_mean
    prediction_data[row.probe] = (grid, female_pred, male_pred, y)

summary = pd.DataFrame(summary_rows).sort_values("interaction_hc3_p")
summary.to_csv(RESULTS / "significant_age_sex_interactions.tsv", sep="\t", index=False)

# Manhattan plot
plot = results.copy()
plot["chr_num"] = pd.to_numeric(plot["chr"], errors="coerce")
plot = plot.dropna(subset=["chr_num", "position", "fully_adjusted_hc3_p"]).sort_values(["chr_num", "position"])
chrom_max = plot.groupby("chr_num")["position"].max()
offset, offsets, centers = 0, {}, {}
for chromosome in range(1, 23):
    offsets[chromosome] = offset
    length = chrom_max.get(chromosome, 0)
    centers[chromosome] = offset + length / 2
    offset += length + 2e7
plot["cumulative_position"] = plot.apply(lambda r: r.position + offsets[int(r.chr_num)], axis=1)
plot["minus_log10_p"] = -np.log10(np.maximum(plot["fully_adjusted_hc3_p"], np.finfo(float).tiny))

fig, ax = plt.subplots(figsize=(12, 4.8))
colors = ["#375a7f", "#8aa6c1"]
for chromosome, group in plot.groupby("chr_num"):
    ax.scatter(group["cumulative_position"], group["minus_log10_p"], s=3, alpha=0.65, color=colors[(int(chromosome) - 1) % 2], rasterized=True)
sig_plot = plot.loc[plot["fully_adjusted_hc3_fdr"] < 0.05]
ax.scatter(sig_plot["cumulative_position"], sig_plot["minus_log10_p"], s=18, color="#c9912b", zorder=3, label="HC3 FDR < 0.05")
robust_plot = sig_plot.loc[sig_plot["fully_adjusted_cluster_fdr"] < 0.05]
ax.scatter(robust_plot["cumulative_position"], robust_plot["minus_log10_p"], s=28, color="#b23a48", zorder=4, label="Also slide-clustered FDR < 0.05")
ax.axhline(-np.log10(0.05 / len(plot)), color="#6c757d", linestyle="--", linewidth=1, label="Bonferroni 0.05")
ax.set_xticks([centers[i] for i in range(1, 23)], [str(i) for i in range(1, 23)])
ax.set_xlabel("Chromosome")
ax.set_ylabel("−log10 interaction P")
ax.set_title("Age×sex interaction after leukocyte and technical adjustment")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES / "interaction_manhattan.png", dpi=300)
fig.savefig(FIGURES / "interaction_manhattan.pdf")
plt.close(fig)

# Q-Q plot and genomic inflation estimate
pvals = results["fully_adjusted_hc3_p"].dropna().sort_values().to_numpy()
expected = (np.arange(1, len(pvals) + 1) - 0.5) / len(pvals)
chisq = stats.chi2.isf(pvals, 1)
lambda_gc = np.nanmedian(chisq) / stats.chi2.ppf(0.5, 1)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.scatter(-np.log10(expected), -np.log10(pvals), s=5, color="#375a7f", alpha=0.65, rasterized=True)
limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, limit], [0, limit], color="#666666", linestyle="--", linewidth=1)
ax.set_xlim(0, limit); ax.set_ylim(0, limit)
ax.set_xlabel("Expected −log10 P")
ax.set_ylabel("Observed −log10 P")
ax.set_title(f"Interaction Q–Q plot (λGC = {lambda_gc:.3f})")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "interaction_qq.png", dpi=300)
fig.savefig(FIGURES / "interaction_qq.pdf")
plt.close(fig)

# Six strongest adjusted trajectories, selecting distinct loci where possible.
chosen = []
for row in summary.itertuples(index=False):
    if all(not (str(row.chr) == str(prev.chr) and abs(float(row.position_hg38) - float(prev.position_hg38)) < 10000) for prev in chosen):
        chosen.append(row)
    if len(chosen) == 6:
        break
fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharex=True)
for ax, row in zip(axes.flat, chosen):
    grid, female_pred, male_pred, y = prediction_data[row.probe]
    female = meta["sex"].eq("F").to_numpy()
    ax.scatter(meta.loc[female, "age"], y[female], s=7, alpha=0.14, color="#ba4f9b", edgecolors="none")
    ax.scatter(meta.loc[~female, "age"], y[~female], s=7, alpha=0.14, color="#347db5", edgecolors="none")
    ax.plot(grid, female_pred, color="#ba4f9b", linewidth=2, label="Female")
    ax.plot(grid, male_pred, color="#347db5", linewidth=2, label="Male")
    ax.set_title(f"{row.gene} ({row.probe})", fontsize=10)
    ax.text(0.03, 0.04, f"interaction FDR={row.interaction_fdr:.2g}", transform=ax.transAxes, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[-1, :]: ax.set_xlabel("Age (years)")
for ax in axes[:, 0]: ax.set_ylabel("Methylation β")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.suptitle("Covariate-adjusted trajectories at top age×sex interaction loci", y=0.995, fontsize=14)
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(FIGURES / "top_interaction_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "top_interaction_trajectories.pdf", bbox_inches="tight")
plt.close(fig)

with open(RESULTS / "diagnostic_metrics.txt", "w") as handle:
    handle.write(f"lambda_gc\t{lambda_gc:.6f}\n")
    handle.write(f"significant_fdr_0.05\t{len(significant)}\n")
    handle.write(f"significant_hc3_and_slide_cluster_fdr_0.05\t{significant['passes_slide_clustered_fdr'].sum()}\n")
print(f"Created figures; lambda_GC={lambda_gc:.3f}; significant={len(significant)}")
