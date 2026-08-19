#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"


def horvath_age(age, adult_age=20.0):
    age = np.asarray(age, dtype=float)
    return np.where(age <= adult_age, np.log(age + 1) - np.log(adult_age + 1), (age - adult_age) / (adult_age + 1))


def full_design(meta):
    age = horvath_age(meta["age"]); age_mean = age.mean(); age = age - age_mean
    male = meta["sex"].eq("M").astype(float).to_numpy()
    pieces = [pd.DataFrame({"intercept":1.0,"age":age,"male":male,"age_x_male":age*male}, index=meta.index)]
    pieces += [meta[["CD8T","CD4T","NK","Bcell","Mono"]].astype(float),
               pd.get_dummies(meta["sentrix_id"].astype(str),drop_first=True,dtype=float),
               pd.get_dummies(meta["sentrix_position"].astype(str),drop_first=True,dtype=float),
               pd.get_dummies(meta["year_of_collection"].astype(str),drop_first=True,dtype=float)]
    design = pd.concat(pieces,axis=1); x=design.to_numpy(float); keep=[]; rank=0
    for j in range(x.shape[1]):
        r=np.linalg.matrix_rank(x[:,keep+[j]])
        if r>rank: keep.append(j); rank=r
    return design.iloc[:,keep], age_mean


plt.rcParams.update({"font.size":12,"axes.titlesize":13,"axes.labelsize":12,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10})
r = pd.read_parquet(RESULTS / "age_sex_interaction_results.parquet")
r["chr_num"] = pd.to_numeric(r["chr"],errors="coerce")
p = r.dropna(subset=["chr_num","position","fully_adjusted_hc3_p"]).sort_values(["chr_num","position"]).copy()
maxpos=p.groupby("chr_num")["position"].max(); offset=0; offsets={}; centers={}
for c in range(1,23):
    offsets[c]=offset; centers[c]=offset+maxpos.get(c,0)/2; offset += maxpos.get(c,0)+2e7
p["x"]=[row.position+offsets[int(row.chr_num)] for row in p.itertuples()]
p["y"]=-np.log10(np.maximum(p["fully_adjusted_hc3_p"],np.finfo(float).tiny))
fig,ax=plt.subplots(figsize=(7.1,5.0))
colors=["#375a7f","#8aa6c1"]
for c,g in p.groupby("chr_num"):
    ax.scatter(g.x,g.y,s=3.2,alpha=.72,color=colors[(int(c)-1)%2],rasterized=True)
hc3=p[p.fully_adjusted_hc3_fdr<.05]
ax.scatter(hc3.x,hc3.y,s=29,color="#c9912b",zorder=3,label="HC3 FDR < 0.05")
both=hc3[hc3.fully_adjusted_cluster_fdr<.05]
ax.scatter(both.x,both.y,s=38,color="#b23a48",zorder=4,label="Also slide-clustered FDR < 0.05")
ax.axhline(-np.log10(.05/len(p)),color="#777",ls="--",lw=1,label="Bonferroni 0.05")
ax.set_xticks([centers[c] for c in range(1,23)],[str(c) for c in range(1,23)])
ax.set_xlabel("Chromosome"); ax.set_ylabel("−log10 interaction P")
ax.legend(frameon=False,loc="upper right"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/"panel_A_compact.png",dpi=300,facecolor="white"); plt.close(fig)

meta=pd.read_csv(RESULTS/"sample_metadata.tsv",sep="\t"); meta=meta[meta.age.notna() & meta.sex.isin(["F","M"])].set_index("geo_accession",drop=False)
design,age_mean=full_design(meta); x=design.to_numpy(float); inv=np.linalg.inv(x.T@x)
probes=["cg01620164","cg23256579","cg00167275","cg17107691"]
labels={"cg01620164":"FIGN","cg23256579":"PRR4","cg00167275":"SHLD2 / GLUD1","cg17107691":"KANK2"}
left=pd.read_parquet(RESULTS/"matrix1_beta.parquet",filters=[("probe","in",probes)]).set_index("probe")
right=pd.read_parquet(RESULTS/"matrix2_beta.parquet",filters=[("probe","in",probes)]).set_index("probe")
betas=pd.concat([left,right],axis=1).loc[probes,meta.index]
fig,axes=plt.subplots(2,2,figsize=(7.1,5.0),sharex=True)
grid=np.linspace(14,94,161); gt=horvath_age(grid)-age_mean; female=meta.sex.eq("F").to_numpy()
for ax,probe in zip(axes.flat,probes):
    y=betas.loc[probe].to_numpy(float); y[~np.isfinite(y)]=np.nanmean(y); coef=inv@x.T@y
    nuisance=design.drop(columns=["intercept","age","male","age_x_male"]).mean(axis=0).to_numpy()@coef[4:]
    fp=coef[0]+coef[1]*gt+nuisance; mp=coef[0]+coef[2]+(coef[1]+coef[3])*gt+nuisance
    ax.scatter(meta.loc[female,"age"],y[female],s=8,alpha=.16,color="#ba4f9b",edgecolors="none")
    ax.scatter(meta.loc[~female,"age"],y[~female],s=8,alpha=.16,color="#347db5",edgecolors="none")
    ax.plot(grid,fp,color="#ba4f9b",lw=2.4,label="Female"); ax.plot(grid,mp,color="#347db5",lw=2.4,label="Male")
    fdr=float(r.loc[r.probe.eq(probe),"fully_adjusted_hc3_fdr"].iloc[0])
    ax.set_title(f"{labels[probe]} ({probe})"); ax.text(.03,.04,f"HC3 FDR = {fdr:.2g}",transform=ax.transAxes,fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
for ax in axes[-1,:]: ax.set_xlabel("Age (years)")
for ax in axes[:,0]: ax.set_ylabel("Methylation β")
handles,leg=axes[0,0].get_legend_handles_labels(); fig.legend(handles,leg,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.5,1.01))
fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(OUT/"panel_B_compact.png",dpi=300,facecolor="white"); plt.close(fig)
print("Created compact panels")
