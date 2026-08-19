# Robust age-by-sex interactions in whole-blood DNA methylation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22019149.svg)](https://doi.org/10.5281/zenodo.22019149)

This repository package supports the reanalysis of the public GSE87571 HumanMethylation450 dataset described in the manuscript *Robust Age-by-Sex Interactions in Whole-Blood DNA Methylation: A Reanalysis of GSE87571*.

## Contents

- `scripts/`: analysis and figure-generation scripts.
- `results/`: reported model summaries and machine-readable probe-level results.
- `figures/`: supplementary figures and additional diagnostic plots.
- `tables/`: the supplementary Excel workbook.
- `docs/`: analysis notes, data-availability information, and release checklist.

## Data source and scope

The source methylation data are public at NCBI GEO under accession [GSE87571](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87571). Raw and processed GEO files are not redistributed in this repository. The included results were generated from 729 participants with nonmissing age and recorded sex. The primary analysis retained 404,783 autosomal CpGs after reliability-mask, detection, and missingness filtering.

## Reproduction

1. Create the software environment: `conda env create -f environment.yml`.
2. Download the required GSE87571 source files and the Illumina HumanMethylation450 annotation manifest.
3. Arrange prepared inputs as documented in `docs/REPRODUCIBILITY.md`.
4. Run `scripts/run_interaction_models.py`, followed by the sensitivity and figure scripts.

The repository contains the reported outputs so readers can inspect all numerical results without rerunning the full array-scale analysis. The scripts use paths relative to the repository root, but the source matrices are intentionally excluded because they are derived from public GEO data and are too large for routine Git hosting.

## Key results

- 13 CpGs passed HC3 FDR < 0.05 in the fully adjusted model.
- 7 CpGs also passed Sentrix-slide-clustered FDR < 0.05.
- Estimated B-cell abundance showed an age-by-sex interaction after correction across six cell-fraction tests.

## Citation and license

Please cite the associated manuscript and the original GSE87571 study. Code is released under the MIT License. Data and numerical results remain subject to the terms of their originating sources.

Version v1.0.1 is permanently archived at [Zenodo](https://doi.org/10.5281/zenodo.22019149).
