# Reproducibility notes

The principal script expects prepared matrices under `results/`:

- `matrix1_beta.parquet` and `matrix2_beta.parquet`
- `matrix1_detection_qc.parquet` and `matrix2_detection_qc.parquet`
- `sample_metadata.tsv`

It also expects the Illumina HumanMethylation450 manifest at `downloads/HumanMethylation450k_15017482_v3.csv.gz`. The primary reliability mask is loaded from the installed methylprep package. These large or externally distributed inputs are not included in this archive.

The supplied output files are sufficient to audit the reported counts, ranked interactions, leukocyte-composition results, probe flags, and plotted summaries. A full rerun additionally requires downloading GSE87571 source files and reconstructing the prepared matrices using the same sample order and metadata fields used by the scripts.

The manuscript transparently notes that a complete raw-IDAT rerun was not used because multi-batch consolidation exceeded the original analysis workspace memory ceiling.

