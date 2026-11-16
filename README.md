# Lymphoma classification from DNA methylation data

Research project for the *Proiect de cercetare* course (Babeș-Bolyai University, Faculty of Mathematics and Computer Science): classification of normal samples and lymphoma subtypes from DNA methylation data produced with different technologies (methylation arrays, RRBS, WGBS, EM-seq).

Author: Diana Timiș. Supervisor: Maria-Iuliana Bocicor.

## Relation to the baseline pipeline

The methodology builds on a baseline pipeline developed in the research group of the supervisor (Twist filtering, DMR identification, feature selection, classification), which is not yet published. This repository contains an independent implementation of that pipeline, written from scratch as a Python package, together with the extensions proposed in this project: a common format for multi-platform data, cross-platform harmonization, region-based features and leakage-free evaluation.

## Structure

```
src/lymphdecoder/
  io/formats.py               common methylation format (validation, reading, writing)
  io/readers.py               one reader class per input format, registered by name
  processing.py               configuration-driven processing of all datasets
  data.py                     sample metadata (class, platform, dataset, sample type)
  preprocessing/qc.py         Twist panel and coverage filtering
  features/regions.py         bins and mean methylation of regions
  features/annotations.py     CpG islands and promoters from UCSC tables or BED files
  features/bin_store.py       per-sample means of bins or annotated regions, cached on disk
  features/dmr.py             DMRs for all pairs of classes, platform-specific density filter
  features/selection.py       training-only imputation, variance filter, mRMR
  models/classifiers.py       XGBoost, random forest, SVM, L1 logistic regression
  evaluation/metrics.py       macro F1, MCC, feature stability (Jaccard)
  evaluation/cv.py            fixed split, nested cross-validation, leaky variant
  evaluation/exploration.py   data summaries, platform overlap, clustering agreement (ARI)
  evaluation/plots.py         confusion matrices and UMAP figures
  experiments.py              experiments E0-E5 of the case study
  utils/reproducibility.py    seeds and record of each run (config, commit, versions)
scripts/process_datasets.py   converts, filters and describes all datasets
scripts/run_experiments.py    runs the experiments of the case study
configs/                      example configuration files
tests/                        unit tests (pytest)
```

## Processing the datasets

All datasets are described in one YAML file (see `configs/case_study_datasets.yaml` and, for a run without data, `configs/demo_datasets.yaml`): where the files are, which reader understands them, the platform, the sample type and the class of each sample. A new input format only needs a new reader class registered under a new name.

```bash
python scripts/process_datasets.py configs/case_study_datasets.yaml
```

The command writes one common-format file per sample, filtered by the Twist panel and by coverage, and the metadata file `samples.csv`.

## Experiments

The experiments of the case study are described in one YAML file (see `configs/case_study_experiments.yaml`; `configs/validation_experiments.yaml` is the template for the full dataset, including E0):

| Experiment | Question |
|---|---|
| E0 | Does this implementation reproduce the baseline on its fixed train/test split? |
| E1 | Are the converted samples plausible (number of CpGs, methylation, coverage)? |
| E2 | How many regions do the platforms measure, and how many do they share? |
| E3 | Do the samples group by class, or by platform and sample type? |
| E4 | Which feature extraction strategy works best (DMRs on bins, CpG islands, promoters)? |
| E5 | How much does information leakage overestimate the performance? |

```bash
python scripts/run_experiments.py configs/case_study_experiments.yaml all
```

Each experiment writes its tables, figures and a `run.json` (configuration, git commit, package versions) to `results/<experiment>/`.

The annotations for E4 are the UCSC hg38 tables `cpgIslandExt.txt.gz` and `ncbiRefSeq.txt.gz` from <https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/>.

Arrays measure far fewer CpGs than sequencing, so the minimum number of CpGs per region (`min_cpgs`) can be set per platform.

The raw and processed data are not stored in the repository.

## Common format

One tab-separated file per sample, with genome build hg38:

| Column | Meaning |
|---|---|
| `chr` | chromosome (`chr1`, ..., `chr22`, `chrX`, `chrY`, `chrM`) |
| `base` | 1-based position of the cytosine |
| `strand` | `+` or `-` for a measurement of one strand; `*` if the strand is unknown (arrays, WGBS) or the two strands of the CpG were combined (then `base` is the C on the + strand) |
| `coverage` | number of reads covering the CpG (`NA` for arrays) |
| `freqC` | methylation percentage, between 0 and 100 |

## Installation and tests

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```
