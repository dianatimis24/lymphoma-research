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
  features/bin_store.py       per-sample bin means cached on disk
  features/dmr.py             naive DMRs for all pairs of classes
  features/selection.py       training-only imputation, variance filter, mRMR
  models/classifiers.py       XGBoost, random forest, SVM, L1 logistic regression
  evaluation/metrics.py       macro F1, MCC, feature stability (Jaccard)
  evaluation/cv.py            fixed split, nested cross-validation, leaky variant
  utils/reproducibility.py    seeds and record of each run (config, commit, versions)
scripts/process_datasets.py   converts, filters and describes all datasets
configs/                      example configuration files
tests/                        unit tests (pytest)
```

## Processing the datasets

All datasets are described in one YAML file (see `configs/datasets.example.yaml`): where the files are, which reader understands them, the platform, the sample type and the class of each sample. A new input format only needs a new reader class registered under a new name.

```bash
python scripts/process_datasets.py configs/datasets.yaml
```

The command writes one common-format file per sample, filtered by the Twist panel and by coverage, and the metadata file `samples.csv`.

Data and results are not stored in the repository.

## Common format

One tab-separated file per sample, with genome build hg38:

| Column | Meaning |
|---|---|
| `chr` | chromosome (`chr1`, ..., `chr22`, `chrX`, `chrY`, `chrM`) |
| `base` | 1-based position of the cytosine of the CpG on the forward strand |
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
