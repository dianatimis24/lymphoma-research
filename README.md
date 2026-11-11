# Lymphoma classification from DNA methylation data

Research project for the *Proiect de cercetare* course (Babeș-Bolyai University, Faculty of Mathematics and Computer Science): classification of normal samples and lymphoma subtypes from DNA methylation data produced with different technologies (methylation arrays, RRBS, WGBS, EM-seq).

Author: Diana Timiș. Supervisor: Maria-Iuliana Bocicor.

## Relation to the baseline pipeline

The methodology builds on a baseline pipeline developed in the research group of the supervisor (Twist filtering, DMR identification, feature selection, classification), which is not yet published. This repository contains an independent implementation of that pipeline, written from scratch as a Python package, together with the extensions proposed in this project: a common format for multi-platform data, cross-platform harmonization, region-based features and leakage-free evaluation.

## Structure

```
src/lymphdecoder/   package with the new code
  io/formats.py     converters to the common methylation format
tests/              unit tests (pytest)
```

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
