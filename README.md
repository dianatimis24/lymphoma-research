# Lymphoma classification from DNA methylation data

Research project for the *Proiect de cercetare* course (Babeș-Bolyai University, Faculty of Mathematics and Computer Science): classification of normal samples and lymphoma subtypes from DNA methylation data produced with different technologies (methylation arrays, RRBS, WGBS, EM-seq).

Author: Diana Timiș. Supervisor: Maria-Iuliana Bocicor.

## Relation to the baseline pipeline

The methodology builds on a baseline pipeline developed in the research group of the supervisor (Twist filtering, DMR identification, feature selection, classification), which is not yet published. This repository contains an independent implementation of that pipeline, written from scratch as a Python package, together with the extensions proposed in this project: a common format for multi-platform data, cross-platform harmonization, region-based features and leakage-free evaluation.

## Structure

```
src/lymphdecoder/   package with the new code
tests/              unit tests (pytest)
```

Data and results are not stored in the repository.

## Installation and tests

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```
