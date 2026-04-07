# Health Insurance Pricing Case: Clustering vs. Linear Regression

## Overview

This project compares two different analytical approaches for solving the same
health insurance pricing case:

1. A clustering-based workflow rebuilt from the completed health insurance case.
2. A linear regression workflow rebuilt in Python using the reasoning style from
   the car rental regression reference file.

The main purpose is not only to compare model outputs, but also to build a
clear, interview-ready explanation of the statistical choices behind each step.

## Project Goal

The project is designed to answer the same business problem with two methods and
evaluate both of them on unseen data.

The comparison should follow the same evaluation logic for both approaches:

1. Split the historical customer dataset into 80% training data and 20% test
   data.
2. Rebuild the clustering solution using only the training set.
3. Evaluate the clustering approach on the 20% holdout set and on 3 new
   incoming clients.
4. Rebuild the case with linear regression using the same split logic.
5. Compare both approaches on unseen data to understand their predictive
   behavior, strengths, and limitations.

## Learning Objectives

This repository is also a guided practice environment for:

- understanding the statistical concepts used in each block of analysis;
- improving professional Git and GitHub habits with beginner-friendly workflow;
- documenting code and notebooks in a way that supports job interview
  explanations;
- practicing English in a project environment.

## Repository Structure

```text
health_insurance_case_linear_regression_and_clustering_compairson/
├── data/
├── docs/
├── notebooks/
├── codex_files/
├── reference_files/
├── .gitignore
└── README.md
```

## Planned Workflow

- Reuse the health insurance clustering case as the reference business problem.
- Recreate the clustering pipeline with an explicit train-test split.
- Build a regression-based notebook in Python inspired by the statistical
  commentary style from the car rental R file.
- Compare predictions and interpretation quality between both methods.
- Keep notebook explanations focused on why each statistical technique is being
  used, not only on how the code works.

## Current Status

- Project folders were created.
- `.gitignore` was configured to ignore `codex_files/`, `reference_files/`, and
  `.vscode/`.
- A root-level `docs/` folder was added for project notes and supporting text
  files.
- The project objectives file is currently stored in `docs/Objetivos do
  projeto.txt`.
- The next implementation step is to start building the notebook workflow and
  populate the `data/` and `notebooks/` folders with the working project files.

## Suggested Next Git Step

After reviewing this README, the next repository setup step should be:

```bash
git init
```
