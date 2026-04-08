# Health Insurance Pricing Case: Clustering vs. Linear Regression

## Overview

This project compares two different analytical approaches for solving the same
health insurance pricing case:

1. a clustering-based approach, rebuilt from the completed health insurance case;
2. a linear regression approach, using multiple regression and model refinement through stepwise AIC evaluation.

The main purpose is not only to compare model outputs, but also to help a
health insurance company decide how much to charge for its plans in a way that
reduces the chance of financial losses while supporting customer acquisition.
The project is also designed to build a clear, interview-ready explanation of
the statistical choices behind each step.

## Project Goal

The project is designed to answer the same business problem with two methods and
evaluate both of them on unseen data.

The business question is broader than rebuilding a clustering notebook. The
real goal is to compare two analytical strategies for pricing health insurance
plans and to understand which approach produces more useful information for
premium-setting decisions.

The comparison should follow the same evaluation logic for both approaches:

1. Split the historical customer dataset into 80% training data and 20% test
   data.
2. Rebuild the clustering solution using only the training set.
3. Evaluate the clustering approach on the 20% holdout set and on 3 new
   incoming clients.
4. Rebuild the case with linear regression using the same split logic.
5. Compare both approaches on unseen data to understand their predictive
   behavior, strengths, and limitations.

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