# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Health Insurance Pricing Case: Clustering vs. Linear Regression**

A data science project comparing two analytical approaches to solve a health insurance pricing problem:
- K-Means clustering approach for customer segmentation and pricing
- Linear regression approach for predictive pricing models

The goal is to evaluate both strategies on unseen data and determine which produces more useful information for premium-setting decisions.

**Main Deliverable:** Jupyter notebook in `notebooks/linear_regression_and_clustering_compairson_health_insurance_case.ipynb`

**Data:** `data/base_health_insurance.csv` (59KB) — historical customer dataset with age, BMI, gender, region, children count, discount eligibility, expenses, and premium.

## Current Work: Model Selection Investigation

The project is entering Phase A/B/C/D of formal model selection (see memory system for plan details):
- **Phase A:** Variable inventory + per-feature linearity tests
- **Phase B:** Train and compare all model families
- **Phase C:** Write theoretical reference doc first
- **Phase D:** Custom approach

User decision pending on which phase to execute.

## Key Instructions

### Language Preference
- **All conversation**: Portuguese only (until user changes decision)
- **Code, commits, markdown, documentation**: English

### File Management & Git
- **NEVER auto-commit** — provide git commands for user to execute (user maintains full control)
- **Use `debug/` folder** for temporary files, scripts, and test artifacts (gitignored)
- **NEVER modify notebooks without permission** — changes require explicit user approval

### Code Validation
- **Always run full validation** before reporting completion
- **Execute the entire notebook** to verify no errors
- **Test thoroughly** — don't say something works until you've verified it runs end-to-end
- **Run validation scripts** if provided — they're in the debug folder or inline in notebook cells

### Development Workflow

**Working with the main notebook:**
```
# Open the notebook (user responsibility — no auto-execution)
notebooks/linear_regression_and_clustering_compairson_health_insurance_case.ipynb

# After making changes, run full notebook validation in Jupyter/VSCode to verify:
# - All cells execute without error
# - Output matches expected format
# - No broken imports or missing dependencies
```

**Adding temporary analysis files:**
```
# Create analysis scripts in debug/ folder (e.g., debug/test_linearity.py)
# Once validated and integrated into main notebook, delete temp file
# Do NOT commit debug/ folder artifacts
```

### Commit Message Format
Use conventional commit prefixes:
- `feat:` — new feature or analysis section
- `fix:` — bug fix in code or notebook
- `refactor:` — restructure code/cells without behavior change
- `chore:` — maintenance (gitignore, artifacts, cleanup)
- `test:` — test additions or validation improvements

Example: `feat: add phase A linearity testing for continuous features`

## Repository Structure

```
├── notebooks/
│   └── linear_regression_and_clustering_compairson_health_insurance_case.ipynb  [MAIN]
├── data/
│   ├── base_health_insurance.csv
│   └── kmeans_model_trained.pkl (gitignored artifact)
├── debug/  (gitignored — temp files, scripts, test artifacts)
├── docs/  (project notes and analysis docs)
├── reference_files/  (external documentation, reference material)
├── codex_files/  (vscode settings and config)
├── .gitignore  (includes: debug/, *.pkl)
└── README.md
```

## Model Families in Comparison

For Phase 3 (model training), implement these families in order:
1. **Linear Regression** (baseline — already exists)
2. **Polynomial Regression** (degree 2 + stepwise selection)
3. **Ridge / Lasso** (regularized)
4. **Random Forest / Gradient Boosting** (non-linear, no shape assumption)
5. **K-Means + cluster-based pricing** (existing — treat as predictive baseline)
6. **GAM** (optional — interpretable non-linearity)

Use same train/test split (80/20) and metrics (R², RMSE, MAE, cross-validation) for all.

## Important Notes on Data

- **Target variable:** `expenses` (continuous — medical expense prediction)
- **Features:**
  - Continuous: `age`, `bmi`
  - Binary: `gender` (male/female), `discount_eligibility` (yes/no)
  - Nominal: `region` (midwest/northeast/south/southeast, one-hot encoded)
  - Ordinal: `children` (0..n)
  - `premium` (current charged value — not used as model feature)
- **Size:** 1,338 rows (split 80/20 train/test)
- **Known preprocessing:** Categorical variables already analyzed with t-test (binary), ANOVA (region), correlation tests

## Environment Setup

Project uses Python with standard ML/stats stack:
- pandas, numpy (data manipulation)
- scikit-learn (models, preprocessing)
- statsmodels (statistical testing, regression diagnostics)
- matplotlib/seaborn (visualization)

Virtual environment: `.venv/` (use `python -m venv .venv` if needed)

Install dependencies from notebook cells (no separate requirements.txt yet).

## Additional References

See memory system (`C:\Users\Usuario\.claude\projects\...\memory\`) for:
- `model_selection_plan.md` — full 6-phase plan with statistical framework
- `reference_doc_estrutura_modelo.md` — summary of System Identification doc (time-series framework adapted to cross-sectional data)
- `session_resume_*.md` — prior session context and decisions
- `feedback_*.md` — specific coding rules and operational preferences
