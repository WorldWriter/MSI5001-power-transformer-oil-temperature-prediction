# Repository Guidelines

## Project Structure & Module Organization
Primary automation lives in `scripts/`: `preprocessing/` for data prep, `models/` for baseline and deep-learning trainers, `evaluation/` for visualization, and `analysis/` for seasonal studies. Notebook experiments belong in `notebooks/` with descriptive names and should snapshot outputs to `artifacts/` when promoted. Long-horizon window studies stay under `Windows_diff/` (scripts, docs, models, visualizations). Persist trained assets in `models/` and derived metrics in `results/csv/`. Place shared documentation in `docs/` and keep raw or processed CSV/Numpy files under `dataset/`.

## Build, Test, and Development Commands
Create an isolated environment and install pinned dependencies before running any pipeline:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # add requirements-torch.txt when using deep models
```
The end-to-end baseline pipeline is orchestrated through `python scripts/run_pipeline.py`, which stages data into `artifacts/run_*`. For focused runs, execute `python scripts/preprocessing/optimized_preprocessing.py`, `python scripts/models/simple_ml_models.py`, or `python scripts/evaluation/visualization_analysis.py`. Window-length experiments rely on `python Windows_diff/scripts/run_experiment.py`.

## Coding Style & Naming Conventions
All Python code follows PEP 8 with 4-space indentation, snake_case functions, and CamelCase classes. Prefer explicit type hints and module-level `main()` entry points as seen in `scripts/run_pipeline.py`. Keep imports grouped by origin (standard library, third-party, local) and alphabetized. Name generated files with clear prefixes (`final_model_comparison.csv`, `rf_1h.pkl`) and document notebook parameters in the first cell markdown.

## Testing Guidelines
There is no pytest suite yet; validate changes by rerunning the relevant pipeline segments. New preprocessing or model work should regenerate metrics in `results/csv/` and achieve at least the published R² baselines before merging. Capture key plots in `visualizations/` and note any regression in the PR description. When adding notebooks, include a concluding cell that prints summary metrics to aid peer verification.

## Commit & Pull Request Guidelines
Keep commits small with present-tense summaries (Chinese or English) that describe the change, e.g. `修复 notebooks 中的缺失特征处理`. Reference affected scripts or notebooks in the body if needed. Pull requests must list: scope overview, commands run, updated artifacts or dataset locations, and any outstanding issues. Link related tutorial/docs pages and supply comparison screenshots or CSV diffs when visual outputs change.

## Environment & Data Handling
Large raw files are excluded from git; store them locally under `dataset/raw/` and load them through scripts that copy into timestamped `artifacts/` folders. Never commit personally identifiable data. Update `requirements*.txt` when dependencies change and include installation notes if additional system packages are required.
