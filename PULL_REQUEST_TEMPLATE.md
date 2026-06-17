---
title: Add forecasting pipeline (Holt-Winters + Random Forest + Bayesian blending)
---

This pull request adds a new forecasting pipeline that combines classical Holt-Winters seasonal smoothing with a Random Forest regression model, and blends the two forecasts using Bayesian inverse-variance weighting. The goal is to provide a robust 13-month volume forecast with a 95% credible interval and an example notebook showing usage.

Files added:
- src/forecast_pipeline.py — importable pipeline and CLI entrypoint
- notebooks/forecast_workflow.ipynb — example notebook
- requirements.txt — dependency list
- README.md — usage and security notes
- .env.example — example environment variables (do NOT commit real credentials)
- .gitignore — Python ignores
- tests/test_feature_engineer.py — unit test
- tests/test_blender.py — unit test

Notes for reviewers:
- The SQL query file is not included. Provide a SQL file that returns Year, Month, and Volume columns.
- No model binaries are committed. If you want to store pickled models, consider Git LFS or external storage.
- Run pytest to execute the small unit tests.

How to test locally:
1. Copy .env.example to .env and fill in SQL settings and path to your SQL query file.
2. pip install -r requirements.txt
3. python src/forecast_pipeline.py
4. pytest -q

