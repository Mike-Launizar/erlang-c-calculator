# Erlang C Calculator - Forecasting pipeline

This branch adds a forecasting pipeline combining Holt-Winters seasonal smoothing and a Random Forest regressor, blended via Bayesian inverse-variance weighting.

Files added
- src/forecast_pipeline.py — importable pipeline and CLI entrypoint
- notebooks/forecast_workflow.ipynb — example notebook showing usage
- requirements.txt — dependencies
- .env.example — example environment variables
- .gitignore — ignores for Python projects
- tests/test_feature_engineer.py — small unit test
- tests/test_blender.py — small unit test

Setup
1. Create a virtual environment and install dependencies:

   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate    # Windows
   pip install -r requirements.txt

2. Create a `.env` file (copy from `.env.example`) and update the values. Do NOT commit `.env` to the repository.

3. Provide your SQL query file and set the SQL_PATH environment variable (or update `SQL_PATH` in the .env file). The SQL should return columns `Year`, `Month`, and `Volume` among any other columns.

Run the pipeline

- From Python:

    from src.forecast_pipeline import DataLoader, VolumeForecastPipeline, build_conn_str_from_env
    import os

    conn = build_conn_str_from_env()
    loader = DataLoader(conn, os.getenv('SQL_PATH', 'query.sql'))
    df = loader.load()

    pipeline = VolumeForecastPipeline(horizon=13)
    forecast_df, weights = pipeline.run(df)

    print(forecast_df)

Security notes
- Do not commit credentials or .env files. Use GitHub Secrets for CI and environment variables in deployed environments.
- Large model artifacts are not committed to this repo. Use Git LFS or external object storage if you need to store pickled models.

