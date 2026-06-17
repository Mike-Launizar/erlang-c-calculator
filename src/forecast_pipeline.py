# forecast_pipeline.py
# Package the forecasting pipeline into an importable module.

import os
from dataclasses import dataclass

import pyodbc
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.api import ExponentialSmoothing

from dotenv import load_dotenv

# Load environment variables from .env when used as a script
load_dotenv()


class DataLoader:
    def __init__(self, conn_str: str, sql_path: str):
        self.conn_str = conn_str
        self.sql_path = sql_path

    def load(self) -> pd.DataFrame:
        """Load data by executing the SQL query in sql_path using conn_str.

        Returns a DataFrame indexed by a Date column constructed from Year/Month.
        """
        with open(self.sql_path, "r") as f:
            sql_query = f.read()

        conn = pyodbc.connect(self.conn_str)
        df = pd.read_sql(sql_query, conn)
        conn.close()

        # Fill numeric missing values with column means (preserve non-numeric columns).
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Construct a proper datetime index from Year and Month columns.
        df["Year"] = df["Year"].astype(str)
        df["Month"] = df["Month"].astype(str)
        df["Date"] = pd.to_datetime(df["Year"] + "-" + df["Month"] + "-01")
        df.set_index("Date", inplace=True)

        return df


class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = df.copy()

        # Numeric time index for trend
        df_feat["t"] = np.arange(len(df_feat))

        # Cycle features
        df_feat["month"] = df_feat.index.month
        df_feat["quarter"] = df_feat.index.quarter

        # Rolling trend features
        df_feat["rolling_trend_3"] = df_feat["Volume"] - df_feat["Volume"].shift(3)
        df_feat["rolling_trend_6"] = df_feat["Volume"] - df_feat["Volume"].shift(6)

        # Lag features
        for lag in [1, 3, 6]:
            df_feat[f"lag_{lag}"] = df_feat["Volume"].shift(lag)

        return df_feat.dropna()


class HoltWintersModel:
    def __init__(self, seasonal_periods: int = 12):
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, series: pd.Series):
        scenarios = [
            {"trend": "add", "damped_trend": True, "seasonal": "add", "initialization_method": None},
            {"trend": "mul", "damped_trend": True, "seasonal": "add", "initialization_method": "estimated"},
            {"trend": "mul", "damped_trend": True, "seasonal": "add", "initialization_method": "legacy-heuristic"},
            {"trend": "mul", "damped_trend": True, "seasonal": "add", "initialization_method": None},
            {"trend": "add", "damped_trend": False, "seasonal": "add", "initialization_method": "estimated"},
            {"trend": "add", "damped_trend": False, "seasonal": "add", "initialization_method": None},
            {"trend": "mul", "damped_trend": False, "seasonal": "add", "initialization_method": "heuristic"},
            {"trend": "mul", "damped_trend": False, "seasonal": "add", "initialization_method": "legacy-heuristic"},
            {"trend": None, "damped_trend": False, "seasonal": "add", "initialization_method": "estimated"},
            {"trend": None, "damped_trend": False, "seasonal": "add", "initialization_method": "heuristic"},
            {"trend": None, "damped_trend": False, "seasonal": "add", "initialization_method": "legacy-heuristic"},
            {"trend": "add", "damped_trend": True, "seasonal": "mul", "initialization_method": "heuristic"},
            {"trend": "mul", "damped_trend": True, "seasonal": "mul", "initialization_method": "heuristic"},
            {"trend": "mul", "damped_trend": True, "seasonal": "mul", "initialization_method": None},
            {"trend": "add", "damped_trend": False, "seasonal": "mul", "initialization_method": "estimated"},
            {"trend": "add", "damped_trend": False, "seasonal": "mul", "initialization_method": None},
            {"trend": "mul", "damped_trend": False, "seasonal": "mul", "initialization_method": "heuristic"},
            {"trend": "mul", "damped_trend": False, "seasonal": "mul", "initialization_method": None},
            {"trend": None, "damped_trend": False, "seasonal": "mul", "initialization_method": "estimated"},
            {"trend": None, "damped_trend": False, "seasonal": "mul", "initialization_method": None},
            {"trend": "mul", "damped_trend": False, "seasonal": None, "initialization_method": None},
        ]

        best_rmse = np.inf

        for params in scenarios:
            try:
                model = ExponentialSmoothing(
                    series,
                    seasonal_periods=self.seasonal_periods,
                    **params
                ).fit()

                rmse = mean_squared_error(series[1:], model.fittedvalues[1:], squared=False)

                if rmse < best_rmse:
                    best_rmse = rmse
                    self.model = model

            except Exception:
                # Ignore failing scenarios but keep going
                continue

        if self.model is None:
            raise RuntimeError("Holt-Winters fitting failed for all scenarios")

        return self

    def forecast(self, steps: int) -> pd.Series:
        return self.model.forecast(steps)

    def residuals(self) -> pd.Series:
        return self.model.resid


class RandomForestModel:
    def __init__(self):
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 3],
        }

        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        grid.fit(X, y)
        self.model = grid.best_estimator_
        return self

    def forecast(self, X_future: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_future)

    def residuals(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        return y - self.model.predict(X)


class BayesianBlender:
    @staticmethod
    def bayesian_variance(errors: np.ndarray, alpha0: float = 1.0, beta0: float = 1.0) -> float:
        alpha = alpha0 + len(errors) / 2.0
        beta = beta0 + 0.5 * np.sum(errors ** 2)
        # Posterior expected variance (inverse-gamma posterior mean approximation)
        return beta / (alpha - 1.0)

    def compute_weights(self, errors_dict: dict) -> tuple:
        variances = {name: self.bayesian_variance(err) for name, err in errors_dict.items()}

        # Prior preference (higher values favor the model)
        priors = {"hw": 90.0, "rf": 10.0}

        inv_vars = {k: priors.get(k, 1.0) / v for k, v in variances.items()}
        total = sum(inv_vars.values())
        weights = {k: v / total for k, v in inv_vars.items()}
        return weights, variances


@dataclass
class VolumeForecastPipeline:
    horizon: int = 13

    def run(self, df: pd.DataFrame):
        fe = FeatureEngineer()
        df_feat = fe.transform(df)

        X = df_feat[["t", "month", "quarter", "lag_1", "lag_3", "lag_6", "rolling_trend_3", "rolling_trend_6"]]
        y = df_feat["Volume"]

        # Holt-Winters fit
        hw = HoltWintersModel().fit(df["Volume"])
        hw_forecast = hw.forecast(self.horizon)

        # Random forest fit
        rf = RandomForestModel().fit(X, y)

        # Build future X by shifting the tail and updating the t index.
        future_X = X.tail(self.horizon).copy()
        future_X["t"] = np.arange(X["t"].iloc[-1] + 1, X["t"].iloc[-1] + 1 + self.horizon)

        rf_forecast = rf.forecast(future_X)

        blender = BayesianBlender()
        weights, vars_ = blender.compute_weights({
            "hw": hw.residuals().dropna().values,
            "rf": rf.residuals(X, y),
        })

        blended = weights["hw"] * hw_forecast.values + weights["rf"] * rf_forecast

        variance = weights["hw"] ** 2 * vars_["hw"] + weights["rf"] ** 2 * vars_["rf"]

        lower = blended - 1.96 * np.sqrt(variance)
        upper = blended + 1.96 * np.sqrt(variance)

        last_date = df.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=self.horizon, freq="MS")

        forecast_df = pd.DataFrame(
            {
                "Blended_Forecast": blended,
                "Lower_95_CI": lower,
                "Upper_95_CI": upper,
            },
            index=forecast_index,
        )

        return forecast_df, weights


# Minimal plotting helper to keep a light dependency on plotly.
# The original notebook/script used Plotly; keeping function for convenience.
import plotly.graph_objects as go


class ForecastVisualizer:
    @staticmethod
    def plot(df: pd.DataFrame, forecast_df: pd.DataFrame):
        fig = go.Figure()

        x_numeric = np.arange(len(df))
        y_actuals = df["Volume"].values

        slope, intercept = np.polyfit(x_numeric, y_actuals, 1)
        trend_values = slope * x_numeric + intercept

        fig.add_trace(go.Scatter(x=df.index, y=df["Volume"], name="Actuals", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df.index, y=trend_values, mode="lines", name="Historical Trendline", line=dict(color="black", dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Blended_Forecast"], name="Blended Forecast", line=dict(color="red"), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Upper_95_CI"], fill=None, mode="lines", line_color="lightgrey", showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Lower_95_CI"], fill="tonexty", mode="lines", line_color="lightgrey", name="Credible Interval"))

        fig.update_layout(title="Volume Forecast", xaxis_title="Date", yaxis_title="Volume", template="plotly_white")

        trend_text = f"Trend slope: {slope:,.2f} per month\nDirection: {'Upward' if slope > 0 else 'Downward'}"

        fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.95, text=trend_text, showarrow=False, align="right", font=dict(size=12), bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)

        fig.show()


# Optional CLI entrypoint for local runs. Reads connection configuration from environment.
def build_conn_str_from_env() -> str:
    # Expected env variables: SQL_DRIVER, SQL_SERVER, SQL_DATABASE, TRUSTED_CONNECTION (yes/no), UID, PWD
    driver = os.getenv("SQL_DRIVER", "SQL Server")
    server = os.getenv("SQL_SERVER", "(your-server)")
    database = os.getenv("SQL_DATABASE", "(your-database)")
    trusted = os.getenv("TRUSTED_CONNECTION", "yes")

    if trusted.lower() in ["yes", "true", "1"]:
        conn = f"Driver={{{driver}}};Server={server};Database={database};Trusted_Connection=yes;"
    else:
        uid = os.getenv("SQL_UID")
        pwd = os.getenv("SQL_PWD")
        conn = f"Driver={{{driver}}};Server={server};Database={database};UID={uid};PWD={pwd};"

    return conn


if __name__ == "__main__":
    # Example run when executed as a script.
    # Requires a .env file (see .env.example) and a SQL file path.
    SQL_PATH = os.getenv("SQL_PATH", "query.sql")
    conn = build_conn_str_from_env()

    loader = DataLoader(conn, SQL_PATH)
    df = loader.load()

    pipeline = VolumeForecastPipeline(horizon=int(os.getenv("FORECAST_HORIZON", 13)))
    forecast_df, weights = pipeline.run(df)

    print("\n======= BLENDED FORECAST (COPY / PASTE READY) =======\n")
    print(forecast_df.round(2))

    print("\nBayesian Model Weights:")
    for k, v in weights.items():
        print(f"{k}: {v:.2%}")
