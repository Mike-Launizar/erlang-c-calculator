import pandas as pd
import numpy as np
from src.forecast_pipeline import FeatureEngineer


def test_feature_engineer_creates_expected_columns():
    # Build a minimal DataFrame with Year, Month, and Volume for 15 months
    dates = pd.date_range(start='2020-01-01', periods=15, freq='MS')
    df = pd.DataFrame({'Volume': np.arange(15)}, index=dates)

    fe = FeatureEngineer()
    df_feat = fe.transform(df)

    # Expect lag_1 to exist and no NaNs
    assert 'lag_1' in df_feat.columns
    assert 'lag_3' in df_feat.columns
    assert 'lag_6' in df_feat.columns
    assert df_feat.isnull().sum().sum() == 0
