import numpy as np
from src.forecast_pipeline import BayesianBlender


def test_blender_weights_sum_to_one():
    blender = BayesianBlender()
    errors = {
        'hw': np.array([1.0, -1.0, 2.0, -0.5]),
        'rf': np.array([0.5, -0.2, 0.0, 0.1])
    }
    weights, variances = blender.compute_weights(errors)

    assert 'hw' in weights and 'rf' in weights
    assert abs(weights['hw'] + weights['rf'] - 1.0) < 1e-8
    assert variances['hw'] > 0
    assert variances['rf'] > 0
