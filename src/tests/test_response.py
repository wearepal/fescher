import numpy as np
import pytest

from src.dynamics.response import LinearResponse, RIRResponse


def test_linear_response():
    rng = np.random.default_rng(0)
    n = 19
    c = 5
    features = rng.normal(loc=0, scale=1, size=(n, c))
    action = rng.normal(loc=0, scale=0.5, size=(c,))

    response_fn = LinearResponse(epsilon=1.0, changeable_features=None)
    new_features = response_fn.respond(features=features, action=action)
    assert new_features.shape == features.shape

    changeable_features = np.array([0, 2], dtype=np.uint8)
    response_fn = LinearResponse(epsilon=1.0, changeable_features=changeable_features)
    new_features = response_fn.respond(features=features, action=action)
    assert new_features.shape == features.shape

    missized_action = rng.normal(loc=0, scale=0.5, size=(n,))
    with pytest.raises(ValueError):
        response_fn.respond(features=features, action=missized_action)


def test_rir_response():
    rng = np.random.default_rng(0)
    n = 19
    c = 5
    features = rng.integers(low=0, high=4, size=(n, c)).astype(np.float64)
    action = rng.uniform(low=0, high=1, size=(n,))

    response_fn = RIRResponse(epsilon=1.0, changeable_features=None)
    new_features = response_fn(features=features, action=action)
    with pytest.raises(AssertionError):
        assert np.testing.assert_array_equal(features, new_features)

    changeable_features = np.array([0, 2], dtype=np.uint8)
    response_fn = RIRResponse(epsilon=1.0, changeable_features=changeable_features)
    new_features = response_fn(features=features, action=action)

    assert new_features.shape == features.shape
