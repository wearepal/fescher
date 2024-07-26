import numpy as np

from src.dynamics.state import State


def test_len(mock_state: State):
    assert len(mock_state) == 2, "Length of State instance should be 2"


def test_num_features(mock_state: State):
    assert mock_state.num_features == 2, "Number of features should be 2"


def test_asdict(mock_state: State):
    state_dict = mock_state.asdict()
    expected_dict = {
        "features": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "labels": np.array([1, 0], dtype=np.uint8),
    }
    np.testing.assert_array_equal(state_dict["features"], expected_dict["features"])
    np.testing.assert_array_equal(state_dict["labels"], expected_dict["labels"])


def test_state_integrity(mock_state: State):
    assert mock_state.features.shape == (2, 2), "features should be of shape (2, 2)"
    assert mock_state.labels.shape == (2,), "labels should be of shape (2,)"
    assert mock_state.features.dtype == np.float64, "features dtype should be float64"
    assert mock_state.labels.dtype == np.uint8, "labels dtype should be uint8"
