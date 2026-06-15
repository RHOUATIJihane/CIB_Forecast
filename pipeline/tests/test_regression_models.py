"""Tests fabrique modèles régression par profil."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.ml.regression_models import make_regressor, normalize_regression_model, resolve_regression_model


def test_normalize_aliases():
    assert normalize_regression_model("random_forest") == "rf"
    assert normalize_regression_model("ridge") == "ridge"
    assert normalize_regression_model("lgbm") == "lgbm"


def test_make_regressor_types():
    assert isinstance(make_regressor("ridge"), Pipeline)
    assert isinstance(make_regressor("rf"), RandomForestRegressor)
    assert make_regressor("lgbm").__class__.__name__ == "LGBMRegressor"


def test_resolve_from_policy_row():
    rules = {"regression_model": "ridge", "use_externals_regression": True}
    assert resolve_regression_model(rules) == "ridge"
