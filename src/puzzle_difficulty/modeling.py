from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .constants import RANDOM_STATE


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def scaled_model(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def numeric_models() -> dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": scaled_model(LinearRegression()),
        "ridge": scaled_model(Ridge(alpha=1.0)),
        "lasso": scaled_model(Lasso(alpha=0.001, max_iter=10_000, random_state=RANDOM_STATE)),
        "elastic_net": scaled_model(
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10_000, random_state=RANDOM_STATE)
        ),
        "svr_rbf": scaled_model(SVR(kernel="rbf", C=1.0, epsilon=0.05)),
        "knn_5": scaled_model(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    }


def nlp_preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("desc_tfidf", TfidfVectorizer(max_features=500), "lemma_description"),
            ("ans_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=300), "lemma_answer"),
            ("num", StandardScaler(), numeric_columns),
        ]
    )


def nlp_models(numeric_columns: list[str]) -> dict[str, object]:
    preprocessor = nlp_preprocessor(numeric_columns)

    def make_model(model):
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    return {
        "ridge": make_model(Ridge(alpha=1.0)),
        "lasso": make_model(Lasso(alpha=0.001, max_iter=10_000, random_state=RANDOM_STATE)),
        "elastic_net": make_model(
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10_000, random_state=RANDOM_STATE)
        ),
        "gradient_boosting": make_model(
            GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
        ),
        "random_forest": make_model(RandomForestRegressor(n_estimators=300, max_depth=5, random_state=RANDOM_STATE)),
    }


def effnet_models(n_components: int) -> dict[str, object]:
    def make_model(model):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
                ("model", model),
            ]
        )

    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge_pca": make_model(Ridge(alpha=10.0)),
        "elastic_net_pca": make_model(
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10_000, random_state=RANDOM_STATE)
        ),
        "svr_rbf_pca": make_model(SVR(kernel="rbf", C=1.0, epsilon=0.05)),
        "random_forest_pca": make_model(
            RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE)
        ),
        "extra_trees_pca": make_model(
            ExtraTreesRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE)
        ),
        "gradient_boosting_pca": make_model(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )
        ),
    }


def combined_preprocessor(
    best_text_approach: str,
    text_columns: list[str],
    effnet_columns: list[str],
    n_components: int,
    nlp_numeric_columns: list[str] | None = None,
) -> ColumnTransformer:
    effnet_transformer = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ]
    )

    if best_text_approach == "nlp":
        if nlp_numeric_columns is None:
            raise ValueError("nlp_numeric_columns is required for the NLP combined preprocessor")
        return ColumnTransformer(
            transformers=[
                ("desc_tfidf", TfidfVectorizer(max_features=500), "lemma_description"),
                ("ans_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=300), "lemma_answer"),
                ("num", StandardScaler(), nlp_numeric_columns),
                ("effnet", effnet_transformer, effnet_columns),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("text", StandardScaler(), text_columns),
            ("effnet", effnet_transformer, effnet_columns),
        ]
    )


def combined_models(preprocessor: ColumnTransformer) -> dict[str, object]:
    def make_model(model):
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    return {
        "dummy_mean": make_model(DummyRegressor(strategy="mean")),
        "linear_regression": make_model(LinearRegression()),
        "ridge": make_model(Ridge(alpha=10.0)),
        "elastic_net": make_model(
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10_000, random_state=RANDOM_STATE)
        ),
        "svr_rbf": make_model(SVR(kernel="rbf", C=1.0, epsilon=0.05)),
        "knn_5": make_model(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "random_forest": make_model(
            RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE)
        ),
        "extra_trees": make_model(
            ExtraTreesRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE)
        ),
        "gradient_boosting": make_model(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate_models(approach_name: str, models: dict[str, object], X, y) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": make_scorer(rmse_score, greater_is_better=False),
    }
    rows = []
    for model_name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        rows.append(
            {
                "approach": approach_name,
                "model": model_name,
                "r2_mean": scores["test_r2"].mean(),
                "r2_std": scores["test_r2"].std(),
                "mae_mean": -scores["test_mae"].mean(),
                "mae_std": scores["test_mae"].std(),
                "rmse_mean": -scores["test_rmse"].mean(),
                "rmse_std": scores["test_rmse"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)

