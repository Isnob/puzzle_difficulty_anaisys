from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import pymorphy3
except ImportError:  # pragma: no cover - notebook/script environment convenience
    pymorphy3 = None

from src.puzzle_difficulty.constants import RANDOM_STATE, TARGET
from src.puzzle_difficulty.features import (
    get_effnet_columns,
    make_answer_description_features,
    make_answer_features,
    make_nlp_features,
    validate_difficulty_formula,
)
from src.puzzle_difficulty.modeling import (
    combined_models,
    combined_preprocessor,
    effnet_models,
    evaluate_models,
    nlp_models,
    numeric_models,
)


DATA_PATH = Path("data_with_efficientnet_features.csv")
REPORTS_DIR = Path("reports")


def build_approaches(df: pd.DataFrame):
    morph = pymorphy3.MorphAnalyzer() if pymorphy3 is not None else None

    answer_features = make_answer_features(df)
    answer_description_features = make_answer_description_features(df)
    nlp_features, nlp_numeric_columns = make_nlp_features(df, morph=morph)

    effnet_columns = get_effnet_columns(df)
    if not effnet_columns:
        raise ValueError("No effnet_feat_* columns found. Run extract_efficientnet_features.py first.")

    n_effnet_components = min(50, len(df) - 1, len(effnet_columns))

    approaches = {
        "answer_only": {
            "X": answer_features,
            "models": numeric_models(),
            "kind": "numeric_text",
            "text_columns": answer_features.columns.tolist(),
        },
        "answer_description": {
            "X": answer_description_features,
            "models": numeric_models(),
            "kind": "numeric_text",
            "text_columns": answer_description_features.columns.tolist(),
        },
        "nlp": {
            "X": nlp_features,
            "models": nlp_models(nlp_numeric_columns),
            "kind": "nlp",
            "nlp_numeric_columns": nlp_numeric_columns,
        },
        "efficientnet_only": {
            "X": df[effnet_columns].copy(),
            "models": effnet_models(n_effnet_components),
            "kind": "image",
        },
    }

    return approaches, effnet_columns, n_effnet_components


def add_best_text_plus_effnet(df: pd.DataFrame, approaches, effnet_columns, n_effnet_components, best_text_approach):
    best_setup = approaches[best_text_approach]
    X_text = best_setup["X"]
    X_combined = pd.concat([X_text, df[effnet_columns]], axis=1)

    if best_setup["kind"] == "nlp":
        preprocessor = combined_preprocessor(
            best_text_approach="nlp",
            text_columns=[],
            effnet_columns=effnet_columns,
            n_components=n_effnet_components,
            nlp_numeric_columns=best_setup["nlp_numeric_columns"],
        )
    else:
        preprocessor = combined_preprocessor(
            best_text_approach=best_text_approach,
            text_columns=best_setup["text_columns"],
            effnet_columns=effnet_columns,
            n_components=n_effnet_components,
        )

    approaches[f"{best_text_approach}_plus_efficientnet"] = {
        "X": X_combined,
        "models": combined_models(preprocessor),
        "kind": "combined",
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET].copy()

    difficulty_check = validate_difficulty_formula(df)
    print("Difficulty formula check:")
    print(f"  max absolute diff: {difficulty_check['absolute_diff'].max():.8f}")
    print(f"  mean absolute diff: {difficulty_check['absolute_diff'].mean():.8f}")

    approaches, effnet_columns, n_effnet_components = build_approaches(df)

    base_metrics = pd.concat(
        [evaluate_models(name, setup["models"], setup["X"], y) for name, setup in approaches.items()],
        ignore_index=True,
    ).sort_values("mae_mean")

    text_metrics = base_metrics[base_metrics["approach"].isin(["answer_only", "answer_description", "nlp"])]
    best_text_approach = text_metrics.iloc[0]["approach"]
    print(f"Best text approach by CV MAE: {best_text_approach}")

    add_best_text_plus_effnet(df, approaches, effnet_columns, n_effnet_components, best_text_approach)
    combined_name = f"{best_text_approach}_plus_efficientnet"
    combined_metrics = evaluate_models(combined_name, approaches[combined_name]["models"], approaches[combined_name]["X"], y)

    all_metrics = pd.concat([base_metrics, combined_metrics], ignore_index=True).sort_values("mae_mean").reset_index(drop=True)
    best_by_approach = all_metrics.groupby("approach", as_index=False).first().sort_values("mae_mean")

    REPORTS_DIR.mkdir(exist_ok=True)
    all_metrics.to_csv(REPORTS_DIR / "all_model_metrics.csv", index=False)
    best_by_approach.to_csv(REPORTS_DIR / "best_model_by_approach.csv", index=False)

    print("\nBest model by approach:")
    print(best_by_approach[["approach", "model", "mae_mean", "rmse_mean", "r2_mean"]])
    print(f"\nSaved reports to {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()

