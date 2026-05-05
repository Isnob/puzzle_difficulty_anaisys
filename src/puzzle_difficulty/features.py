from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .constants import FORMULA_COLUMNS, LEAKAGE_COLUMNS, TARGET

VOWELS = set("аеёиоуыэюя")
RARE_LETTERS = set("фщъёцэ")


def calculate_difficulty(frame: pd.DataFrame) -> pd.Series:
    started = frame["started"].replace(0, np.nan)
    calculated = (
        0.6 * (1 - frame["solved"] / started)
        + 0.25 * (frame["users_with_hints"] / started)
        + 0.15 * (frame["total_hints"] / started)
    )
    return calculated.fillna(0)


def validate_no_leakage(feature_names: list[str] | pd.Index) -> None:
    leakage = sorted(set(feature_names) & set(LEAKAGE_COLUMNS))
    if leakage:
        raise ValueError(f"Leakage columns are not allowed as model features: {leakage}")


def validate_difficulty_formula(frame: pd.DataFrame) -> pd.DataFrame:
    check = frame[[TARGET, *FORMULA_COLUMNS]].copy()
    check["difficulty_calculated"] = calculate_difficulty(frame)
    check["absolute_diff"] = (check[TARGET] - check["difficulty_calculated"]).abs()
    return check


def first_answer_variant(answer: object) -> str:
    if pd.isna(answer):
        return ""
    return str(answer).split("|")[0].strip()


def extract_text_stats(text: object, prefix: str = "") -> pd.Series:
    if pd.isna(text):
        text = ""

    normalized = first_answer_variant(text).lower()
    compact = normalized.replace(" ", "")
    words = [word for word in normalized.split() if word]

    letters = [char for char in compact if char.isalpha()]
    cyrillic_letters = [char for char in letters if "а" <= char <= "я" or char == "ё"]
    latin_letters = [char for char in letters if "a" <= char <= "z"]

    len_chars = len(compact)
    vowel_count = sum(char in VOWELS for char in compact)
    consonant_count = sum(char.isalpha() and char not in VOWELS for char in compact)
    rare_count = sum(char in RARE_LETTERS for char in compact)
    unique_chars = len(set(compact))

    return pd.Series(
        {
            f"{prefix}len_chars": len_chars,
            f"{prefix}len_words": len(words),
            f"{prefix}avg_word_len": len_chars / len(words) if words else 0.0,
            f"{prefix}vowel_count": vowel_count,
            f"{prefix}consonant_count": consonant_count,
            f"{prefix}rare_count": rare_count,
            f"{prefix}unique_chars": unique_chars,
            f"{prefix}vowel_ratio": vowel_count / len_chars if len_chars else 0.0,
            f"{prefix}rare_ratio": rare_count / len_chars if len_chars else 0.0,
            f"{prefix}unique_ratio": unique_chars / len_chars if len_chars else 0.0,
            f"{prefix}cyrillic_ratio": len(cyrillic_letters) / len(letters) if letters else 0.0,
            f"{prefix}latin_ratio": len(latin_letters) / len(letters) if letters else 0.0,
            f"{prefix}has_dash": int("-" in normalized),
            f"{prefix}has_digits": int(any(char.isdigit() for char in normalized)),
        }
    )


def make_answer_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame["answer"].apply(extract_text_stats)
    validate_no_leakage(features.columns)
    return features


def make_answer_description_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.concat(
        [
            frame["answer"].apply(lambda value: extract_text_stats(value, prefix="ans_")),
            frame["description"].apply(lambda value: extract_text_stats(value, prefix="desc_")),
        ],
        axis=1,
    )
    validate_no_leakage(features.columns)
    return features


def clean_and_lemmatize(text: object, morph=None) -> str:
    if pd.isna(text):
        return "пусто"

    words = re.findall(r"[а-яёa-z]+", str(text).lower())
    if not words:
        return "пусто"

    if morph is None:
        return " ".join(words)

    return " ".join(morph.parse(word)[0].normal_form for word in words)


def extract_rebus_logic(row: pd.Series) -> pd.Series:
    desc = str(row["description"]).lower()
    ans = first_answer_variant(row["answer"]).lower()
    pos_prepositions = [" в ", " на ", " под ", " над ", " за ", " перед ", " из "]
    rebus_terms = ["запятая", "перевернутый", "вверх ногами", "зачеркнуть", "буква", "цифра"]

    return pd.Series(
        {
            "comma_count": desc.count(",") + desc.count("'"),
            "desc_len": len(desc),
            "ans_len": len(ans.replace(" ", "")),
            "has_pos_prep": int(any(prep in desc for prep in pos_prepositions)),
            "has_rebus_terms": int(any(term in desc for term in rebus_terms)),
            "is_multi_word_ans": int(" " in ans.strip()),
            "variant_count": str(row["answer"]).count("|") + 1,
        }
    )


def make_nlp_features(frame: pd.DataFrame, morph=None) -> tuple[pd.DataFrame, list[str]]:
    nlp_frame = frame.copy()
    nlp_frame["lemma_description"] = nlp_frame["description"].apply(
        lambda value: clean_and_lemmatize(value, morph=morph)
    )
    nlp_frame["lemma_answer"] = nlp_frame["answer"].apply(
        lambda value: clean_and_lemmatize(first_answer_variant(value), morph=morph)
    )

    rebus_features = nlp_frame.apply(extract_rebus_logic, axis=1)
    features = pd.concat([nlp_frame[["lemma_description", "lemma_answer"]], rebus_features], axis=1)
    validate_no_leakage(features.columns)
    return features, rebus_features.columns.tolist()


def get_effnet_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("effnet_feat_")]

