DIFFICULTY_FORMULA = """
If started == 0, difficulty = 0.
Otherwise:
difficulty = 0.6 * (1 - solved / started)
           + 0.25 * (users_with_hints / started)
           + 0.15 * (total_hints / started)
""".strip()

TARGET = "difficulty"

ALLOWED_RAW_INPUTS = ["answer", "description", "img_url"]

FORMULA_COLUMNS = ["started", "solved", "users_with_hints", "total_hints"]

LEAKAGE_COLUMNS = [
    "started",
    "solved",
    "users_with_hints",
    "total_hints",
    "solve_rate",
    "hint_usage",
    "avg_hints",
]

RANDOM_STATE = 42

