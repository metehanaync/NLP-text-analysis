"""Central configuration: file paths, category groupings, and model names."""

DATA_DIR = "datalar"

# Aggregate datasets used on the Main / Major Updates / Bug Fixes tabs.
BUGFIX_DATASET = f"{DATA_DIR}/bugetiketli_veri.csv"
MAJOR_UPDATE_DATASET = f"{DATA_DIR}/etiketli_veri.csv"
HAPPINESS_DATASET = f"{DATA_DIR}/siniflandirilmis_data.csv"
MAIN_DATASET = f"{DATA_DIR}/main_veri.csv"

# (display label, csv path) pairs rendered as per-category expanders.
MAJOR_UPDATE_CATEGORIES = [
    ("AI", f"{DATA_DIR}/siniflandirilmis_ai.csv"),
    ("Boss", f"{DATA_DIR}/siniflandirilmis_boss.csv"),
    ("Level", f"{DATA_DIR}/siniflandirilmis_level.csv"),
    ("Graphics", f"{DATA_DIR}/siniflandirilmis_graphics.csv"),
    ("Difficulty", f"{DATA_DIR}/siniflandirilmis_difficulty.csv"),
]

BUG_FIX_CATEGORIES = [
    ("Freeze", f"{DATA_DIR}/siniflandirilmis_freeze.csv"),
    ("Appstore", f"{DATA_DIR}/siniflandirilmis_appstore.csv"),
    ("Bug", f"{DATA_DIR}/siniflandirilmis_bug.csv"),
    ("Glitch", f"{DATA_DIR}/siniflandirilmis_glitch.csv"),
    ("Crash", f"{DATA_DIR}/siniflandirilmis_crash.csv"),
]

# Labels offered to the zero-shot classifier for each grouping above.
MAJOR_UPDATE_LABELS = [label for label, _ in MAJOR_UPDATE_CATEGORIES]
BUG_FIX_LABELS = [label for label, _ in BUG_FIX_CATEGORIES]

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

SENTIMENT_COLORS = {
    "positive": "#1f77b4",
    "negative": "#d62728",
    "neutral": "#ff7f0e",
}
