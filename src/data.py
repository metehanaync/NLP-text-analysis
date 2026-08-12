"""Cached CSV loading and date normalization."""

import pandas as pd
import streamlit as st

from src.config import (
    BUGFIX_DATASET,
    HAPPINESS_DATASET,
    MAIN_DATASET,
    MAJOR_UPDATE_DATASET,
)

# All source CSVs use US-style M/D/Y dates; parsing with an explicit format
# avoids the ambiguous cases (e.g. "11/4/2021") that a locale-guessing parser
# would otherwise misread.
DATE_FORMAT = "%m/%d/%Y"


@st.cache_data
def load_main_datasets() -> dict[str, pd.DataFrame]:
    """Load the four aggregate datasets used on the Main/Major Updates/Bug Fixes tabs."""
    try:
        df_bugfix = pd.read_csv(BUGFIX_DATASET)
        df_major = pd.read_csv(MAJOR_UPDATE_DATASET)
        df_happiness = pd.read_csv(HAPPINESS_DATASET)
        df_main = pd.read_csv(MAIN_DATASET)
    except FileNotFoundError as exc:
        st.error(f"Required data file not found: {exc.filename}")
        st.stop()

    df_bugfix["date"] = pd.to_datetime(df_bugfix["date"], format=DATE_FORMAT).dt.date
    df_major["date"] = pd.to_datetime(df_major["date"], format=DATE_FORMAT).dt.date
    df_happiness["Date"] = pd.to_datetime(df_happiness["Date"], format=DATE_FORMAT).dt.date
    df_main["date"] = pd.to_datetime(df_main["date"], format=DATE_FORMAT).dt.date

    return {
        "bugfix": df_bugfix,
        "major": df_major,
        "happiness": df_happiness,
        "main": df_main,
    }


@st.cache_data
def load_category_data(csv_path: str) -> pd.DataFrame:
    """Load a single per-category CSV, sorted by its own Date column."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT).dt.date
    return df.sort_values(by="Date")
