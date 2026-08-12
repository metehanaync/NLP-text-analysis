"""Chart-building helpers shared by the Streamlit tabs."""

import datetime

import plotly.graph_objects as go
import streamlit as st

from src.config import SENTIMENT_COLORS
from src.data import load_category_data


def _label_counts(data, column: str) -> dict:
    counts: dict = {}
    for label in data[column]:
        counts[label] = counts.get(label, 0) + 1
    return counts


def plot_major_updates(data) -> None:
    if data.empty:
        st.write("No major updates found after selected date.")
        return

    counts = _label_counts(data, "zero_shot_label")
    fig = go.Figure(go.Bar(
        x=list(counts.values()),
        y=list(counts.keys()),
        orientation="h",
        marker_color="lightsalmon",
        marker_line_color="darkred",
        marker_line_width=2,
    ))
    fig.update_layout(title="Major Updates Analysis", xaxis_title="Count", yaxis_title="Labels")
    st.write(fig)


def plot_bug_fixes(data) -> None:
    if data.empty:
        st.write("No bug fixes found after selected date.")
        return

    counts = _label_counts(data, "zero_shot_label")
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    fig = go.Figure(go.Bar(
        x=[count for _, count in sorted_counts],
        y=[label for label, _ in sorted_counts],
        orientation="h",
        marker_color="lightgreen",
        marker_line_color="darkgreen",
        marker_line_width=2,
    ))
    fig.update_layout(title="Bug Fixes Analysis", xaxis_title="Count", yaxis_title="Labels")
    st.write(fig)


def render_category_expander(label: str, csv_path: str, selected_date: datetime.date) -> None:
    """Render one "<label>" expander: a sentiment bar chart + data table.

    Replaces what used to be a separately hand-written ~35-line block per
    category (AI, Boss, Level, Graphics, Difficulty, Freeze, Appstore, Bug,
    Glitch, Crash).
    """
    with st.expander(label):
        data = load_category_data(csv_path)
        filtered = data[data["Date"] >= selected_date]
        sentiment_counts = filtered["Label"].value_counts()

        st.subheader("Emotion Classification Graphs")
        fig = go.Figure()
        for sentiment_label, count in sentiment_counts.items():
            fig.add_trace(go.Bar(
                x=[count],
                y=[sentiment_label.capitalize()],
                orientation="h",
                marker_color=SENTIMENT_COLORS.get(sentiment_label, "gray"),
                name=sentiment_label.capitalize(),
            ))
        fig.update_layout(
            title="Sentiment analysis",
            xaxis_title="Number",
            yaxis_title="Sentiment",
            width=650,
            height=400,
            margin=dict(l=150),
        )
        st.plotly_chart(fig)
        st.dataframe(filtered)
