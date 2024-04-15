import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import datetime

def main():
    """Reads data, creates customizable horizontal bar charts, and displays them."""

    # Error handling
    try:
        df = pd.read_csv("etiketli_veri.csv")
    except FileNotFoundError:
        st.error("'etiketli_veri.csv' file not found. Please ensure it exists in the same directory as your script.")
        exit(1)

    # Check for columns
    if "zero_shot_label" not in df.columns or "zero_shot_score" not in df.columns:
        st.error("Columns 'zero_shot_label' and 'zero_shot_score' not found. Please ensure they exist in your data.")
        exit(1)

    labels = df["zero_shot_label"].tolist()
    scores = df["zero_shot_score"].tolist()

    # Validate data length
    if len(labels) != len(scores):
        st.error("Label and score lengths don't match. Please ensure they have the same number of entries.")
        exit(1)

    # Streamlit application
    st.title("Etiketlenmiş Veri Analizi")

    # Prepare data for chart
    data_for_chart = {
        "Label": labels
    }

    # Create Plotly bar chart with customization options
    fig = go.Figure(go.Bar(
        x=[1] * len(labels),  # Set a dummy x-value for all bars (hidden)
        y=data_for_chart["Label"],
        text=[f"Count: {labels.count(label)}" for label in labels],  # Add label count as text on each bar
        orientation='h',
        marker_color='skyblue',  # Set bar color (example)
        marker_line_color='darkblue',  # Set bar outline color (example)
        marker_line_width=2  # Set bar outline width (example)
    ))

    # Streamlit display with centering
    st.write(fig)

    # Filter data based on selected dates
    start_year = datetime.date(2020, 1, 1)
    end_year = datetime.date(2024, 12, 31)
    selected_date_formajor = st.date_input(
        "What is your major update date:",
        min_value=start_year,
        max_value=end_year,
        value=start_year,
    )
    selected_date_forbugfix = st.date_input(
        "What is your bug fix date:",
        min_value=start_year,
        max_value=end_year,
        value=start_year,
    )

    start_date_major = selected_date_formajor
    start_date_bugfix = selected_date_forbugfix
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y").dt.date  # Convert date column to datetime.date
    filtered_df_major = df[df["date"] >= start_date_major]
    filtered_df_bugfix = df[df["date"] >= start_date_bugfix]

    # Plot filtered data
    plot_major_updates(filtered_df_major)
    plot_bug_fixes(filtered_df_bugfix)

def plot_major_updates(data):
    st.subheader("Major Updates After Selected Date")
    if not data.empty:
        # Create a dictionary to store label counts
        label_counts = {}
        for label in data["zero_shot_label"]:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Sort label counts by value (descending order)
        sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

        # Prepare data for the chart
        data_for_chart = {
            "Label": [label for label, count in sorted_label_counts],
            "Count": [count for label, count in sorted_label_counts]
        }

        # Create the bar chart
        fig_major = go.Figure(go.Bar(
            x=data_for_chart["Count"],  # Use label counts on x-axis
            y=data_for_chart["Label"],
            orientation='h',
            marker_color='lightsalmon',
            marker_line_color='darkred',
            marker_line_width=2
        ))
        fig_major.update_layout(title="Major Updates Analysis",
                                xaxis_title="Count",  # Change x-axis title
                                yaxis_title="Labels")
        st.write(fig_major)
    else:
        st.write("No major updates found after selected date.")

def plot_bug_fixes(data):
    st.subheader("Bug Fixes After Selected Date")
    if not data.empty:
        # Create a dictionary to store label counts
        label_counts = {}
        for label in data["zero_shot_label"]:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Sort label counts by value (descending order)
        sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

        # Prepare data for the chart
        data_for_chart = {
            "Label": [label for label, count in sorted_label_counts],
            "Count": [count for label, count in sorted_label_counts]
        }

        # Create the bar chart
        fig_bugfix = go.Figure(go.Bar(
            x=data_for_chart["Count"],  # Use label counts on x-axis
            y=data_for_chart["Label"],
            orientation='h',
            marker_color='lightgreen',
            marker_line_color='darkgreen',
            marker_line_width=2
        ))
        fig_bugfix.update_layout(title="Bug Fixes Analysis",
                                 xaxis_title="Count",  # Change x-axis title
                                 yaxis_title="Labels")
        st.write(fig_bugfix)
    else:
        st.write("No bug fixes found after selected date.")

if __name__ == "__main__":
    main()
