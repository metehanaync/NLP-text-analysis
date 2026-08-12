import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import softmax

from src.charts import plot_bug_fixes, plot_major_updates, render_category_expander
from src.config import BUG_FIX_CATEGORIES, BUG_FIX_LABELS, MAJOR_UPDATE_CATEGORIES, MAJOR_UPDATE_LABELS
from src.data import load_main_datasets
from src.models import get_sentiment_model_bundle, get_sentiment_pipeline, get_zero_shot_pipeline

st.set_page_config(layout="centered", page_title="dert.solution", page_icon="🟡")

st.write("""
# dert.solutions

shown are the diversity of the comments in your app!

""")

DATE_RANGE_START = datetime.date(2020, 1, 1)
DATE_RANGE_END = datetime.date(2024, 12, 31)


def render_main_tab(datasets: dict) -> None:
    st.title("Data Analysis")
    label_counts = datasets["main"]["zero_shot_label"].value_counts()
    fig_main = go.Figure(go.Bar(
        x=label_counts.index,
        y=label_counts.values,
        marker_color=["green", "red", "blue", "orange", "deepskyblue"],
    ))
    fig_main.update_layout(title="USER FEEDBACK", xaxis_title="Label", yaxis_title="Score")
    st.plotly_chart(fig_main)

    st.title("Happiness Rates")
    sentiment_counts = datasets["happiness"]["Label"].value_counts()
    fig_happiness = go.Figure(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        marker_color=["yellow", "fuchsia", "lightslategray"],
    ))
    fig_happiness.update_layout(title="HAPPINES RATES", xaxis_title="Sentyiments", yaxis_title="Score")
    st.plotly_chart(fig_happiness)

    st.title("Feedbacks  by  dert.solutions")
    st.markdown(
        " - As far as we observed, 50% of those who commented like your application."
        " Additionally, the person who likes the game is less likely to comment than the person who dislikes it."
        "This makes your application more **successful**"
    )
    st.write("")
    st.write("")

    x = ["May 2020", "May 2022"]
    fig_bugfix_trend = go.Figure(data=go.Scatter(x=x, y=[21.43, 75], mode="lines+markers", name="Freeze"))
    for line in [
        {"y": [89.87, 83.33], "name": "Appstore"},
        {"y": [32.68, 33.33], "name": "Bug"},
        {"y": [33.76, 25], "name": "Glitch"},
        {"y": [60.56, 69.44], "name": "Crash"},
    ]:
        fig_bugfix_trend.add_trace(go.Scatter(x=x, y=line["y"], mode="lines", name=line["name"]))
    fig_bugfix_trend.update_layout(
        title="Happiness rate", yaxis_title="%", xaxis_title="", yaxis_tickformat=",.2f"
    )
    st.plotly_chart(fig_bugfix_trend)
    st.markdown("- You seem to have solved the freezing problem after bug fix updates.")
    st.markdown(
        "- The happiness rates for Crash and Appstore are high, indicating satisfaction,"
        " but there still appears to be an issue with glitches and bugs persisting"
    )

    st.title("Recommendation")
    st.markdown(
        "- Unfixed bug and glitch problems reduce the game quality in the long run."
        " Additionally, an advantage can be gained by using bugs in the game."
        " This affects the dynamics of your game. We recommend that"
        " you make an update on bugs and glitches as soon as possible."
        " We recommend that you increase the update frequency."
    )

    st.title("Major Uptade Analysis")
    fig_major_trend = go.Figure(data=go.Scatter(x=x, y=[41.94, 50], mode="lines+markers", name="AI"))
    for line in [
        {"y": [53.79, 58.14], "name": "Level"},
        {"y": [69.68, 65.85], "name": "Graphics"},
        {"y": [43.42, 53.19], "name": "Difficult"},
        {"y": [35.71, 0], "name": "Boss"},
    ]:
        fig_major_trend.add_trace(go.Scatter(x=x, y=line["y"], mode="lines", name=line["name"]))
    fig_major_trend.update_layout(
        title="Happiness rate", yaxis_title="%", xaxis_title="", yaxis_tickformat=",.2f"
    )
    st.plotly_chart(fig_major_trend)
    st.markdown("- Your graphics team seems to be doing their job well.")
    st.markdown(
        "- In May 2022, all 4 comments made about **Boss** were negative, hence the happiness rate appears to be 0%."
    )

    st.title("Last Recommendation and Summary")
    st.markdown(
        "- We've spotted people who think the game has become boring because few updates have been released."
        " We recommend that you update more frequently."
    )
    st.markdown("- You can eliminate monotony by increasing the diversity of AI within the game")
    st.markdown(
        "- The vast majority of comments are about difficulty."
        " But you don't need to worry because you have adjusted the balance well."
    )


def render_major_updates_tab(df_major: pd.DataFrame) -> None:
    st.subheader("Major Updates Analysis")
    selected_date = st.date_input(
        "Select Major Update Date:",
        min_value=DATE_RANGE_START,
        max_value=DATE_RANGE_END,
        value=DATE_RANGE_START,
    )
    plot_major_updates(df_major[df_major["date"] >= selected_date])
    for label, csv_path in MAJOR_UPDATE_CATEGORIES:
        render_category_expander(label, csv_path, selected_date)


def render_bug_fixes_tab(df_bugfix: pd.DataFrame) -> None:
    st.subheader("Bug Fixes Analysis")
    selected_date = st.date_input(
        "Select Bug Fix Date:",
        min_value=DATE_RANGE_START,
        max_value=DATE_RANGE_END,
        value=DATE_RANGE_START,
    )
    plot_bug_fixes(df_bugfix[df_bugfix["date"] >= selected_date])
    for label, csv_path in BUG_FIX_CATEGORIES:
        render_category_expander(label, csv_path, selected_date)


def render_classification_tab() -> None:
    st.write("")
    st.markdown("Classify text instantly with this powerful application.")
    st.write("")

    MAX_KEY_PHRASES = 500

    with st.form(key="my_form"):
        text = st.text_area(
            "Enter your text to classify",
            height=200,
            max_chars=MAX_KEY_PHRASES,
            help="Maximum 500 character",
        )
        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            st.subheader("Classification:")

            zero_shot_result = get_zero_shot_pipeline()(text, MAJOR_UPDATE_LABELS)

            tokenizer, config, model = get_sentiment_model_bundle()
            encoded_input = tokenizer(text, return_tensors="pt")
            output = model(**encoded_input)
            scores = softmax(output[0][0].detach().numpy())
            ranking = np.argsort(scores)
            labels = [config.id2label[rank] for rank in ranking]
            probabilities = [scores[rank] for rank in ranking]

            colors = {"negative": "red", "neutral": "gray", "positive": "springgreen"}
            fig = go.Figure()
            for label, prob in zip(labels, probabilities):
                fig.add_trace(go.Bar(
                    x=[prob], y=[label], orientation="h", marker_color=colors[label], name=label.capitalize()
                ))
            fig.update_layout(
                title="Sentiment Analysis", xaxis_title="Probability", yaxis_title="Sentiment",
                width=650, height=400, margin=dict(l=150),
            )
            st.plotly_chart(fig)

            scores_zero = zero_shot_result["scores"]
            labels_zero = zero_shot_result["labels"]
            color_palette = px.colors.qualitative.Set1
            fig = go.Figure(go.Bar(
                x=scores_zero,
                y=labels_zero,
                orientation="h",
                marker_color=[color_palette[i % len(color_palette)] for i in range(len(labels_zero))],
            ))
            fig.update_layout(
                title="Classification Result", xaxis_title="Score", yaxis_title="Label",
                yaxis=dict(autorange="reversed"), width=600, height=400, margin=dict(l=150),
            )
            st.plotly_chart(fig)


def render_csv_upload_tab() -> None:
    st.title("CSV File Viewer and Text Classification")
    uploaded_file = st.file_uploader("Please select a CSV file", type="csv")

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)

        classifier_nlp = get_sentiment_pipeline()
        classifier_zero = get_zero_shot_pipeline()

        df["sentiment"] = df["text"].apply(lambda x: classifier_nlp(x)[0]["label"])

        sentiment_counts = df["sentiment"].value_counts()
        st.write("Classified Text Distribution:")
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Classified Text Distribution:")
        st.plotly_chart(fig)

        st.write("Classified CSV file:")
        st.dataframe(df)

        if not {"id", "date", "text"}.issubset(df.columns):
            st.error("The CSV file should contain headers for 'id', 'date', and 'text'..")
            return

        bugfix_labels = [classifier_zero(text, BUG_FIX_LABELS)["labels"][0] for text in df["text"]]
        bugfix_counts = pd.Series(bugfix_labels).value_counts()
        st.subheader("Bug Fix BarChart")
        fig1 = go.Figure(go.Bar(
            x=bugfix_counts.values, y=bugfix_counts.index, orientation="h",
            marker=dict(color="yellow", line=dict(color="orange", width=2)),
        ))
        fig1.update_layout(title="Graphics 1", xaxis_title="Score", yaxis_title="Labels")
        st.plotly_chart(fig1)

        major_labels = [classifier_zero(text, MAJOR_UPDATE_LABELS)["labels"][0] for text in df["text"]]
        major_counts = pd.Series(major_labels).value_counts()
        st.subheader("Major Uptade BarChart")
        fig2 = go.Figure(go.Bar(
            x=major_counts.values, y=major_counts.index, orientation="h",
            marker=dict(color="orange", line=dict(color="yellow", width=2)),
        ))
        fig2.update_layout(title="Graphics 2", xaxis_title="Score", yaxis_title="Label")
        st.plotly_chart(fig2)

    except Exception as e:
        st.error(f"Error: {e}")


def main() -> None:
    datasets = load_main_datasets()
    tabs = st.tabs(["Main", "Major Updates", "Bug Fixes", "Classificaton", "Load.csv"])

    with tabs[0]:
        render_main_tab(datasets)
    with tabs[1]:
        render_major_updates_tab(datasets["major"])
    with tabs[2]:
        render_bug_fixes_tab(datasets["bugfix"])
    with tabs[3]:
        render_classification_tab()
    with tabs[4]:
        render_csv_upload_tab()


if __name__ == "__main__":
    main()
