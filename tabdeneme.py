 #### import libraries ####
import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import datetime
import pandas as pd
 ### setting page layout, title and icon ###
st.set_page_config(
    layout="centered",
    page_title="dert.solution",
    page_icon="🟡"
)


st.write("""
# dert.solutions

shown are the diversity of the comments in your app!

""")
 ### defination zero-shot ###
def oneshot_text(metin):
    oneshoter = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["Level", "Boss", "Difficulty", "AI", "Graphics"]
    oneshot_result = oneshoter(metin, candidate_labels)
    return oneshot_result
def text_cllasification(text):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


def main():
    """Reads data, creates customizable horizontal bar charts, and displays them in Streamlit tabs."""

    # Error handling
    try:
        df = pd.read_csv("etiketli_veri.csv")
    except FileNotFoundError:
        st.error("'etiketli_veri.csv' file not found. Please ensure it exists in the same directory as your script.")
        exit(1)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y").dt.date
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

    # Prepare data for charts (moved outside functions for efficiency)
    data_for_chart = {
        "Label": labels,
    }

    # Streamlit application with tabs
    tabs = st.tabs(["Main", "Major Updates", "Bug Fixes", 'Text Classificaton'])

    with tabs[0]:  # Main tab
        st.title("Etiketlenmiş Veri Analizi")
        fig = go.Figure(go.Bar(
            x=[1] * len(labels),  # Set a dummy x-value for all bars (hidden)
            y=data_for_chart["Label"],
            text=[f"Count: {labels.count(label)}" for label in labels],  # Add label count as text on each bar
            orientation='h',
            marker_color='skyblue',  # Set bar color (example)
            marker_line_color='darkblue',  # Set bar outline color (example)
            marker_line_width=2  # Set bar outline width (example)
        ))
        st.write(fig)

    with tabs[1]:  # Major Updates tab
        st.subheader("Major Updates Analysis")

        # Filter data based on selected date (avoiding global variables)
        start_year = datetime.date(2020, 1, 1)
        end_year = datetime.date(2024, 12, 31)
        selected_date = st.date_input(
            "Select Major Update Date:",
            min_value=start_year,
            max_value=end_year,
            value=start_year,
        )

        filtered_df_major = df[df["date"] >= selected_date]

        # Plot filtered data
        plot_major_updates(filtered_df_major)

    with tabs[2]:  # Bug Fixes tab
        st.subheader("Bug Fixes Analysis")

        # Filter data based on selected date (avoiding global variables)
        start_year = datetime.date(2020, 1, 1)
        end_year = datetime.date(2024, 12, 31)
        selected_date = st.date_input(
            "Select Bug Fix Date:",
            min_value=start_year,
            max_value=end_year,
            value=start_year,
        )

        filtered_df_bugfix = df[df["date"] >= selected_date]

        # Plot filtered data
        plot_bug_fixes(filtered_df_bugfix)

    with tabs[3]: # Text Cllasificaton
        # Then, we create a intro text for the app, which we wrap in a st.markdown() widget.

        st.write("")
        st.markdown(
            """

        Classify text instantly with this powerful application.

        """
        )

        st.write("")

        # Now, we create a form via `st.form` to collect the user inputs.

        # All widget values will be sent to Streamlit in batch.
        # It makes the app faster!

        with st.form(key="my_form"):

            ############ ST TAGS ############

            # The block of code below is to display some text samples to classify.
            # This can of course be replaced with your own text samples.

            # MAX_KEY_PHRASES is a variable that controls the number of phrases that can be pasted:
            # The default in this app is 50 phrases. This can be changed to any number you like.

            MAX_KEY_PHRASES = 500

            new_line = "\n"

            text = st.text_area(
                # Instructions
                "Enter your text to classify",
                # 'sample' variable that contains our keyphrases.

                # The height
                height=200,
                max_chars=MAX_KEY_PHRASES,
                # The tooltip displayed when the user hovers over the text area.
                help="Maximum 500 character"
            )

            # The block of code below:

            # 1. Converts the data st.text_area into a Python list.
            # 2. It also removes duplicates and empty lines.
            # 3. Raises an error if the user has entered more lines than in MAX_KEY_PHRASES.

            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                st.subheader("Classification:")
                result = oneshot_text(text)

                # Extract the scores for each label
                if isinstance(result, list):
                    scores = result[0]["scores"]
                    labels = result[0]["labels"]
                else:
                    scores = result["scores"]
                    labels = result["labels"]

                # Prepare data for Plotly chart
                data_for_chart = {
                    "Label": labels,
                    "Score": scores
                }

                # Create a Plotly bar chart
                fig = go.Figure(go.Bar(
                    x=data_for_chart["Score"],
                    y=data_for_chart["Label"],
                    orientation='h'
                ))

                # Customize layout and set chart size
                fig.update_layout(
                    title="Classification Results",
                    xaxis_title="Score",
                    yaxis_title="Label",
                    yaxis=dict(autorange="reversed"),
                    width=600,  # Set the width of the chart
                    height=400,  # Set the height of the chart
                    margin=dict(l=150),  # Adjust left margin for better alignment
                )

                # Center the chart horizontally
                st.write("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                st.plotly_chart(fig)
                st.write("</div>", unsafe_allow_html=True)

def plot_major_updates(data):
    if not data.empty:
        # Create a dictionary to store label counts
        label_counts = {}
        for label in data["zero_shot_label"]:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Prepare data for the chart
        data_for_chart = {
            "Label": list(label_counts.keys()),
            "Count": list(label_counts.values())
        }

        # Create the bar chart
        fig_major = go.Figure(go.Bar(
            x=data_for_chart["Count"],
            y=data_for_chart["Label"],
            orientation='h',
            marker_color = 'lightsalmon',
            marker_line_color ='darkred',
            marker_line_width = 2
        ))
        fig_major.update_layout(title="Major Updates Analysis",
                                xaxis_title="Count",
                                yaxis_title="Labels")
        st.write(fig_major)
    else:
        st.write("No major updates found after selected date.")

def plot_bug_fixes(data):
    if not data.empty:
        # Create a dictionary to store label counts (similar to plot_major_updates)
        label_counts = {}
        for label in data["zero_shot_label"]:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Sort label counts by value (descending order) (similar to plot_major_updates)
        sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

        # Prepare data for the chart (similar to plot_major_updates)
        data_for_chart = {
            "Label": [label for label, count in sorted_label_counts],
            "Count": [count for label, count in sorted_label_counts]
        }

        # Create the bar chart (similar to plot_major_updates)
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
