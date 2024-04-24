 #### import libraries ####
import streamlit as st
from transformers import pipeline , AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import datetime
import pandas as pd
import numpy as np
from scipy.special import softmax
import plotly.graph_objects as go
import plotly.express as px


 ### setting page layout, title and icon ###
st.set_page_config(
    layout="centered",
    page_title="dert.solution",
    page_icon="🟡"
)

### title on the site ###
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


def preprocess(text): # for classification tabs
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
## For classification tabs
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def main():
    """Reads data, creates customizable horizontal bar charts, and displays them in Streamlit tabs."""

    # Error handling
    try:  # importing datas
        df_bugfix = pd.read_csv("datalar/bugetiketli_veri.csv")
        df_major = pd.read_csv("datalar/etiketli_veri.csv")
        df_happines = pd.read_csv("datalar/siniflandirilmis_data.csv")

    except FileNotFoundError:
        st.error("One or both of the files not found. Please ensure they exist in the same directory as your script.")
        exit(1)


            #Converting dates to a suitable format
    df_bugfix["date"] = pd.to_datetime(df_bugfix["date"], format="%m/%d/%Y").dt.date
    df_major["date"] = pd.to_datetime(df_major["date"], format="%m/%d/%Y").dt.date
    df_happines["Date"] = pd.to_datetime(df_happines["Date"], format='%m/%d/%Y').dt.date


    # # Streamlit application with tabs
    tabs = st.tabs(["Main", "Major Updates", "Bug Fixes", "Classificaton", "Load.csv"])

    with tabs[0]:  # Main tab
        st.title("Data Analysis")  # datas with 5 main class
        label_counts_formain = df_major['zero_shot_label'].value_counts() #calculate how many values there are
        figmajor = go.Figure(go.Bar(
            x=label_counts_formain.index,
            y=label_counts_formain.values,
            marker_color=['green', 'red', 'blue', 'orange', 'deepskyblue'],  #colors
        ))
        figmajor.update_layout(
            title="USER FEEDBACK",
            xaxis_title="Label",
            yaxis_title="Score",
        ) #Updating graphics
        st.plotly_chart(figmajor)

        st.title("Happiness Rates")  # Happines rates
        label_counts_forsentiments = df_happines['Label'].value_counts()  ## Main tab senytiments
        fighappines = go.Figure(go.Bar(
            x=label_counts_forsentiments.index,
            y=label_counts_forsentiments.values,
            marker_color=['yellow', 'fuchsia', 'lightslategray']
        ))
        fighappines.update_layout(
            title="HAPPINES RATES",
            xaxis_title="Sentyiments",
            yaxis_title='Score'
        )
        st.plotly_chart(fighappines)

        st.title("Feedbacks  by  dert.solutions")
        st.markdown(" - As far as we observed, 50% of those who commented like your application."
                    " Additionally, the person who likes the game is less likely to comment than the person who dislikes it."
                    "This makes your application more **successful**")
        st.write("")
        st.write("")
        x = ['May 2020', 'May 2022']
        y = [21.43, 75]

        # Create the line chart
        fig_for_analysis = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers', name='Freeze'))

        # Define additional line data and names
        additional_lines = [
            {'x': x, 'y': [89.87, 83.33], 'name': 'Appstore'},
            {'x': x, 'y': [32.68, 33.33], 'name': 'Bug'},
            {'x': x, 'y': [33.76, 25], 'name': 'Glitch'},
            {'x': x, 'y': [60.56, 69.44], 'name': 'Crash'}
        ]

        # Add each additional line to the chart
        for line_data in additional_lines:
            fig_for_analysis.add_trace(
                go.Scatter(x=line_data['x'], y=line_data['y'], mode='lines', name=line_data['name']))

        # Visualize the chart
        fig_for_analysis.update_layout(title='Happiness rate',
                                       yaxis_title='%',
                                       xaxis_title="",
                                       yaxis_tickformat=",.2f")  # Show 2 decimal places after comma

        # Show the chart in Streamlit
        st.plotly_chart(fig_for_analysis)
        st.markdown("- You seem to have solved the freezing problem after bug fix updates.")
        st.markdown("- The happiness rates for Crash and Appstore are high, indicating satisfaction,"
                    " but there still appears to be an issue with glitches and bugs persisting")
        st.title("Recommendation")
        st.markdown("- Unfixed bug and glitch problems reduce the game quality in the long run."
                    " Additionally, an advantage can be gained by using bugs in the game."
                    " This affects the dynamics of your game. We recommend that"
                    " you make an update on bugs and glitches as soon as possible."
                    " We recommend that you increase the update frequency.")
        st.title("Lets Another Analys")
        a = ['May 2020', 'May 2022']
        b = [41.94, 50]
        # Create the line chart
        fig_for_analysis2 = go.Figure(data=go.Scatter(x=a, y=b, mode='lines+markers', name='AI'))

        # Define additional line data and names
        additional_lines = [
            {'x': x, 'y': [53.79, 58.14], 'name': 'Level'},
            {'x': x, 'y': [69.68, 65.85], 'name': 'Graphics'},
            {'x': x, 'y': [43.42, 53.19], 'name': 'Difficult'}
        ]

        # Add each additional line to the chart
        for line_data in additional_lines:
            fig_for_analysis2.add_trace(
                go.Scatter(x=line_data['x'], y=line_data['y'], mode='lines', name=line_data['name']))

        # Visualize the chart
        fig_for_analysis2.update_layout(title='Happiness rate',
                                       yaxis_title='%',
                                       xaxis_title="",
                                       yaxis_tickformat=",.2f")  # Show 2 decimal places after comma

        # Show the chart in Streamlit
        st.plotly_chart(fig_for_analysis2)
        st.markdown("- Your graphics team seems to be doing their job well.")
        st.title("Last Recommendation and Summary")
        st.markdown("- We've spotted people who think the game has become boring because few updates have been released."
                    " We recommend that you update more frequently.")
        st.markdown("- You can eliminate monotony by increasing the diversity of AI within the game")
        st.markdown("- The vast majority of comments are about difficulty."
                    " But you don't need to worry because you have adjusted the balance well.")



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
        #Filter data by selected date
        filtered_df_major = df_major[df_major["date"] >= selected_date]

        # Plot filtered data
        plot_major_updates(filtered_df_major)
        with st.expander("AI"): # Aı box
            ai_veri = pd.read_csv("datalar/siniflandirilmis_ai.csv")
            ai_veri['Date'] = pd.to_datetime(ai_veri['Date'], dayfirst=True).dt.date


            # sort by date
            ai_veri = ai_veri.sort_values(by='Date')

            # Filter data by selected date
            filtered_data = ai_veri[ai_veri['Date'] >= selected_date]

            # count sentiments
            sentiment_counts = filtered_data['Label'].value_counts()

            colors_ai = {
                'positive': '#1f77b4',
                'negative': '#d62728',
                'neutral': '#ff7f0e'
            }

            # creating graphics
            st.subheader("Emotion Classification Graphs")
            fig = go.Figure()
            for label, count in sentiment_counts.items():
                fig.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
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

            # Showing datas
            st.dataframe(filtered_data)

        with st.expander("Boss"):  #boss box

            boss_veri = pd.read_csv("datalar/siniflandirilmis_boss.csv") #read boss data
            boss_veri['Date'] = pd.to_datetime(ai_veri['Date'], dayfirst=True).dt.date

            # sort by date
            boss_veri = boss_veri.sort_values(by='Date')

            # filter datas
            filtered_databoss = boss_veri[boss_veri['Date'] >= selected_date]

            # counting sentiments
            sentimentboss_counts = filtered_databoss['Label'].value_counts()



            # creating graphics
            st.subheader("Emotion Classification Graphs")
            figbos = go.Figure()
            for label, count in sentimentboss_counts.items():
                figbos.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figbos.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figbos)

            # showing datas
            st.dataframe(filtered_databoss)

        with st.expander("Level"): #level box
            level_veri = pd.read_csv("datalar/siniflandirilmis_level.csv")
            level_veri['Date'] = pd.to_datetime(level_veri['Date'], dayfirst=True).dt.date

            # sort by date
            level_veri = level_veri.sort_values(by='Date')

            # filtered datas
            filtered_datalevel = level_veri[level_veri['Date'] >= selected_date]

            # counting sentiments
            sentimentlevel_counts = filtered_datalevel['Label'].value_counts()

            # creating graphics
            st.subheader("Emotion Classification Graphs")
            figlevel = go.Figure()
            for label, count in sentimentlevel_counts.items():
                figlevel.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figlevel.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figlevel)

            # show datas
            st.dataframe(filtered_datalevel)
        with st.expander("Graphics"):
            graphics_veri = pd.read_csv("datalar/siniflandirilmis_graphics.csv")
            graphics_veri['Date'] = pd.to_datetime(graphics_veri['Date'], dayfirst=True).dt.date

            # sort by daate
            graphics_veri = graphics_veri.sort_values(by='Date')

            # filter datas
            filtered_datagraphics = graphics_veri[graphics_veri['Date'] >= selected_date]

            # counting
            sentimentgraphics_counts = filtered_datagraphics['Label'].value_counts()

            # creating graphics
            st.subheader("Emotion Classification Graphs")
            figgraphics = go.Figure()
            for label, count in sentimentgraphics_counts.items():
                figgraphics.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figgraphics.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figgraphics)

            # showing datas
            st.dataframe(filtered_datagraphics)
        with st.expander("Difficulty"): #difficulty box
            difficulty_veri = pd.read_csv("datalar/siniflandirilmis_difficulty.csv")
            difficulty_veri['Date'] = pd.to_datetime(difficulty_veri['Date'], dayfirst=True).dt.date

            # sorting by date
            difficulty_veri = difficulty_veri.sort_values(by='Date')

            # filter
            filtered_datadifficulty = difficulty_veri[difficulty_veri['Date'] >= selected_date]

            # counting
            sentimentdifficulty_counts = filtered_datadifficulty['Label'].value_counts()

            # creating graphics
            st.subheader("Emotion Classification Graphs")
            figdifficulty = go.Figure()
            for label, count in sentimentdifficulty_counts.items():
                figdifficulty.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figdifficulty.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figdifficulty)


            st.dataframe(filtered_datadifficulty)

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

        filtered_df_bugfix = df_bugfix[df_bugfix["date"] >= selected_date]

        # Plot filtered data
        plot_bug_fixes(filtered_df_bugfix)
        with st.expander("Freeze"):
            freeze_veri = pd.read_csv("datalar/siniflandirilmis_freeze.csv")
            freeze_veri['Date'] = pd.to_datetime(freeze_veri['Date'], dayfirst=True).dt.date

            # sorting by date
            freeze_veri = freeze_veri.sort_values(by='Date')

            # filter
            filtered_datafreeze = freeze_veri[freeze_veri['Date'] >= selected_date]

            # counting
            sentimentfreeze_counts = filtered_datafreeze['Label'].value_counts()

            # creating graphics
            st.subheader("Emotion Classification Graphs")
            figfreeze = go.Figure()
            for label, count in sentimentfreeze_counts.items():
                figfreeze.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figfreeze.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figfreeze)
            # showing datas
            st.dataframe(filtered_datafreeze)

        with st.expander("Appstore"): # appstore box
            appstore_veri = pd.read_csv("datalar/siniflandirilmis_appstore.csv")
            appstore_veri['Date'] = pd.to_datetime(appstore_veri['Date'], dayfirst=True).dt.date

            #sorting by date
            appstore_veri = appstore_veri.sort_values(by='Date')

            # filter
            filtered_dataappstore = appstore_veri[appstore_veri['Date'] >= selected_date]

            #counting
            sentimentappstore_counts = filtered_dataappstore['Label'].value_counts()

            # creating graphics
            st.subheader("Emotion Classification Graphs")
            figappstore = go.Figure()
            for label, count in sentimentappstore_counts.items():
                figappstore.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figappstore.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figappstore)

            # showing datas
            st.dataframe(filtered_dataappstore)
        with st.expander("Bug"):
            bug_veri = pd.read_csv("datalar/siniflandirilmis_bug.csv")
            bug_veri['Date'] = pd.to_datetime(bug_veri['Date'], dayfirst=True).dt.date

            # Sort by date
            bug_veri = bug_veri.sort_values(by='Date')

            # filtered
            filtered_databug = bug_veri[bug_veri['Date'] >= selected_date]

            # counting
            sentimentbug_counts = filtered_databug['Label'].value_counts()

            # Çcreating graphics
            st.subheader("Emotion Classification Graphs")
            figbug = go.Figure()
            for label, count in sentimentbug_counts.items():
                figbug.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figbug.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figbug)

            # Showing datas
            st.dataframe(filtered_databug)
        with st.expander("Glitch"): #glitch box
            glitch_veri = pd.read_csv("datalar/siniflandirilmis_glitch.csv")
            glitch_veri['Date'] = pd.to_datetime(glitch_veri['Date'], format='%m/%d/%Y').dt.date


            # sort by date
            glitch_veri = glitch_veri.sort_values(by='Date')

            # filtered by date
            filtered_dataglitch = glitch_veri[glitch_veri['Date'] >= selected_date]

            # counting
            sentimentglitch_counts = filtered_dataglitch['Label'].value_counts()

            # Çreating graphics
            st.subheader("Emotion Classification Graphs")
            figglitch = go.Figure()
            for label, count in sentimentglitch_counts.items():
                figglitch.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figglitch.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figglitch)

            # showing datas
            st.dataframe(filtered_dataglitch)

        with st.expander("Crash"): # crash box
            crash_veri = pd.read_csv("datalar/siniflandirilmis_crash.csv")
            crash_veri['Date'] = pd.to_datetime(crash_veri['Date'], format='%m/%d/%Y').dt.date


            # sort by date
            crash_veri = crash_veri.sort_values(by='Date')

            # filter by date
            filtered_datacrash = crash_veri[crash_veri['Date'] >= selected_date]

            # counting sentiments
            sentimentcrash_counts = filtered_datacrash['Label'].value_counts()

            # creating bar graphics
            st.subheader("Emotion Classification Graphs")
            figcrash = go.Figure()
            for label, count in sentimentcrash_counts.items():
                figcrash.add_trace(go.Bar(
                    x=[count],
                    y=[label.capitalize()],
                    orientation='h',
                    marker_color=colors_ai[label],
                    name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                ))
            figcrash.update_layout(
                title="Sentiment analysis",
                xaxis_title="Number",
                yaxis_title="Sentiment",
                width=650,
                height=400,
                margin=dict(l=150),
            )
            st.plotly_chart(figcrash)

            # showing datas
            st.dataframe(filtered_datacrash)

    with tabs[3]: # Text Cllasificaton tabs
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
            # The default in this app is 500 phrases. This can be changed to any number you like.

            MAX_KEY_PHRASES = 500

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

                text = preprocess(text)
                encoded_input = tokenizer(text, return_tensors='pt')
                output = model(**encoded_input)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)

                ranking = np.argsort(scores)
                ranking = ranking[::-1]

                labels = [config.id2label[rank] for rank in ranking]
                probabilities = [scores[rank] for rank in ranking]

                # Define colors for different sentiments
                colors = {'negative': 'red', 'neutral': 'gray', 'positive': 'springgreen'}

                # Create bar chart with custom colors
                fig = go.Figure()
                for label, prob in zip(labels, probabilities):
                    fig.add_trace(go.Bar(
                        x=[prob],
                        y=[label],
                        orientation='h',
                        marker_color=colors[label],
                        name=label.capitalize()  # Yazıyı sırala ve baş harfi büyük yap
                    ))

                fig.update_layout(
                    title="Sentiment Analysis",
                    xaxis_title="Probability",
                    yaxis_title="Sentiment",
                    width=650,
                    height=400,
                    margin=dict(l=150),
                )

                st.plotly_chart(fig)

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
                color_palette = px.colors.qualitative.Set1

                # creating bar chart
                fig = go.Figure(go.Bar(
                    x=data_for_chart["Score"],
                    y=data_for_chart["Label"],
                    orientation='h',
                    marker_color=[color_palette[i % len(color_palette)] for i in range(len(data_for_chart["Label"]))]
                    # different colors for every bar
                ))


                # Layout settings
                fig.update_layout(
                    title="Classification Result",
                    xaxis_title="Score",
                    yaxis_title="Label",
                    yaxis=dict(autorange="reversed"),
                    width=600,
                    height=400,
                    margin=dict(l=150),
                )

                # Center the chart horizontally
                st.write("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                st.plotly_chart(fig)
                st.write("</div>", unsafe_allow_html=True)

    with tabs[4]: #loading .csv file
        # Streamlit tittle
        st.title("CSV File Viewer and Text Classification")

        # load csv file
        uploaded_file = st.file_uploader("Please select a CSV file", type="csv")

        #uploading model for clasffication
        classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

        if uploaded_file is not None:
            try:
                # reading csv file
                df = pd.read_csv(uploaded_file)

                # text classification
                df['sentiment'] = df['text'].apply(lambda x: classifier(x)[0]['label'])

                # visulation
                sentiment_counts = df['sentiment'].value_counts()

                # creating graphics
                st.write("Classified Text Distribution:")
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                             title="Classified Text Distribution:")
                st.plotly_chart(fig)

                # showing the data
                st.write("Classified CSV file:")
                st.dataframe(df)

            except Exception as e:
                st.error(f"Hata: {e}")


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
