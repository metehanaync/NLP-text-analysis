# NLP Text Analysis — dert.solutions

A Streamlit dashboard that turns raw game-review text into actionable product feedback. It runs sentiment analysis and zero-shot topic classification over a set of user comments and visualizes the results (happiness rates, bug/glitch trends, major-update reception) as interactive Plotly charts.

## Features

- **Main** — high-level breakdown of feedback by topic and sentiment, plus written recommendations derived from the data.
- **Major Updates** — sentiment over time for feature-style topics (AI, Boss, Level, Graphics, Difficulty), filterable by date.
- **Bug Fixes** — sentiment over time for stability-related topics (Freeze, Appstore, Bug, Glitch, Crash), filterable by date.
- **Classification** — paste any text and get live sentiment scoring plus zero-shot topic classification.
- **Load.csv** — upload your own CSV (`id`, `date`, `text` columns) and get the same classification pipeline applied to it.

## Tech stack

- [Streamlit](https://streamlit.io/) for the web UI
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment, `facebook/bart-large-mnli` for zero-shot classification
- [Plotly](https://plotly.com/python/) for charts
- pandas / numpy / scipy for data handling

## Project structure

```
.
├── app.py              # Streamlit entry point — page layout and tabs
├── src/
│   ├── config.py        # dataset paths, category groupings, model names
│   ├── data.py           # cached CSV loading + date parsing
│   ├── models.py          # cached Hugging Face model/pipeline loaders
│   └── charts.py           # chart-building helpers
├── datalar/              # source CSV datasets
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/metehanaync/NLP-text-analysis.git
cd NLP-text-analysis
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open it in your browser.

The first run downloads the Hugging Face models (a few hundred MB); subsequent runs are cached by Streamlit's `@st.cache_resource` and load instantly.

## License

MIT — see [LICENSE](LICENSE).
