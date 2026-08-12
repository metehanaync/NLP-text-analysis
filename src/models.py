"""Cached loaders for the Hugging Face models/pipelines used across the app.

Streamlit reruns the whole script on every interaction, so without
`st.cache_resource` these (multi-hundred-MB) models would be re-downloaded
and re-loaded into memory on every click.
"""

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
import streamlit as st

from src.config import SENTIMENT_MODEL, ZERO_SHOT_MODEL


@st.cache_resource
def get_zero_shot_pipeline():
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)


@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


@st.cache_resource
def get_sentiment_model_bundle():
    """Tokenizer/model/config bundle for the manual softmax scoring path."""
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    config = AutoConfig.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    return tokenizer, config, model
