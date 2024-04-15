import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the pre-trained model and tokenizer
model_path = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True, from_tf=True)