from transformers import pipeline
import torch

sentiment_analyzer = pipeline("sentiment-analysis")

while True:
    texts = input("Enter your text: ")
    if texts.lower() == "exit":
        break

    results = sentiment_analyzer(texts)[0]
    print(f"-> Sentiment: {results['label']} (confidence: {results['score']:.2f})\n")