from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator  # Import Google Translator
import torch
import os
from scipy.special import softmax
import nltk
nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)

# Initialize sentiment analysis models
sia = SentimentIntensityAnalyzer()
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize translator
translator = Translator()

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        "roberta_neg": scores[0],
        "roberta_neu": scores[1],
        "roberta_pos": scores[2]
    }
    return scores_dict

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    review = request.form["review"]

    # Translate the review to English
    translated = translator.translate(review, src='auto', dest='en')
    translated_text = translated.text

    # Analyze sentiment
    vader_result = sia.polarity_scores(translated_text)
    roberta_result = polarity_scores_roberta(translated_text)

    result = {
        "vader": vader_result,
        "roberta": roberta_result
    }
    return render_template("result.html", text=review, translated_text=translated_text, result=result)

if __name__ == "__main__":
    app.run(debug=True)


try:
    translated = translator.translate(review, src='auto', dest='en')
    translated_text = translated.text
except Exception as e:
    translated_text = "Error in translation"
    print(f"Translation failed: {e}")
