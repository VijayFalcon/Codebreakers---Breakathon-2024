from flask import Flask, request, render_template
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)

# Load pre-trained models
with open('vader_analyzer.pkl', 'rb') as f:
    sia = pickle.load(f)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['review']
    vader_result = sia.polarity_scores(text)
    roberta_result = polarity_scores_roberta(text)
    combined_result = {
        "vader": vader_result,
        "roberta": roberta_result
    }
    return render_template('results.html', text=text, result=combined_result)

if __name__ == '__main__':
    app.run(debug=True)
