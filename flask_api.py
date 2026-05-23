from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLTK
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
NEGATIONS = {'no','not','nor','neither','never','none',"n't"}

app = Flask(__name__)

# Load model once
model = load_model("lstm_sentiment_model.keras")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

with open("model_config.pkl","rb") as f:
    config = pickle.load(f)


def clean_text(text):
    text=str(text).lower()
    text=re.sub(r'http\S+|www\S+|https\S+','',text)
    text=re.sub(r'@\w+','',text)
    text=re.sub(r'#','',text)
    text=re.sub(r'[^a-z\s]','',text)

    filtered=[
        w for w in text.split()
        if w not in STOP_WORDS or w in NEGATIONS
    ]

    return " ".join(filtered)


def predict_sentiment(text):

    cleaned=clean_text(text)

    seq=tokenizer.texts_to_sequences([cleaned])

    padded=pad_sequences(
        seq,
        maxlen=config["MAX_LEN"],
        padding='post'
    )

    prob=float(model.predict(padded,verbose=0)[0][0])

    label="Positive" if prob>=0.5 else "Negative"

    return {
        "input":text,
        "cleaned":cleaned,
        "prediction":label,
        "confidence":round(max(prob,1-prob)*100,2)
    }


@app.route("/predict",methods=["POST"])

def predict():

    data=request.get_json()

    text=data["text"]

    result=predict_sentiment(text)

    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True)