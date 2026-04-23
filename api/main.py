import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = FastAPI(
    title="A Computationally Efficient Neural Network for Sarcasm Detection in News Headlines",
    description="Detects whether a news headline is sarcastic or genuine using a deep learning model.",
    version="1.0.0"
)

# load model and tokenizer once at startup
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)
print("Files in BASE_DIR:", os.listdir(BASE_DIR))
model = load_model(os.path.join(BASE_DIR, 'sarcasm_model.h5'))

with open(os.path.join(BASE_DIR, 'tokenizer.json'), 'r') as f:
    tokenizer = tokenizer_from_json(f.read())


MAX_LEN = 60

STOPWORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','this','that','these','those',
    'am','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','a','an','the','and','but','if','or','as','of',
    'at','by','for','with','to','from','in','out','on','off','into',
    'about','after','before','so','than','too','very','just','can',
    'will','not','no','nor','more','also','then','when','where','how'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(words)

def run_prediction(headline: str):
    cleaned = clean_text(headline)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob    = model.predict(padded, verbose=0)[0][0]
    label   = 'Sarcastic' if prob > 0.5 else 'Not Sarcastic'
    conf    = prob if prob > 0.5 else 1 - prob
    return label, round(float(conf) * 100, 1)


# ── request / response models ─────────────────────────────
class HeadlineInput(BaseModel):
    headline: str

    class Config:
        json_schema_extra = {
            "example": {
                "headline": "Scientists confirm doing nothing is more productive than going to the gym"
            }
        }

class PredictionOutput(BaseModel):
    headline:   str
    prediction: str
    confidence: float


# ── routes ────────────────────────────────────────────────
@app.get('/health', tags=["Health"])
def health():
    return {"status": "ok"}

@app.post('/predict', response_model=PredictionOutput, tags=["Prediction"])
def predict(data: HeadlineInput):
    label, conf = run_prediction(data.headline)
    return {
        "headline":   data.headline,
        "prediction": label,
        "confidence": conf
    }
