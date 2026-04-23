# A Computationally Efficient Neural Network for Sarcasm Detection in News Headlines

A deep learning model that classifies news headlines as **sarcastic** or **genuine**, deployed as a REST API using FastAPI.

---

## About the Project

Sarcasm is one of the hardest things for machines to understand. A headline like *"Scientists confirm that doing nothing is more productive than going to the gym"* reads as positive on the surface but means the opposite.

This project builds a neural network that learns to spot those patterns from a dataset of ~26,000 labeled headlines — sarcastic ones from *The Onion* and genuine ones from *HuffPost*.

The trained model is served via a FastAPI backend with Swagger UI for easy testing.

---

## Model Architecture

```
Input (news headline)
        ↓
Text Preprocessing     — lowercase, remove punctuation, stopword removal
        ↓
Tokenization + Padding — vocab size: 10,000 | max length: 60
        ↓
Embedding Layer        — dim: 200
        ↓
Global Max Pooling     — extracts the strongest signal per feature
        ↓
Dense (128)  + Dropout (0.5)
Dense (64)   + Dropout (0.2)
Dense (32)
        ↓
Dense (1)  — Sigmoid → probability between 0 and 1
        ↓
Output: Sarcastic (>0.5) or Not Sarcastic (≤0.5)
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~98.5% |
| Precision (Sarcastic) | 0.99 |
| Recall (Sarcastic) | 0.98 |
| F1 Score | 0.99 |
| Training Epochs | 5 |

---

## Tech Stack

- **Python 3.12**
- **TensorFlow / Keras** — model training
- **FastAPI** — API framework
- **Uvicorn** — ASGI server
- **Render** — deployment

---

## Project Structure

```
sarcasm-detector/
├── api/
│   ├── main.py                  ← FastAPI app
│   ├── sarcasm_model.h5         ← trained Keras model
│   └── tokenizer.pkl            ← fitted tokenizer
│
├── notebook/
│   └── Sarcasm_Detection_in_News_Headlines.ipynb  ← training notebook
│
├── .gitignore 
├── README.md 
├── requirements.txt             ← dependencies
└── render.yaml                  ← Render deployment config
```

---

## API Endpoints

### `POST /predict`

Classifies a headline as sarcastic or genuine.

**Request**
```json
{
  "headline": "Scientists confirm doing nothing is more productive than going to the gym"
}
```

**Response**
```json
{
  "headline": "Scientists confirm doing nothing is more productive than going to the gym",
  "prediction": "Sarcastic",
  "confidence": 96.3
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/sarcasm-detector.git
cd sarcasm-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the server**
```bash
cd api
uvicorn api.main:app --reload
```

**4. Open Swagger UI**
```
http://localhost:8000/docs
```

---

## Dataset

**News Headlines Dataset for Sarcasm Detection**
- ~26,700 headlines total
- Sarcastic headlines sourced from [The Onion](https://www.theonion.com)
- Genuine headlines sourced from [HuffPost](https://www.huffpost.com)
- Each entry has: `headline`, `is_sarcastic` (0 or 1), `article_link`

Original dataset by Rishabh Misra — [Download from Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

---

## Live Demo

Deployed on Render: `your-render-url-here`

Swagger UI: `your-render-url-here/docs`

---

## Future Work

- Test on Twitter / Reddit sarcasm datasets to check generalization
- Compare performance with BiLSTM and BERT
- Add a simple frontend UI
