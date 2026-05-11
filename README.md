# 📄 Research Paper Publishability Classifier

A machine learning pipeline that predicts whether a research paper is likely to be accepted at a top academic conference, based purely on **linguistic and readability features** extracted from the paper's text.

---

## 🧠 How It Works

The classifier reads the first ~5,000 characters of a paper's PDF (abstract + introduction), extracts 8 writing-quality features, and uses a trained Logistic Regression model to predict publishability.

**Features used:**

| Feature | Description |
|---|---|
| Gunning Fog Index | Vocabulary complexity (higher = harder) |
| Flesch-Kincaid Grade Level | Reading grade level |
| Coleman-Liau Index | Character-based readability (better for technical terms) |
| Average Sentence Length | Mean words per sentence |
| Sentence Length Variation | Std deviation of sentence lengths |
| Passive Voice Percentage | Ratio of passive-voice sentences |
| Lexical Density | Ratio of content words to total words |
| Perplexity (bigram) | Structural predictability of writing |

---

## 📦 Dataset

Papers are downloaded directly from public conference proceedings:

- **Publishable** (~100 papers each): NeurIPS 2023, CVPR 2024, EMNLP 2024, ACL 2024, ICCV 2023
- **Non-publishable** (~300 papers): arXiv papers filtered to exclude top-venue mentions, selected using queries targeting basic/survey/beginner-level work

---

## 🗂️ Pipeline Overview

```
Step 1  → Install dependencies
Step 2  → Download papers from NeurIPS, CVPR, EMNLP, ACL, ICCV, arXiv
Step 3  → Load libraries and NLP tools (spaCy, NLTK, scikit-learn)
Step 4  → Define feature extraction functions
Step 5  → Extract features from all PDFs → save to CSV
Step 6  → Prepare training data and binary labels
Step 7  → Scale features (MinMaxScaler) and train Logistic Regression
Step 8  → 5-fold cross-validation
Step 9  → Feature importance analysis
Step 10 → Save model and scaler as .pkl files
Step 11 → Predict on new/test papers
Step 12 → Visualise prediction probabilities
```

---

## 🚀 Getting Started

### Prerequisites

This notebook is designed to run on **Google Colab** (uses `/content/` paths).

### Install dependencies

```bash
pip install PyPDF2 pdfminer.six textblob scipy gensim networkx nltk scikit-learn spacy
python -m spacy download en_core_web_sm
python -m nltk.downloader all
```

### Run the notebook

Open `Main.ipynb` in Google Colab and run cells in order (Step 1 → Step 12).

> ⚠️ **Note:** ICCV 2025 and EMNLP 2025 may not be published yet. The notebook defaults to ICCV 2023 and EMNLP 2024 as safe fallbacks.

---

## 📁 Output Files

| File | Description |
|---|---|
| `new_training_features.csv` | Extracted features for all training papers |
| `trained_model_v2.pkl` | Trained Logistic Regression model |
| `scaler_v2.pkl` | Fitted MinMaxScaler |
| `predicted_publishability_v2.csv` | Predictions on test papers with probabilities |

---

## 📊 Model

- **Algorithm:** Logistic Regression (`C=4.28`, `solver=liblinear`, `class_weight=balanced`)
- **Evaluation:** 5-fold cross-validation (F1 score + accuracy)
- **Scaling:** MinMaxScaler fitted only on training data to prevent leakage

---

## 📚 Dependencies

- `PyPDF2` — PDF text extraction
- `spaCy` (`en_core_web_sm`) — NLP
- `NLTK` — tokenisation, CMU pronouncing dictionary, stopwords
- `scikit-learn` — model training and evaluation
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualisation
- `requests`, `beautifulsoup4` — paper scraping

---

## ⚠️ Limitations

- Only the first ~5,000 characters of each paper are analysed (abstract + introduction).
- The "non-publishable" label is a proxy — arXiv papers not rejected by any venue, just unlikely candidates based on content signals.
- The model captures **writing style**, not scientific novelty or technical correctness.
