# Hate Speech Classification 🛡️🗣️

This project focuses on detecting hate speech from text using both traditional Machine Learning (ML) and Deep Learning (DL) techniques. The goal is to build and evaluate models that can accurately classify and flag hateful content.

---

## 📁 Project Structure

```yaml
HATE-SPEECH-CLASSIFICATION:
  Data:
    - Hate Speech.tsv
    - Hate Speech augmented.tsv
    - preprocessed_data.csv
  env: {}
  models: {}
  Notebook:
    - Lab1.ipynb  # ML approaches
    - Lab2.ipynb  # DL approaches
  production_model:
    - finalized_model.pkl
  utils:
    - preprocessing.py
    - vectorization.py
  .gitignore: null
  4.2.0: null
  app.py: null
  README.md: null
  requirements.txt: null


---

## 📊 Dataset

- **Original Data**: `Hate Speech.tsv`
- **Augmented Data**: `Hate Speech augmented.tsv`
- **Processed Data**: `preprocessed_data.csv`

The dataset contains text labeled as hate speech or not. Augmentation techniques were applied to improve generalization and model robustness.

---

## 🔍 Approaches

### 🧠 Machine Learning (Lab1.ipynb)

- Preprocessing: TF-IDF Vectorization
- Models Trained:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Random Forest
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

### 🤖 Deep Learning (Lab2.ipynb)

- Preprocessing: Tokenization & Padding
- Models Trained:
  - LSTM
  - BiLSTM
  - GRU
- Libraries used: TensorFlow, Keras
- Evaluation via validation loss, accuracy, confusion matrix

---

## ⚙️ Utility Scripts

- `preprocessing.py`: Handles data cleaning and text normalization (lowercasing, stopword removal, stemming, etc.)
- `vectorization.py`: Converts text to numerical features (TF-IDF for ML or token sequences for DL)

---

## 🚀 Streamlit App

An interactive web app is built with [Streamlit](https://streamlit.io/), allowing users to input text and receive real-time predictions.

To launch the app locally:

```bash
streamlit run app.py


# deployment Link 
https://hate-speech-classification-aliaymann.streamlit.app/