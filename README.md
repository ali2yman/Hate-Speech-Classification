# Hate Speech Classification 🛡️🗣️

This project focuses on detecting hate speech from text using both traditional Machine Learning (ML) and Deep Learning (DL) techniques. The goal is to build and evaluate models that can accurately classify and flag hateful content.

---

## 📁 Project Structure

     ```
     ├── Data/                  # Folder for datasets (raw and preprocessed)
     ├── utils/
     │   ├── preprocessing.py    # Code for data cleaning and preprocessing
     │   ├── vectorization.py    # Code for vectorization and embedding   
     ├── notebooks/              # Jupyter notebooks for exploratory data analysis
     │   ├── Lab1.ipynb          # Applying ML models 
     │   ├── Lab2.ipynb          # Applying DL models
     ├── production_model/
         ├── finalized_model.pkl # this  is model for the pipline 
     ├── app.py                  # U can Run the app from here (streamlit)               
     ├── requirements.txt        # List of required Python libraries
     ├── README.md               # Project documentation
     ```

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