from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize

class UnifiedTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, method='count', embedding_model=None, embedding_dim=300,
                 min_df=1, max_df=1.0, ngram_range=(1, 1), min_count=1):
        self.method = method
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.min_count = min_count
        
        self.vectorizer = None

    def fit(self, X, y=None):
        if self.method in ['count', 'tfidf']:
            VectorizerClass = CountVectorizer if self.method == 'count' else TfidfVectorizer
            self.vectorizer = VectorizerClass(
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range
            )
            self.vectorizer.fit(X)
        return self

    def transform(self, X):
        if self.method in ['count', 'tfidf']:
            return self.vectorizer.transform(X)

        elif self.method in ['word2vec', 'glove', 'fasttext']:
            return np.array([self._embed_text(doc) for doc in X])

        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _embed_text(self, text):
        tokens = word_tokenize(text)
        if hasattr(self.embedding_model, 'get_word_vector'):
            vectors = [self.embedding_model.get_word_vector(word)
                       for word in tokens if word.strip() != '']
        else:
            vectors = [self.embedding_model[word]
                       for word in tokens if word in self.embedding_model]

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)
