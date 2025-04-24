from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import unicodedata
import nltk 
import re
import emoji
from sklearn.base import BaseEstimator, TransformerMixin


# Download necessary resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



class TextPreprocessor(BaseEstimator, TransformerMixin):
    # No fitting needed in preprocessing, just return self
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X['tweet'].apply(text_preprocessing)  


def text_preprocessing(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Normalize characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Lowercase
    text = text.lower()

    # Expand contractions
    contractions = {
        "can't": "can not", "won't": "will not", "n't": " not", "'re": " are",
        "'s": " is", "'d": " would", "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    for pattern, repl in contractions.items():
        text = re.sub(pattern, repl, text)

    # Remove URLs, usernames, hashtags, emojis, punctuation, numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'#', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove newlines/tabs, extra whitespaces
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Join back into string
    text = ' '.join(tokens)

    return text