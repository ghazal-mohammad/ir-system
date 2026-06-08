import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# make sure needed data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]


def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text, use_stemming=False):
    """
    Full preprocessing pipeline.
    Steps: lowercase -> remove punctuation -> tokenize -> stopwords -> lemmatize/stem
    use_stemming: if True uses stemming, otherwise uses lemmatization
    """
    if not text or not isinstance(text, str):
        return []

    text = to_lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)

    # remove short tokens (less than 2 chars)
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


def preprocess_to_string(text, use_stemming=False):
    """Returns preprocessed text as a single string (needed for TF-IDF vectorizer)."""
    tokens = preprocess(text, use_stemming)
    return ' '.join(tokens)
