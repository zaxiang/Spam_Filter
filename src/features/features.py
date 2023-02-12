import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def get_features(data):
    features = data['sentence'].apply(preprocessing)
    return features

def preprocessing(sentence):
    tokens = sentence.split(" ")
    return [token for token in tokens if token!="" and token != " "]

def cleaning(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = re.sub(r'[^\w\s]', '', sentence.lower()).replace("\n", " ").split(" ")
    cleaned = [token for token in tokens if token not in stop_words]
    return " ".join(cleaned)