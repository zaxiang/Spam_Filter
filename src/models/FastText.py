import fasttext
import numpy as np
import json
from sklearn.model_selection import train_test_split

from src.features.features import *
from src.data.generate_data import *

def FastText(data):
    train_set, test_set = train_test_split(data, test_size=0.4)
    generate_fasttext_train_data(train_set)
    model = fasttext.train_unsupervised(input='spam-train.txt', epoch=600, lr=0.05, wordNgrams=4, loss='hs', dim=40)
    vec_per_label = get_vectors_per_label("", model)
    features = get_features(data)
    vec_per_doc = get_vector_per_doc(features)
    labels = data['labels']
    pred = predict(vec_per_doc, vec_per_label)
    return labels, pred

def get_vectors_per_label(filename, model):
    f = open(filename)
    seeds = json.load(f)
    vector_per_label = []
    for key, value in seeds.items():
        lst = []
        for w in value:
            lst.append(model.get_word_vector(w))
        arr = np.asarray(lst)
        total = np.average(arr, axis=0)
        vector_per_label.append(total)
    return vector_per_label

def get_vector_per_doc(feature):
    vector_per_doc = []
    for feat in feature:
        lst = []
        for w in feat:
            lst.append(model.get_word_vector(w))
        arr = np.asarray(lst)
        total = np.average(arr, axis=0)
        vector_per_doc.append(total)
    return vector_per_doc

def predict(vector_per_doc, vector_per_label):
    predictions = []
    labels = list(seeds.keys())
    for doc in vector_per_doc:
        cosine = []
        for label in vector_per_label:
            cosine.append(np.dot(doc,label)/(norm(doc)*norm(label)))
        max_value = max(cosine)
        max_index = cosine.index(max_value)
        predictions.append(labels[max_index])
    return predictions   