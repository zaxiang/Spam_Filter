#Word2Vec

import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

from sklearn import metrics

import json

import pickle
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()
import re

from numpy.linalg import norm


# import json
# import pandas as pd
# import numpy
# import pickle
# import string
# from nltk.stem.porter import *
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from collections import defaultdict
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from numpy.linalg import norm
# # !python -m pip install -U gensim
# from gensim.test.utils import common_texts
# from gensim.models import Word2Vec
# import nltk.corpus
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer 
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# lemmatizer = WordNetLemmatizer()


class Word2vector:

	def __init__(self, vector_size, window, min_count, workers, targets):

		self.targets = targets

		def cleaning(sentence):
		    stop_words = set(stopwords.words('english'))
		    tokens = re.sub(r'[^\w\s]', '', sentence.lower()).replace("\n", " ").split(" ")
		    cleaned = [token for token in tokens if token not in stop_words]
		    return " ".join(cleaned)


		labels = ["insurance-etc","investment", "medical-sales", "phising", "sexual", "software-sales"]
		text = []
		classes = []

		for cla in labels:
			path = "data/raw/spam/Annotated/"
			if self.targets == "data":
				path = "data/raw/spam/Annotated/"
			elif self.targets == "test":
				path = "test/testdata/"

			all_files = os.listdir(path + cla)
			for fil in all_files:
				if fil.endswith(".txt"):

					file_path = path + cla + "/" + fil
					with open(file_path, 'r', encoding='ISO-8859-1') as f:
						text.append(cleaning(str(f.read())))
						classes.append(cla)

		            # file_path = "data/raw/spam/Annotated/" + cla + "/" + fil
					
		                        
		                
		self.data = pd.DataFrame({'sentence':text, 'label':classes})

		def preprocessing(sentence):
		    tokens = sentence.split(" ")
		    return [token for token in tokens if token!="" and token != " "]
		    
		self.features = self.data["sentence"].apply(preprocessing)

		self.vector_size = vector_size
		self.window = window
		self.min_count = min_count
		self.workers = workers

	def training(self):

		self.model = Word2Vec(sentences=self.features, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
		self.model.save("word2vec.model")
		self.model = Word2Vec.load("word2vec.model")
		print("training word2vec model with 800 epochs")
		self.model.train(self.features, total_examples=len(self.data), epochs=800)

	def get_vectors_per_label(self, filename):

	    f = open(filename)
	    seeds = json.load(f)
	    vector_per_label = []
	    for key, value in seeds.items():
	        lst = []
	        for w in value:
	            lst.append(self.model.wv[w])
	        arr = np.asarray(lst)
	        total = np.average(arr, axis=0)
	        vector_per_label.append(total)
	    return vector_per_label

	def get_vector_per_doc(self, feature):

	    vector_per_doc = []
	    for feat in feature:
	        lst = []
	        for w in feat:
	            lst.append(self.model.wv[w])
	        arr = np.asarray(lst)
	        total = np.average(arr, axis=0)
	        vector_per_doc.append(total)
	    return vector_per_doc

	def get_prediction(self):

		seed_path = "data/out/seedwords.json"
		f = open(seed_path)
		self.seeds_dic = json.load(f)

		self.training()
		vector_per_doc = self.get_vector_per_doc(self.features)
		vector_per_label = self.get_vectors_per_label(seed_path)

		def predict_word2vec(vector_per_doc, vector_per_label):
		    predictions = []
		    labels = list(self.seeds_dic.keys())
		    for doc in vector_per_doc:
		        cosine = []
		        for label in vector_per_label:
		            cosine.append(np.dot(doc,label)/(norm(doc)*norm(label)))
		        max_value = max(cosine)
		        max_index = cosine.index(max_value)
		        predictions.append(labels[max_index])
		    return predictions   

		prediction_word2vec = predict_word2vec(vector_per_doc, vector_per_label)
		self.data["prediction_word2vec"] = prediction_word2vec

	def get_accuracy(self):

		self.get_prediction()

		micro = metrics.f1_score(self.data["label"], self.data["prediction_word2vec"], average="micro")
		macro = metrics.f1_score(self.data["label"], self.data["prediction_word2vec"], average="macro")

		return micro, macro

