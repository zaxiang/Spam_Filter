#TFIDF

import json
import pandas as pd
import numpy
import pickle
import string
from nltk.stem.porter import *
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from numpy.linalg import norm

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

class tfidf_Tokenization:

	def __init__(self, class_list):

		f = open("seedwords.json")
		self.seeds_dic = json.load(f)


		# class_list = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']
		lis = []
		
		for cla in class_list:
			all_files = os.listdir("annotated/" + cla)
			for fil in all_files:
				if fil.endswith(".txt"):
					file_path = "annotated/" + cla + "/" + fil
					with open(file_path, 'rb') as f:
						lis.append(f.read())
						

		self.X_train = lis


	def tokenization(self, token_doc):

		tfidf_vectorizer = TfidfVectorizer()
		tokenizer = tfidf_vectorizer.build_tokenizer()
	    
		punct = string.punctuation
		stemmer = PorterStemmer()

		english_stops = set(stopwords.words('english'))

		X_token = []
		for doc in token_doc:
			doc = str(doc.lower())
			doc = [i for i in doc if not (i in punct)] # non-punct characters
			doc = ''.join(doc) # convert back to string
			words = tokenizer(doc) # tokenizes
			words = [w for w in words if w not in english_stops] #remove stop words
			words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words] #stemmer and lemmatizer

			X_token.append(words)

		return X_token


	def token_X(self):

		print('tokenizing documents...')
		return self.tokenization(self.X_train)


	def modify_seeds(self):

		for clas in self.seeds_dic:
			cla_lis = self.seeds_dic[clas]

			token_input = [' '.join(cla_lis)]
			token_seed = self.tokenization(token_input)

			new_lis = token_seed[0]
			self.seeds_dic[clas] = new_lis

		return self.seeds_dic


class tfidf:

	def __init__(self, token, seeds_dic, class_list):

		#class_list = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']
		lis = []
		label = []

		for cla in class_list:
			all_files = os.listdir("annotated/" + cla)
			for fil in all_files:
				if fil.endswith(".txt"):
					file_path = "annotated/" + cla + "/" + fil
					with open(file_path, 'rb') as f:
						lis.append(f.read())
						label.append(cla)

		self.X_train = lis

		self.token = token
		self.seeds_dic = seeds_dic
		self.label = label


	def get_idf(self):

		print('get_idf')

		dic_idf = defaultdict(int)
		for doc in self.token:
			unique_token = set(doc)
			for w in unique_token:
				dic_idf[w] += 1

		self.idf_dic = dic_idf
		return dic_idf

	def get_tfidf_stat(self, doc, seeds): #the passed in doc is already tokenized

		sum_tfidf = 0
    
		dic_tfidf = defaultdict(int)
    
		for w in doc:
			dic_tfidf[w] += 1 #get the tf
        
		counter = 0
		for s in seeds:
			if s in self.idf_dic:
				counter += 1
				dic_tfidf[s] = dic_tfidf[s] * numpy.log((len(self.X_train) / self.idf_dic[s]))
			else:
				print("no way... the seed {} is not in any tokenized document".format(s))
				dic_tfidf[s] = 0
			sum_tfidf += dic_tfidf[s] 
        
		return sum_tfidf / counter

	def get_class(self, doc):

		dic_scores = defaultdict(int)

		for c in list(self.seeds_dic.keys()):
			tfidf = self.get_tfidf_stat(doc, self.seeds_dic[c])
			dic_scores[c] = tfidf

		# print(dic_scores) #checking 

		# if len(set(list(dic_scores.values()))) == 1: 
		# 	if int(list(set(list(dic_scores.values())))[0]) == 0: #having tie: all 0 case
		# 		return "no_label"
		# 		print("we have no class for this document")
		# 	else: #having tie: 
		# 		return list(dic_scores.keys())[0]
		# 		print("the tfidf are all the same for this document, weird...")

		return max(dic_scores, key=dic_scores.get)


	def get_prediction(self):

		prediction = []
		print("getting predictions")
		for doc in self.token:
			prediction.append(self.get_class(doc))

		return prediction

	def get_accuracy(self):

		self.get_idf() #get idf attribute only once

		prediction = self.get_prediction()
    
		print('calculating accuracy')
		#print('Accuracy: ', accuracy_score(news_df.label.tolist(), news_prediction, normalize=False))
		micro = f1_score(self.label, prediction, average='micro')
		macro = f1_score(self.label, prediction, average='macro')
		# print('F1-score micro: ', micro)
		# print('F1-score macro: ', macro)

		return micro, macro
    
