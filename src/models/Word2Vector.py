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


class w2v_Tokenization:

	def __init__(self, class_list, vector_size, window, min_count, workers):


		def cleaning(sentence):
		    stop_words = set(stopwords.words('english'))
		    tokens = re.sub(r'[^\w\s]', '', sentence.lower()).replace("\n", " ").split(" ")
		    cleaned = [token for token in tokens if token not in stop_words]
		    return " ".join(cleaned)

		labels = ["insurance-etc","investment", "medical-sales", "phising", "sexual", "software-sales"]
		text = []
		classes = []

		for cla in labels:
		    all_files = os.listdir("data/raw/spam/Annotated/" + cla)
		    for fil in all_files:
		        if fil.endswith(".txt"):
		            file_path = "data/raw/spam/Annotated/" + cla + "/" + fil
		            with open(file_path, 'r', encoding='ISO-8859-1') as f:
		                text.append(f.read())
		                classes.append(cla)
		                        
		self.data = pd.DataFrame({'sentence':text, 'label':classes})
		self.vector_size = vector_size
		self.window = window
		self.min_count = min_count
		self.workers = workers


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
			words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words] #lemmatizer and stemmer
			#Note: remove stemmer for fine dataset

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


	def get_token_wordDic(self):
     
		#preparing model and word dic 

		token_list = self.token_X() #2D array with word in each document
		token_documents = token_list.copy()

		#uncomment this for fine dataset
		
		# new_seeds = self.modify_seeds()
		# for lis in list(new_seeds.values()): #add the seed words to the word dic as well
		# 	token_list.append(lis)
    
		model = Word2Vec(sentences=token_list, 
		vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers) #use vector size 500
		
		model.save("word2vec.model")
    
		print("getting the vector for each word...")
		#save word vector in a dictionary
		word_dic = dict({})
		for idx, key in enumerate(model.wv.key_to_index):
			word_dic[key] = model.wv[key]
        
		return token_documents, word_dic


class Word2vector:

	def __init__(self, token, word_dic, seeds_dic, class_list):


		lis = []
		label = []

		for cla in class_list:
			all_files = os.listdir("data/raw/spam/Annotated/" + cla)
			for fil in all_files:
				if fil.endswith(".txt"):
					file_path = "data/raw/spam/Annotated/" + cla + "/" + fil
					with open(file_path, 'rb') as f:
						lis.append(f.read())
						label.append(cla)

		self.X_train = lis

		self.token = token
		self.word_dic = word_dic
		self.seeds_dic = seeds_dic
		self.label = label


	def get_all_document_vector(self):
    
		#starting vectorizing for each document
		print("getting the vector for each document...")

		DocVec = []
		for doc in self.token: #for each document
			doc_vec = numpy.array(100)
			counter = 0
			for word in doc: #for each word
				if word in self.word_dic.keys():
					counter += 1
					doc_vec = numpy.add(doc_vec, self.word_dic[word]) #add word vectors
			doc_vec = numpy.divide(doc_vec, counter) #divide by #of words in document (w/ vector)
			DocVec.append(doc_vec)
        
		return DocVec #return the vector representation of the all the document in a dict

	def get_seed_vector(self, seed_class):
    
		doc = self.seeds_dic[seed_class]
		SeedVec = numpy.array(100)
		num_seed = 0
		for word in doc: #for each seed word
			if word in self.word_dic.keys(): #should have all the seed words in the word dict
				num_seed += 1
				SeedVec = numpy.add(SeedVec, self.word_dic[word]) #add seed vectors
			# else:
			# 	print("no way.... I thought I added the seed words")

		if num_seed != 0:    #we should have a complete vector for each class
			SeedVec = numpy.divide(SeedVec, num_seed) #divide by #of seed words
    
		return SeedVec

	def get_cosin(self, doc_vector, class_vector):
		return numpy.dot(doc_vector,class_vector)/(norm(doc_vector)*norm(class_vector))

	def get_prediction(self):

		prediction = []
    
		All_DocVec = self.get_all_document_vector()#get all document vector
    
		print("getting the prediction...")
    
		for doc_vector in All_DocVec: #for each document, find the class with greatest similarity
			max_class = "none"
			max_similarity = -1000
			for cla in list(self.seeds_dic.keys()): 
				seed_vector = self.get_seed_vector(cla)
				similarity = self.get_cosin(doc_vector, seed_vector)
				if similarity >= max_similarity:
					max_similarity = similarity
					max_class = cla
			prediction.append(max_class) #append the prediction 
    
		return prediction

	def get_accuracy(self):
		prediction = self.get_prediction()
    
		print('calculating accuracy')
		#print('Accuracy: ', accuracy_score(news_df.label.tolist(), news_prediction, normalize=False))
		micro = f1_score(self.label, prediction, average='micro')
		macro = f1_score(self.label, prediction, average='macro')
		# print('F1-score micro: ', micro)
		# print('F1-score macro: ', macro)

		return micro, macro
        
