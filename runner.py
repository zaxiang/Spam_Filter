#run tfidf and word2vector file
from src.models.TFIDF import *
from src.models.Word2Vector import *


class TFIDF_runner:

	def __init__(self, dataset, cla):

		self.dataset = dataset

		object_token = tfidf_Tokenization('./data/raw/{}/{}'.format(dataset, cla))

		token = object_token.token_X()
		seeds = object_token.modify_seeds()

		self.object_tfidf = tfidf('./data/raw/{}/{}'.format(dataset, cla), token, seeds)

		print ("micro and macro f1 scores for " + self.dataset + " are " + str(self.object_tfidf.get_accuracy())) 

class W2V_Runner:

	def __init__(self, dataset, cla):

		self.dataset = dataset

		object_token = w2v_Tokenization('./raw/data/{}/{}'.format(dataset, cla), 100, 5, 1, 4)

		token, word_dic = object_token.get_token_wordDic()
		seeds = object_token.modify_seeds()

		self.object_w2v = Word2vector('./raw/data/{}/{}'.format(dataset, cla), token, word_dic, seeds)

		print ( "micro and macro f1 scores for " + self.dataset + " are " + str(self.object_w2v.get_accuracy()))
