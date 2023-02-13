import sys

# Model Imports
from src.models.TFIDF import *
from src.models.Word2Vector import *
from src.models.FastText import *


class TFIDF_runner:

	def __init__(self):

		print("Baseline tf-idf model:")
		class_lis = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']

		object_token = tfidf_Tokenization(class_lis)
		token = object_token.token_X() #data_path in the model file
		seeds = object_token.modify_seeds()

		object_tfidf = tfidf(token, seeds, class_lis)

		#get the tfidf model accuracy
		accuracy = object_tfidf.get_accuracy()
		
		# self.dataset = dataset
		# object_token = tfidf_Tokenization('./data/test/testdata/{}/{}'.format(dataset, cla))
		# token = object_token.token_X()
		# seeds = object_token.modify_seeds()
		# self.object_tfidf = tfidf('./data/test/testdata/{}/{}'.format(dataset, cla), token, seeds)

		print ("tfidf model: micro and macro f1 scores on all annotated data for are " + str(accuracy)) 
		# print ("tfidf Done")


class W2V_Runner:

	def __init__(self):

		print("Baseline Word2Vector model:")
		class_lis = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']
		
		#class_lis, vector_size, window, min_count, workers
		object_token = w2v_Tokenization(class_lis, 200, 5, 1, 4)

		token, word_dic = object_token.get_token_wordDic()
		seeds = object_token.modify_seeds()

		object_w2v = Word2vector(token, word_dic, seeds, class_lis)

		#get the w2v model accuracy
		accuracy = object_w2v.get_accuracy()

		print ("word2vec model: micro and macro f1 scores on all annotated data for are " + str(accuracy))



def main(targets):

	TFIDF_runner()
	W2V_Runner()


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)


# def main(targets):
# 	if 'data' in targets:
# 		print("Running on full data!")
# 		data = ...
# 	elif 'test' in targets:
# 		print("Running on test data!")
# 		data = ...
# 	ft = FastText("test")
