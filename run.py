import sys

# Model Imports
from src.models.TFIDF import *
from src.models.Word2Vector import *
from src.models.FastText import *


class TFIDF_runner:

	def __init__(self, targets):

		print("Baseline tf-idf model:")
		class_lis = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']

		object_token = tfidf_Tokenization(class_lis, targets)
		token = object_token.token_X() #data_path in the model file
		seeds = object_token.modify_seeds()

		object_tfidf = tfidf(token, seeds, class_lis, targets)

		#get the tfidf model accuracy
		accuracy = object_tfidf.get_accuracy()
		self.targets = targets
		
		# self.dataset = dataset
		# object_token = tfidf_Tokenization('./data/test/testdata/{}/{}'.format(dataset, cla))
		# token = object_token.token_X()
		# seeds = object_token.modify_seeds()
		# self.object_tfidf = tfidf('./data/test/testdata/{}/{}'.format(dataset, cla), token, seeds)

		print ("tfidf model: micro and macro f1 scores are " + str(accuracy)) 
		# print ("tfidf Done")


class W2V_Runner:

	def __init__(self, targets):

		print("Baseline Word2Vector model:")
		class_lis = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']
		
		#class_lis, vector_size, window, min_count, workers
		object_w2v = Word2vector(110, 5, 1, 8, targets)

		#get the w2v model accuracy
		accuracy = object_w2v.get_accuracy()
		self.targets = targets

		print ("word2vec model: micro and macro f1 scores are " + str(accuracy))
		


def main(targets):

	TFIDF_runner(targets)
	W2V_Runner(targets)


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
