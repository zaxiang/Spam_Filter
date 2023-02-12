import sys

# Model Imports
from src.models.TFIDF import *
from src.models.Word2Vector import *
# from src.models.FastText import *


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

		print ("micro and macro f1 scores on all annotated data for are " + str(accuracy)) 
		# print ("tfidf Done")


def main(targets):

	TFIDF_runner()

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
