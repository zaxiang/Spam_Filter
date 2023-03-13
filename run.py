import sys

# Model Imports
from src.models.TFIDF import *
from src.models.Word2Vector import *
from src.models.FastText import *
from src.models.ConWea import *


class TFIDF_runner:

	def __init__(self, target):

		print("Baseline TF-IDF model:")
		class_lis = ['insurance-etc', 'investment', 'medical-sales', 'phising', 'sexual', 'software-sales']

		object_token = tfidf_Tokenization(class_lis, target)
		token = object_token.token_X() #data_path in the model file
		seeds = object_token.modify_seeds()

		object_tfidf = tfidf(token, seeds, class_lis, target)

		#get the tfidf model accuracy
		micro, macro = object_tfidf.get_accuracy()
		print('Micro F1 = ' + str(micro))
		print('Macro F1 = ' + str(macro))
		print('\n\n')
		


class W2V_Runner:

	def __init__(self, target):

		print("Baseline Word2Vec model:")
		
		#class_lis, vector_size, window, min_count, workers
		object_w2v = Word2vector(110, 5, 1, 8, target)

		#get the w2v model accuracy
		micro, macro = object_w2v.get_accuracy()
		print('Micro F1 = ' + str(micro))
		print('Macro F1 = ' + str(macro))
		print('\n\n')


class FastText_Runner:

	def __init__(self, target):
		print("Baseline FastText word embeddings model:")
		pred, label = FastText(target)
		micro, macro = get_accuracy(pred, label)
		print('Micro F1 = ' + str(micro))
		print('Macro F1 = ' + str(macro))
		print('\n\n')

class ConWea_Runner:
	def __init__(self, targets):
		print("ConWea Model:")
		

def main(targets):
	if 'data' in targets:
		print("Running on full data!")
		target = 'data'
	elif 'test' in targets:
		print("Running on test data!")
		target = 'test'
	TFIDF_runner(target)
	W2V_Runner(target)
	FastText_Runner(target)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)


