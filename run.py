import sys

# Model Imports
from src.models.TFIDF import *
from src.models.Word2Vector import *
from src.models.FastText import *


def main(targets):
	if 'data' in targets:
		print("Running on full data!")
	elif 'test' in targets:
		print("Running on test data!")

if __name__ == '__main__':
	targets = sys.argv[1:]
	main(targets)