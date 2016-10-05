import tensorflow as tf 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random, pickle
import numpy as np 
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg):
	lexicon = []

	for file in [pos, neg]:
		with open(file, 'r') as f:
			contents = f.readlines()
			for line in contents[:hm_lines]:
				words = word_tokenize(line.lower())
				lexicon += list(words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)

	meaningful_words =[]
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			meaningful_words.append(w)

	print(len(meaningful_words))
	return meaningful_words

def sample_handling(sample, lexicon, classification):
	featureset = []

	with open(sample, 'r') as f:
		contents = f.readlines()
		for line in contents[:hm_lines]:
			current_words = word_tokenize(line.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]

			features = np.zeros(len(lexicon))
			for w in current_words:
				if w.lower() in lexicon:
					index_value = lexicon.index(w.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features, classification])

	print(featureset[:2])
	return featureset

#l = create_lexicon('pos.txt', 'neg.txt')
#f = sample_handling('pos.txt', l, [1,0])

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling(pos, lexicon, [1,0])
	features += sample_handling(neg, lexicon, [0,1])
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(test_size * len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x  = list(features[:,0][-testing_size:])
	test_y  = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y

	
if __name__ == "__main__":
 	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
 	with open("sentiment.pickle", 'wb') as f:
 		pickle.dump([train_x, train_y, test_x, test_y], f)



