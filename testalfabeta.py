import numpy as np
from sklearn import datasets
from sklearn.cross_validation import LeaveOneOut

from alfabeta import AlfaBeta
from encoder import BaseEncoder, IntegerEncoder

from collections import Counter

iris = datasets.load_iris()
Y = iris.target
ds = np.column_stack( (iris.data, Y) )
#ds = np.genfromtxt( './datasets/ecoli.csv', delimiter=",", filling_values=0 )

# print encoder.positive()
# print encoder.integer()

N = np.size( ds, axis=0 )
loo = LeaveOneOut( N )
correct = 0
for train_index, test_index in loo:
	train_ds = ds[train_index]
	test = ds[test_index][0]


	#integer_encoder = IntegerEncoder( train_ds[:, 0:-1] )

	#train_ds_integer = integer_encoder.encode()
	#test_integer = integer_encoder.encode_single( test[0:-1] )


	#base_encoder = BaseEncoder( train_ds_integer, base=2 )
	#train_ds_based = base_encoder.encode()
	#train_ds_encoded = np.column_stack( (train_ds_integer, train_ds[:, -1]) )
	#test_encoded = np.array( test_integer )
	#classifier = AlfaBeta( train_ds_encoded )
	#classifier.train()
	#c = classifier.classify( test_encoded )

	classifier = AlfaBeta( train_ds )
	classifier.train()
	c = classifier.classify( test[0:-1] )

	if c == test[-1]:
		correct += 1

	print test, int( test[-1] ), '->', c

print '\nAccuracy: ', float( correct ) / float( N )
