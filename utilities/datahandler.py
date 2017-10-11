from __future__ import print_function
#import root_numpy
#import ROOT
import numpy as np
import time
import math



def Loader(BkgArrayFileName, SigArrayFileName):

	BkgArray = np.load(BkgArrayFileName)
	SigArray = np.load(SigArrayFileName)

	return BkgArray, SigArray


def create_labels(bkgArray, sigArray):
	bkgArray = bkgArray.tolist()
	sigArray = sigArray.tolist()

	bkgVal = np.zeros(len(bkgArray))
	sigVal = np.ones(len(sigArray))

	X  = bkgArray + sigArray
	y  = np.concatenate((bkgVal, sigVal), axis = 0)

	X = np.array(X)
	y = np.array(y)

	return X, y


def create_fold(Sig, Bkg, fold_iteration, k_fold, backend, img_channels, img_rows, img_cols):

	#Assuming the fold_iteration variable starts at 0
	S = Sig.tolist()
	B = Bkg.tolist()

	Bfraction = int(math.floor(len(B)/k_fold)) #Fold_size
	Sfraction = int(math.floor(len(S)/k_fold)) #fold_size

	j = fold_iteration
	bkgTrain = []
	bkgTest = []
	sigTrain = []
	sigTest  = []

	bkgTrain = B[(j+1)*Bfraction: len(B)] + B[0: j*Bfraction]
	bkgTest = B[(j)*Bfraction: (j+1)*Bfraction]

	sigTrain = S[(j+1)*Sfraction: len(S)] + S[0: j*Sfraction]
	sigTest  = S[(j) * Sfraction: (j+1)*Sfraction]

	bkgTrainVal = np.zeros(len(bkgTrain))
	bkgTestVal  = np.zeros(len(bkgTest))

	sigTrainVal = np.ones(len(sigTrain))
	sigTestVal  = np.ones(len(sigTest))

	X_train = bkgTrain + sigTrain
	X_test  = bkgTest  + sigTest

	#Y_train = bkgTrainVal + sigTrainVal
	#Y_test  = bkgTestVal  + sigTestVal

	Y_train = np.concatenate((bkgTrainVal, sigTrainVal), axis = 0)
	Y_test  = np.concatenate((bkgTestVal, sigTestVal), axis = 0)

	X_train = np.array(X_train)
	X_test  = np.array(X_test)

	if backend == 'tf':
		input_shape = (X_train.shape[1], X_train.shape[2], 1)
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
		X_test  = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
	elif backend == 'th':
		input_shape = (1, X_train.shape[1], X_train.shape[2])
		X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
		X_test  = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
	else:
		print("Backend Was Not Recognised, please check your ~/.keras/keras.json file")

	return X_train, X_test, Y_train, Y_test, input_shape


def SliceAndDice(bkgArray, sigArray, trainingFraction, backend, img_rows, img_cols):
	bkgArray = bkgArray.tolist()
	sigArray = sigArray.tolist()

	bkgFrac = int(trainingFraction * len(bkgArray))
	sigFrac = int(trainingFraction * len(sigArray))

	bkgTrain, bkgTest = bkgArray[:bkgFrac], bkgArray[bkgFrac:]
	sigTrain, sigTest = sigArray[:sigFrac], sigArray[sigFrac:]

	bkgTrainVal = np.zeros(len(bkgTrain))
	bkgTestVal = np.zeros(len(bkgTest))

	sigTrainVal = np.ones(len(sigTrain))
	sigTestVal = np.ones(len(sigTest))

	X_train = bkgTrain + sigTrain
	X_test = bkgTest + sigTest

	Y_train = np.concatenate((bkgTrainVal, sigTrainVal), axis=0)
	Y_test = np.concatenate((bkgTestVal, sigTestVal), axis=0)

	X_train = np.array(X_train)
	X_test = np.array(X_test)

	#Shape_Array = [X_train.shape[0], X_train.shape[1], X_train.shape[2]]
	#print(Shape_Array)

	# Format Output Depending on Tensor Manipulation Backend
	if backend == "tf":
		input_shape = (X_train.shape[1], X_train.shape[2], 1)
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

	elif backend == "th":
		input_shape = (1, X_train.shape[1], X_train.shape[2])
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
		X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols, )
	else:
		print("Unrecognised Backend")

	return X_train, X_test, Y_train, Y_test, input_shape



def create_shape(X_train, X_test, Y_train, Y_test, backend, img_rows, img_cols):

	# Format Output Depending on Tensor Manipulation Backend
	if backend == "tf":
		input_shape = (X_train.shape[1], X_train.shape[2], 1)
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

	elif backend == "th":
		input_shape = (1, X_train.shape[1], X_train.shape[2])
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
		X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols, )
	else:
		print("Unrecognised Backend")

	return X_train, Y_train, X_test, Y_test, input_shape





def AllIn(bkgArray, sigArray, backend, img_rows, img_cols):

	bkgArray = bkgArray.tolist()
	sigArray = sigArray.tolist()

	bkgTestVal = np.zeros(len(bkgArray))
	sigTestVal = np.ones(len(sigArray))

	X_test = bkgArray + sigArray

	Y_test = np.concatenate((bkgTestVal, sigTestVal),axis=0)

	X_test = np.array(X_test)

	#Format Output Depending on Tensor Manipulation Backend
	if backend == "tf":
		input_shape = ( X_test.shape[1] , X_test.shape[2], 1)
		X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

	elif backend == "th":
		input_shape = (1, X_test.shape[1] , X_test.shape[2])
		X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols,)
	else:
		print("Unrecognised Backend")

	return X_test, Y_test, input_shape

def writeinfofile(mylist1, mylist2, folder, scriptname, timestamp):
    outputfile = folder+scriptname +"_" + timestamp + ".txt"
    f = open(outputfile, 'w')

    for i in xrange(len(mylist1)):
        f.write("%s:\t%s\n" %(mylist1[i], mylist2[i]))

    f.close()

