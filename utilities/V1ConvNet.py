
from __future__ import print_function
import os #Selects which gpu is to be used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import time
import h5py
import pydot
import matplotlib.pyplot as plt
#import Keras and options
import keras
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import tensorflow as tf

#Custom scripts
import DataHandler
import Analysis
import LucasAnalysis

#Models import files
import ShallowNet
#import ShallowNetTest
import VGG19
import vggFile
import Resnet2 as resnet
#from inception_v3 import InceptionV3
#from xception import Xception
#import imagenet_utils
#import LeakyResnet as resnet
##import fc_test

np.random.seed(1337)  # for reproducibility

#Saving output files
save_model = True
save_schematic = False
save_plots = True
data_augmentation = False

#Set some parameters
batch_size = 500
nb_classes = 2
nb_epoch = 10
img_channels, img_rows, img_cols = 1, 25, 25
storage_sig = "/mnt/storage/lborgna/WprimePreP/"
storage_bkg = "/mnt/storage/lborgna/DijetPreP/"
#storage_sig = "/mnt/storage/lborgna/FullSupervisedData/"
#storage_bkg = "/mnt/storage/lborgna/FullSupervisedData/"


print("Loading Data:Training")
#sig_file = "Sig_m5000_Train_0_30000.npy"
#bkg_file = "Bkg_m5000_Train_0_30000.npy"
sig_file = "Pre_Bkg_All.root"
bkg_file = "Pre_SignalAll.root"
bkArray, sigArray  = DataHandler.Import(storage_bkg +  bkg_file, storage_sig + sig_file)
#bkgArray, sigArray = DataHandler.Loader(storage_bkg +  bkg_file, storage_sig + sig_file)

print("Train-Test-Split")
TrainList, TestList, TrainVals, TestVals, input_shape = DataHandler.SliceAndDice(bkgArray, sigArray, 0.7, K.image_dim_ordering(), img_rows, img_cols)

# convert class vectors to binary class matrices: Required by Categorical Cross-entropy [1] -> [1 0 ] & [0] -> [0 1]
TrainVals = np_utils.to_categorical(TrainVals, nb_classes)
TestVals = np_utils.to_categorical(TestVals, nb_classes)

#Create Model, Compile and Train
print("Getting Network")
model = ShallowNet.GetNetArchitecture(input_shape)
#model  = vggFile.GetNetArchitecture(input_shape)
#model = VGG19.VGG_19(input_shape)
#model = fc_test.GetNetArchitecture(input_shape)
#model = InceptionV3(input_shape = input_shape, include_top=False, classes = nb_classes)
#model  = Xception(input_shape = input_shape, include_top = False, classes = nb_classes)
#model = resnet.ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
#Addresnet model

print("Training Network")
model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

"""

if not data_augmentation:
        print ('Not Using Data Augmentation.')
        history = model.fit(TrainList, TrainVals,
                            batch_size = batch_size,
                            nb_epoch = nb_epoch,
                            validation_data = (TestList, TestVals),
                            shuffle = True, 
"""

history = model.fit(TrainList, TrainVals, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(TestList, TestVals))

print("Evaluating Network")

score = model.evaluate(TestList, TestVals, verbose=0)

#Predictions_Test = model.predict(TestList, verbose=1)


#Reformat Results
#TestVals = TestVals[:,0]
#Predictions_Test = Predictions_Test[:,0]


# print("testvals")
# print(TestVals)
# print("Predictions")
# print(Predictions_Test)
# print("lens")
# print(len(Predictions_Test))
# print("Unique vals")
# print(np.unique(Predictions_Test))
# print(len(np.unique(Predictions_Test)))

#Call Analysis Function
#Analysis.generate_results(TestVals, Predictions_Test)
#Analysis.GetClassificationMetrics(TestVals, Predictions_Test)


# Print Final Scores
print('Test score:', score[0])
print('Test accuracy:', score[1])

# plot the model schematic (for reports / documentation)
scriptname = os.path.basename(__file__)
schematic_name = scriptname+".png"


print("Saving Model")

# save model
timestr = time.strftime("at_%Y.%m.%d_%H.%M.%S")
namestr = scriptname+"_Model_"
extstr = ".h5"
FullNameStr = namestr+timestr+extstr


if (save_model == True):
	model.save("/mnt/storage/lborgna/TrainedModels/"+FullNameStr)
	print("Model Saved as, %s", FullNameStr)
if (save_schematic == True):
	plot_model("/mnt/storage/lborgna/Schematics/"+schematic_name+FullNameStr-".h5"+".png")
	print("Schematic Saved")

print(FullNameStr)
print(history.history.keys())


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(history.history['acc'],'g-')
ax1.plot(history.history['val_acc'],'g--')

ax2.plot(history.history['loss'], 'b-')
ax2.plot(history.history['val_loss'], 'b--')
ax1.set_xlabel('Number of epochs')
ax1.set_ylabel('Accuracy', color = 'g')
ax2.set_ylabel('Loss', color = 'b')
plt.savefig('training_'+timestr+'.png')
#plt.show()

"""

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy per epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.savefig('acc-epoch.png')
#plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss per epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.savefig('loss-epoch.png')
#plt.show()
"""

#----------------------------------------------------
# 		Generating ROC Curve
#----------------------------------------------------

test_bkg = "Bkg_m5000_Test_30000_60000.npy"
test_sig = "Sig_m5000_Test_0_30000.npy"
bkgTest, sigTest = DataHandler.Loader(storage_bkg + test_bkg, storage_sig + test_sig)

#TestList, TestVals, input_shape = DataHandler

X_Test, y_Test, input_shape = DataHandler.AllIn(bkgTest, sigTest, K.image_dim_ordering(), img_rows, img_cols)

Y_Test = np_utils.to_categorical(y_Test, nb_classes)

score = model.evaluate(X_Test, Y_Test, verbose = 0)
Predictions_Test = model.predict(X_Test, verbose = 1)

Y_Test = Y_Test[:,1]
Predictions_Test = Predictions_Test[:,1]

print("testvals: Y_Test")
print(Y_Test)
print("Predictions")
print(Predictions_Test)
print("lens")
print(len(Predictions_Test))
print("Unique Vals")
print(np.unique(Predictions_Test))
name = "ROC"+timestr
LucasAnalysis.generate_results(Y_Test, Predictions_Test, name+".png")
LucasAnalysis.Save_Results(Y_Test, Predictions_Test, name)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

