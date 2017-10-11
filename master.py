from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import numpy as np
import time
import h5py
import pydot
import matplotlib.pyplot as plt
import keras
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import pydot
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import networks.LeNet5  as LeNet5
import networks.VGG19   as VGG19
import networks.VGG16   as VGG16
import networks.Resnet2 as resnet
import networks.FC      as FC
import networks.bn_shallow as BN_LeNet5
import networks.Conv5   as Conv5
import networks.LeNetTanh as LeNet5Tanh
import networks.TR_LeNet5 as TR_LeNet5
import networks.SingleCN as CN1

import utilities.datahandler as datahandler
import utilities.analysis    as analysis
import utilities.read_activations as read_activations

from datapath import sig_train, bkg_train, sig_test, bkg_test


def create_model(network, input_shape, img_channels, img_rows, img_cols, nb_classes):
    print("Acquring Network Model: ")
    if network == "LeNet5":
        model = LeNet5.GetNetArchitecture(input_shape)
        model_name = "LeNet5"

    elif network == "VGG16":
        model = VGG16.GetNetArchitecture(input_shape)
        model_name = "VGG16"

    elif network == "VGG19":
        model = VGG19.GetNetArchitecture(input_shape)
        model_name = "VGG19"

    elif network == "resnet18":
        model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet18"

    elif network == "resnet34":
        model = resnet.ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet34"

    elif network == "resnet50":
        model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet50"

    elif network == "resnet101":
        model = resnet.ResnetBuilder.build_resnet_101((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet101"

    elif network == "resnet152":
        model = resnet.ResnetBuilder.build_resnet_152((img_channels, img_rows, img_cols), nb_classes)
        model_name = "Resnet152"

    elif network == 'BN_LeNet5':
        model = BN_LeNet5.GetNetArchitecture(input_shape)
        model_name = 'BN_LeNet5'

    elif network == 'FC':
        model = FC.GetNetArchitecture(input_shape)
        model_name = 'FC'

    elif network =='Conv5':
        model = Conv5.GetNetArchitecture(input_shape)
        model_name = 'Conv5'
    elif network =="LeNetTanh":
        model = LeNet5Tanh.GetNetArchitecture(input_shape)
        model_name = "LeNetTanh"
    elif network == "TR_LeNet5":
        model = TR_LeNet5.GetNetArchitecture(input_shape)
        model_name = "TR_LeNet5"
    elif network =="CN1":
        model = CN1.GetNetArchitecture(input_shape)
        model_name = "CN1"
    return model, model_name

def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, data_augmentation, no_callbacks):


    if data_augmentation == 'True':
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images  # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=(X_test, Y_test),
                              epochs=nb_epoch, verbose=1, max_q_size=100,
                              callbacks = [lr_reducer, early_stopper])
        return history

    elif no_callbacks == 'True':
        print('Not Using Data Augmentation or Early Stopping and LR Reducer')
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs= nb_epoch,
                            validation_data= (X_test, Y_test),
                            shuffle = True)
        return history

    else:
        print ('Not Using Data Augmentation.')
        history = model.fit(X_train, Y_train,
                            batch_size = batch_size,
                            epochs = nb_epoch,
                            validation_data = (X_test, Y_test),
                            shuffle = True,
                            callbacks=[lr_reducer, early_stopper]) # Do we need to shuffle?
        return history


if __name__ == "__main__":

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)

    np.random.seed(1337)
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S")

    batch_size = 500
    nb_classes = 2
    nb_epoch = 50
    img_channels, img_rows, img_cols = 1, 25, 25
    test_size = 0.7
    backend = K.image_dim_ordering()
    k_folds = 1

    data_augmentation = 'False'
    savemodel = 'True'
    save_schematic ='True'
    save_plots = 'True'
    save_roc = 'True'
    save_cm = 'True'
    precision_recall = 'True'
    no_callbacks ='False'
    conv_filter = 'False'
    model_summary = 'False'

    scriptname = os.path.basename(__file__)


    bkgArray, sigArray = datahandler.Loader(bkg_train, sig_train)


    network = "BN_LeNet5" #LeNet5, VGG16, VGG19, resnet18, 34, 50, 101, 152

    loss_function = 'categorical_crossentropy'
    optimizer = 'adadelta'
    plt.figure()
    for j in range (0, k_folds):
        if k_folds == 1:
            X_train, X_test, Y_train, Y_test, input_shape = datahandler.SliceAndDice(bkgArray, sigArray, test_size, backend, img_rows, img_cols)
        else:
            X_train, X_test, Y_train, Y_test, input_shape = datahandler.create_fold(sigArray, bkgArray, j, k_folds, backend, img_channels, img_rows, img_cols)

        print ("Running Fold: ",j+1 ,"/", k_folds)

        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)

        model = None # Clearing the NN.
        model, model_name = create_model(network, input_shape, img_channels, img_cols, img_rows, nb_classes)
        model.compile(loss = loss_function, optimizer = optimizer, metrics = ['accuracy'])

        history = train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, data_augmentation, no_callbacks)
        y_score = model.predict(X_test, verbose = 0)
        Y_test = Y_test[:,1]
        y_score = y_score[:,1]
        kfoldroc = '/home/lborgna/NN/V2ConvNet/kfoldroc/kroc_'+timestamp +'.png'
        analysis.kfold_roc(Y_test, y_score)


    if k_folds != 1:
        plt.savefig(kfoldroc)

    if model_summary == 'True':
        orig_stdout = sys.stdout
        f = open('/home/lborgna/NN/V2ConvNet/summary/'+model_name+timestamp+'.txt', 'w')
        sys.stdout = f
        print(model.summary())
        sys.stdout = orig_stdout
        f.close()

    print (sigArray[0].shape)
    print (X_test[0:1].shape)
    sig_img = sigArray[0].reshape(1, img_rows, img_cols, 1)
    if conv_filter == "True":
        a = read_activations.get_activations(model, sig_img, print_shape_only=True)  # with just one sample.

        read_activations.display_activations(a)

    plt.clf()

    model_storage = "/mnt/storage/lborgna/TrainedModels/TrainFull/"
    info_storage = model_storage +"info/"

    mylist1 = ['Sig Train', 'Bkg train', 'Sig Test', 'Sig Test',"Time and Date: ",'Backend: ', 'input_shape', "Network Model: ", "Epochs: ", 'batch_size: ', 'test_size: ',  " K folds: ", "Data Augmentation: ","Loss Function: ", "Optimizer: "]
    mylist2 = [sig_train, bkg_train, sig_test,bkg_test,timestamp, backend, input_shape ,model_name, nb_epoch, batch_size,test_size, k_folds, data_augmentation, loss_function, optimizer]

    redundancy = "/home/lborgna/NN/V2ConvNet/info/"

    if savemodel:
        print ("Storing Information File: ")
        datahandler.writeinfofile(mylist1 = mylist1, mylist2 = mylist2, folder = info_storage, scriptname = scriptname, timestamp = timestamp)
        datahandler.writeinfofile(mylist1 = mylist1, mylist2 = mylist2, folder = redundancy, scriptname = scriptname, timestamp = timestamp)
        print("Saving Model: Final")
        namestr = os.path.basename(__file__)
        ext = ".h5"
        model.save(model_storage + namestr + model_name + timestamp + ext)

    if save_schematic:
        schematic_name = model_name+timestamp+"schematic.png"
        plot_model(model, "/home/lborgna/NN/V2ConvNet/schematics/"+schematic_name)
        print("schematic saved: ", schematic_name)

    if save_plots:
        fig, ax1 = plt.subplots()
        plt.grid(b=True, which = 'major', color='k', linestyle = '-')
        plt.grid(b=True, which = 'minor', color='k', linestyle = '-')
        ax2 = ax1.twinx()
        acc, = ax1.plot(history.history['acc'], 'g-')
        valacc, = ax1.plot(history.history['val_acc'], 'g--')

        loss, = ax2.plot(history.history['loss'], 'b-')
        valloss, = ax2.plot(history.history['val_loss'], 'b--')
        ax1.set_xlabel('Number of Epochs')
        ax1.set_ylabel('Accuracy', color = 'g')
        ax1.tick_params(axis='y', colors='green', which='both')
        ax1.yaxis.label.set_color('g')
        ax2.set_ylabel('loss', color = 'b')
        ax2.tick_params(axis='y', colors='blue', which='both')
        ax2.yaxis.label.set_color('b')
        lgd = plt.legend((acc, valacc, loss, valloss), ('accuracy', 'validation accuracy', 'loss', 'validation loss'),
                         loc='upper center', bbox_to_anchor=(0.5, -0.10), shadow=True, ncol=4)


        plt.title('Model Training Performance', fontsize = 20)
        plt.tight_layout()
        plt.savefig('/home/lborgna/NN/V2ConvNet/training_plots/train_'+timestamp+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        np.save('/home/lborgna/NN/V2ConvNet/training_plots/trainingarrays/acc_'+timestamp+'.npy',history.history['acc'])
        np.save('/home/lborgna/NN/V2ConvNet/training_plots/trainingarrays/valacc_'+timestamp+'.npy',history.history['val_acc'])
        np.save('/home/lborgna/NN/V2ConvNet/training_plots/trainingarrays/loss_'+timestamp+'.npy',history.history['loss'])
        np.save('/home/lborgna/NN/V2ConvNet/training_plots/trainingarrays/valloss_'+timestamp+'.npy',history.history['val_loss'])




    if save_roc or precision_recall:


        plt.clf()
        bkgTest, sigTest = datahandler.Loader(bkg_test, sig_test)

        X_Test, y_Test, input_shape = datahandler.AllIn(bkgTest, sigTest, K.image_dim_ordering(), img_rows, img_cols)
        Y_Test = np_utils.to_categorical(y_Test, nb_classes)

        score = model.evaluate(X_Test, Y_Test, verbose = 0)
        Predictions_Test = model.predict(X_Test, verbose = 1)
        Y_Test = Y_Test[:, 1]
        Predictions_Test = Predictions_Test[:, 1]
        rocoutputfile = "/home/lborgna/NN/V2ConvNet/roc_curves/roc_" + timestamp + ".png"

        if save_roc:
            analysis.generate_results(Y_Test, Predictions_Test, rocoutputfile)
            analysis.save_results(Y_Test, Predictions_Test, timestamp)
            print('Test score: ', score [0])
            print('Test Accuracy: ', score[1])
        if precision_recall:
            average_precision = average_precision_score(y_Test, Predictions_Test)
            print('Average precision-recall score: {0:0.2f}'.format(average_precision))

            plt.clf()
            precision, recall, _ = precision_recall_curve(y_Test, Predictions_Test)
            plt.plot(recall, precision, color = 'b', alpha = 0.2)
            plt.fill_between(recall, precision, alpha = 0.2, color = 'b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall curve AUC: {0:0.4f}'.format(average_precision))
            plt.savefig('/home/lborgna/NN/V2ConvNet/precision_recall/PR_' + timestamp + '.png')
            np.save('/home/lborgna/NN/V2ConvNet/precision_recall/PR_arrays/precision_'+timestamp+'.npy', precision)
            np.save('/home/lborgna/NN/V2ConvNet/precision_recall/PR_arrays/recall_'+timestamp+'.npy', recall)
            np.save('/home/lborgna/NN/V2ConvNet/precision_recall/PR_arrays/aucpr_' + timestamp + '.npy', average_precision)

    if save_cm:
        plt.clf()
        class_names = ['Signal', 'Background']
        normalize = True
        cnf_matrix = confusion_matrix(Y_Test, Predictions_Test.round())
        np.set_printoptions(precision = 2)
        analysis.plot_confusion_matrix(timestamp, cnf_matrix, classes = class_names, normalize = normalize)





    print("All Done - Timestamp: ", timestamp)
