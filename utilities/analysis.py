
###################################################
#                                                 #
#   Master's Project Analysis Module              #
#   Version 0.6                                   #
#   Sam Wright February 10th 2017                 #
#                                                 #
###################################################

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, precision_score, confusion_matrix, recall_score,f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
#import ROOT
import numpy as np
import itertools


#Function modified from Vijaya Kolachalama, posted 29th May 2016
#Available https://vkolachalama.blogspot.co.uk/2016/05/keras-implementation-of-mlp-neural.html

def generate_results(testvals, predictionstest,outputfilename):
    fpr, tpr, _ = roc_curve(testvals, predictionstest)
    #SW: Note - auc fn. uses trapezoid rule
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(outputfilename)
    print('\nAUC: %f' % roc_auc)
    
def kfold_roc(testvals, predictionstest):
    fpr, tpr, _ = roc_curve(testvals, predictionstest)
    #SW: Note - auc fn. uses trapezoid rule
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
    #plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('K-fold ROC Curve For Validation (VGG16)')
    plt.legend(loc='best')
    plt.grid(True)
    #plt.show()
    print('\nAUC: %f' % roc_auc)


def save_results(Y_test, predictions_test, timestamp):
    fpr, tpr, _ = roc_curve(Y_test, predictions_test)
    roc_auc = auc(fpr, tpr)
    storage = "/home/lborgna/NN/V2ConvNet/roc_curves/roctrainingarrays/"
    np.save(storage + "tpr_" + timestamp + '.npy', tpr)
    np.save(storage + 'fpr_' + timestamp + '.npy', fpr)
    np.save(storage + 'auc_' + timestamp + '.npy', roc_auc)



"""

def save_results(testvals, predictionstest, timestamp):
	fpr, tpr, _ = roc_curve(testvals, predictionstest)
	roc_auc = auc(fpr, tpr)
    store = "/home/lborgna/NN/V2ConvNet/roc_curves/roctrainingarrays/"

	np.save(store+"tpr_"+timestamp+".npy",tpr)
	np.save(store+"fpr_" + timestamp + ".npy",fpr)
	np.save(store+"auc_" + timestamp + ".npy", roc_auc)

"""



def plot_confusion_matrix(timestamp, cm, classes, normalize = False, cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    if normalize:
	cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
	print("Normalized Confusion Matrix")
    else:
	print("Confusion Matrix, Without Normalization")
    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	plt.text(j, i, format(cm[i, j], fmt),
		horizontalalignment="center",
		color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/lborgna/NN/V2ConvNet/cm/cm_' + timestamp +'.png')
