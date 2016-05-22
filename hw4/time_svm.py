import scipy.io as sio
import numpy as np
from sklearn import svm, metrics, datasets, preprocessing
from sklearn.decomposition import PCA
import sys

# run PCA and run SVM
def run(train_images, train_labels, test_images, test_labels, kernel_type, type_multi):
    # reshape the training/testing data into a classifiable form
    train_data_mat = np.reshape(train_images, (train_images.shape[0]*train_images.shape[1],train_images.shape[3]))
    test_data_mat = np.reshape(test_images, (test_images.shape[0]*test_images.shape[1],test_images.shape[3]))

    train_data_mat = np.array(np.mat(train_data_mat).transpose())
    test_data_mat = np.array(np.mat(test_data_mat).transpose())

    # normalize the data
    train_data_mat = preprocessing.normalize(train_data_mat, norm='l2')
    test_data_mat = preprocessing.normalize(test_data_mat, norm='l2')
    
    # fit svm to pca-reduced
    classif_1vr = svm.SVC(kernel=kernel_type)
    classif_1vr.fit(train_data_mat, train_labels[0])
    
if __name__ == "__main__":
    raw_digits = sio.loadmat('/mnt0/siajat/cs391L/data/digits.mat')

    train_images = raw_digits['trainImages']
    train_labels = raw_digits['trainLabels']
    test_images = raw_digits['testImages']
    test_labels = raw_digits['testLabels']

    run(train_images, train_labels, test_images, test_labels, "linear", sys.argv[1])
