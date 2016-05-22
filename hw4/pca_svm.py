import scipy.io as sio
import numpy as np
from sklearn import svm, metrics, datasets, preprocessing
from sklearn.decomposition import PCA
import pickle

# run PCA and run SVM
def PCA_SVM(train_images, train_labels, test_images, test_labels, kernel_type, do_PCA, comps):
    # reshape the training/testing data into a classifiable form
    train_data_mat = np.reshape(train_images, (train_images.shape[0]*train_images.shape[1],train_images.shape[3]))
    test_data_mat = np.reshape(test_images, (test_images.shape[0]*test_images.shape[1],test_images.shape[3]))

    train_data_mat = np.array(np.mat(train_data_mat).transpose())
    test_data_mat = np.array(np.mat(test_data_mat).transpose())

    # normalize the data
    train_data_mat = preprocessing.normalize(train_data_mat, norm='l2')
    test_data_mat = preprocessing.normalize(test_data_mat, norm='l2')
    
    # do PCA if necessary
    if do_PCA:
        # learn the covariance 
        pca = PCA(n_components=comps, whiten=True)
        pca.fit(train_data_mat)
    
        # use pca to reduce dimensionality of training data
        train_data_mat = pca.transform(train_data_mat)
        test_data_mat = pca.transform(test_data_mat)
    
    # fit svm to pca-reduced
    classif_1vr = svm.SVC(kernel=kernel_type)
    classif_1vr.fit(train_data_mat, train_labels[0])

    targets = test_labels[0]
    
    # make prediction on the test data set
    predict = classif_1vr.predict(test_data_mat)
 
    # calculate the accuracy 
    acc = calc_acc(targets, predict) 

    return "PCA=" + str(do_PCA) + ", num_comps= " + str(comps) + ", kernel=" + str(kernel_type)  + ", acc: " + str(acc) + "\n"


# run PCA and run SVM
def tune_SVM(train_images, train_labels, test_images, test_labels, kernel_type, tune1, tune2, tune3):
    # reshape the training/testing data into a classifiable form
    train_data_mat = np.reshape(train_images, (train_images.shape[0]*train_images.shape[1],train_images.shape[3]))
    test_data_mat = np.reshape(test_images, (test_images.shape[0]*test_images.shape[1],test_images.shape[3]))

    train_data_mat = 1.0*np.array(np.mat(train_data_mat).transpose())
    test_data_mat = 1.0*np.array(np.mat(test_data_mat).transpose())

    # normalize the data
    train_data_mat = preprocessing.normalize(train_data_mat, norm='l2')
    test_data_mat = preprocessing.normalize(test_data_mat, norm='l2')
     
    
    if kernel_type is "linear": 
        classif_1vr = svm.SVC(kernel=kernel_type)
    elif kernel_type is "rbf": 
        classif_1vr = svm.SVC(kernel=kernel_type, gamma=tune1)
    elif kernel_type is "sigmoid":
        classif_1vr = svm.SVC(kernel=kernel_type, gamma=tune1, coef0=tune2)
    elif kernel_type is "poly":
        classif_1vr = svm.SVC(kernel=kernel_type, gamma=tune1, coef0=tune2, degree=tune3)
    
    # fit the SVM to the training set
    classif_1vr.fit(train_data_mat, train_labels[0])
    
    targets = test_labels[0]
    
    # make prediction on the test data set
    predict = classif_1vr.predict(test_data_mat)
 
    # calculate the accuracy 
    acc = calc_acc(targets, predict) 

    return "kernel=" + str(kernel_type)  + ", tune1=" + str(tune1)  + ", tune2=" + str(tune2) + ", tune3=" + str(tune3) + ", acc: " + str(acc) + "\n"


def SVM_vary_train(train_images, train_labels, test_images, test_labels, kernel_type, tune1, tune2, tune3, train_amm):
    # reshape the training/testing data into a classifiable form
    train_data_mat = np.reshape(train_images, (train_images.shape[0]*train_images.shape[1],train_images.shape[3]))
    test_data_mat = np.reshape(test_images, (test_images.shape[0]*test_images.shape[1],test_images.shape[3]))

    train_data_mat = 1.0*np.array(np.mat(train_data_mat).transpose())
    test_data_mat = 1.0*np.array(np.mat(test_data_mat).transpose())

    # normalize the data
    train_data_mat = preprocessing.normalize(train_data_mat, norm='l2')
    test_data_mat = preprocessing.normalize(test_data_mat, norm='l2')
     
    
    if kernel_type is "linear": 
        classif_1vr = svm.SVC(kernel=kernel_type, C=tune1)
    elif kernel_type is "rbf": 
        classif_1vr = svm.SVC(kernel=kernel_type, gamma=tune1)
    elif kernel_type is "sigmoid":
        classif_1vr = svm.SVC(kernel=kernel_type, gamma=tune1, coef0=tune2)
    elif kernel_type is "poly":
        classif_1vr = svm.SVC(kernel=kernel_type, gamma=tune1, coef0=tune2, degree=tune3)
    
    # fit the SVM to the training set
    classif_1vr.fit(train_data_mat[0:train_amm,:], train_labels[0][0:train_amm])
    
    targets = test_labels[0]
    
    # make prediction on the test data set
    predict = classif_1vr.predict(test_data_mat)
 
    # calculate the accuracy 
    acc = calc_acc(targets, predict) 

    return "kernel=" + str(kernel_type)  + ", tune1=" + str(tune1)  + ", tune2=" + str(tune2) + ", tune3=" + str(tune3) + ", train_amm=" + str(train_amm) + ", acc: " + str(acc) + "\n"




def calc_acc(targets, predict):
    num_correct = sum([1 for i in xrange(0, len(targets)) if targets[i] == predict[i]])
    acc = num_correct*1.0 / len(targets)*1.0
    return acc


if __name__ == "__main__":
    raw_digits = sio.loadmat('/mnt0/siajat/cs391L/data/digits.mat')

    train_images = raw_digits['trainImages']
    train_labels = raw_digits['trainLabels']
    test_images = raw_digits['testImages']
    test_labels = raw_digits['testLabels']
    
    '''
    print train_images.shape
    
    # reshape the training/testing data into a classifiable form
    train_data_mat = np.reshape(train_images, (train_images.shape[0]*train_images.shape[1],train_images.shape[3]))
    test_data_mat = np.reshape(test_images, (test_images.shape[0]*test_images.shape[1],test_images.shape[3]))

    train_data_mat = np.array(np.mat(train_data_mat).transpose())
    test_data_mat = np.array(np.mat(test_data_mat).transpose())
    
    

    # normalize the data
    train_data_mat = preprocessing.normalize(train_data_mat, norm='l2')
    test_data_mat = preprocessing.normalize(test_data_mat, norm='l2')
    
    # learn the covariance 
    pca = PCA(n_components=100, whiten=True)
    pca.fit(train_data_mat)
    
    # use pca to reduce dimensionality of training data
    train_data_mat_red = pca.transform(train_data_mat)
    test_data_mat_red = pca.transform(test_data_mat)
    print train_data_mat_red.shape
    print test_data_mat_red.shape
    
    # fit svm to pca-reduced
    classif_1vr = svm.svc(kernel='linear')
    #classif_1vR.fit(train_data_mat[0:10000,], train_labels[0][0:10000,])
    classif_1vR.fit(train_data_mat_red, train_labels[0])
    #print np.reshape(train_data_mat[2,], (28,28))

    # make predictions on the test data set
    #targets = test_labels[0:10000, ]
    predict = classif_1vR.predict(test_data_mat_red)
 
    print predict

    pickle.dump(test_labels[0], open('targets.p','wb'))
    pickle.dump(predict, open('predict.p','wb'))
    '''
    
    # PCA experiments
    with open('pca_svm.txt', 'w') as outfile:
        for comps in xrange(10, 100, 5):
            curr = PCA_SVM(train_images, train_labels, test_images, test_labels, "linear", True, comps)
            outfile.write(curr)

    # linear SVM with no parameter tuning (C=1.0)
    with open('full_svm.txt', 'w') as outfile:
        curr = PCA_SVM(train_images, train_labels, test_images, test_labels, "linear", False, 10)
        outfile.write(curr) 
    
    # rbf with parameter tuning
    with open('rbf_svm.txt', 'w') as outfile:
        for gamma in [0.0001, 0.001, 0.1, 1.0 ,10.0, 100.0, 1000.0]:
            curr = tune_SVM(train_images, train_labels, test_images, test_labels, "rbf", gamma, "","")
            outfile.write(curr)
    
    # <<DID NOT WORK>>
    # polynomial kernel tuning the degree
    #with open('poly_svm.txt', 'w') as outfile:
    #    for degree in [1,2,3,4,5,6,7,8,9,10]:
    #        curr = tune_SVM(train_images, train_labels, test_images, test_labels, "poly", 'auto', 0.0, degree)
    #        outfile.write(curr)
    
    with open('size_svm.txt','w') as outfile:
        for size in [10000, 20000, 30000, 40000, 50000, 60000]:
            curr = SVM_vary_train(train_images, train_labels, test_images, test_labels, "rbf", 1.0, "","", size)
            outfile.write(curr)

    # linear with tuning (test SVM))
    with open('linear_svm.txt', 'w') as outfile:
        for C in [0.0001, 0.001, 0.1, 1.0 ,10.0, 100.0, 1000.0]:
            curr = SVM_vary_train(train_images, train_labels, test_images, test_labels, "linear", C, "","", 60000)
            outfile.write(curr)
