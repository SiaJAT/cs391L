import sys
import pickle
import numpy as np
import scipy.io as sio
from scipy.spatial import distance

class ReturnTuple(object):
	def __init__(self, mean_vector, eigen_vector_matrix, eigen_vals, cov_mat):
			self.mean_vector = mean_vector
			self.eigen_vector_matrix = eigen_vector_matrix
			self.eigen_vals = eigen_vals
			self.cov_mat = cov_mat

class ClassifTuple(object):
	def __init__(self, acc_L1, acc_L2):
		self.acc_L1 = acc_L1
		self.acc_L2 = acc_L2

def gen_eigenvector_matrix(mat):
	num_examps, vec_len = mat.shape[1], mat.shape[0]
	
	# sum across all training examples and average to get mean vector
	mean_vec = train_mat.sum(axis=1)/float(num_examps)

	print "finished mean_vec"

	# create covariance matrix
	cov_mat = np.zeros((vec_len, vec_len))
	for examp in xrange(0, num_examps):
		mean_diff = mat[:,examp] - mean_vec
		cov_mat += np.outer(mean_diff, mean_diff)
	#	print examp		
	# normalize the covariance matrix
	cov_mat = np.divide(cov_mat,1.0*num_examps)

	print "finished cov mat"

	# calculate the eigenvals/vecs of cov_mat which is a Hermitian
	eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
	#_ , eigen_vals, eigen_vecs = np.linalg.svd(cov_mat)

	print "got eigenvals / vecs"

	# get the l2_norm
	#l2_norm = np.linalg.norm(eigen_vecs,axis=0)	
	
	# account for possible divide by zero issues
	#l2_norm_safe = np.array([x if x != 0 else 1 for x in l2_norm])

	# normalize the eigenvectors
	#eigen_vecs = np.divide(eigen_vecs,l2_norm_safe)
	
	#print "normalized, now returning"

	mean_vec = np.reshape(mean_vec, (vec_len, 1))

	return ReturnTuple(mean_vec, eigen_vecs, eigen_vals, cov_mat)

def gen_test_projection(k, numTrain, trainImages, testImages):
	train_mat = np.reshape(trainImages, (trainImages.shape[0]*trainImages.shape[1], trainImages.shape[3]))
	test_mat = np.reshape(testImages, (testImages.shape[0]*testImages.shape[1], testImages.shape[3]))
	
	V = gen_eigenvector_matrix(train_mat[:,0:numTrain-1])

	V_row, V_cols = V.eigen_vector_matrix.shape
	start_k, end_k = (V_cols - k), V_cols

	V_red = (V.eigen_vector_matrix[:,start_k:end_k])

	# test projection
	test_projection = np.mat(V_red).transpose()*(test_mat - V.mean_vector)

	# reconstruction
	recon = np.mat(V_red)*test_projection + V.mean_vector

	np.save("reconstruction_" + str(k) + "k_" + str(numTrain) + "t", recon)



def classify(k, numTrain, testStart, testEnd, trainImages, trainlabels, testImages, testLabels):
	train_mat = np.reshape(trainImages, (trainImages.shape[0]*trainImages.shape[1], trainImages.shape[3]))
	test_mat = np.reshape(testImages, (testImages.shape[0]*testImages.shape[1], testImages.shape[3]))
	
	V = gen_eigenvector_matrix(train_mat[:,0:numTrain-1])

	classif_L2 = np.zeros((testEnd - testStart, 2))
	classif_L1 = np.zeros((testEnd - testStart, 2))

	V_row, V_cols = V.eigen_vector_matrix.shape
	start_k, end_k = (V_cols - k), V_cols
	
	
	V_red = (V.eigen_vector_matrix[:,start_k:end_k])

	# project each trainign image into the image space
	train_projection = np.mat(V_red).transpose()*np.mat(train_mat)
	train_projection = train_projection.transpose()

	# project each test image into the image space
	test_projection = np.mat(V_red).transpose()*(test_mat - V.mean_vector)
	

	# the "closest" training image determines the label of the test image
	for i in xrange(testStart,testEnd):
			
			#print test_projection.shape
			#print train_projection.shape
			
			curr_proj = np.array(test_projection[:,i])
			curr_proj = np.reshape(curr_proj, (1, k))
			
			#print "test_proj " + str(curr_proj.shape)
			#print "train_proj " + str(train_projection[:,0].shape)
	
			vals_L2 = distance.cdist(train_projection, curr_proj, metric='euclidean')
			vals_L1 = distance.cdist(train_projection, curr_proj, metric='cityblock')

			min_index_L2 = np.argmin(vals_L2) 
			min_index_L1 = np.argmin(vals_L1)

			classif_L2[i - testStart,0] = trainLabels[:,min_index_L2]
			classif_L1[i - testStart,0] = trainLabels[:,min_index_L1]

			classif_L2[i - testStart,1] = testLabels[:,i]
			classif_L1[i - testStart,1] = testLabels[:,i]

			print "iter test: " + str(i)

	acc_L1 = (1.0*sum([1 if classif_L1[i, 0] == classif_L1[i,1] else 0 for i in xrange(0, len(classif_L1))]))/(1.0 *len(classif_L1) )
	acc_L2 = (1.0*sum([1 if classif_L2[i, 0] == classif_L2[i,1] else 0 for i in xrange(0, len(classif_L2))]))/(1.0 *len(classif_L2) )
		
	return ClassifTuple(classif_L1, classif_L2)


def knn(neighbors, k, numTrain, testStart, testEnd, trainImages, trainlabels, testImages, testLabels):
	train_mat = np.reshape(trainImages, (trainImages.shape[0]*trainImages.shape[1], trainImages.shape[3]))
	test_mat = np.reshape(testImages, (testImages.shape[0]*testImages.shape[1], testImages.shape[3]))
	
	V = gen_eigenvector_matrix(train_mat[:,0:numTrain-1])

	classif_L2 = np.zeros((testEnd - testStart, 2))
	classif_L1 = np.zeros((testEnd - testStart, 2))

	V_row, V_cols = V.eigen_vector_matrix.shape
	start_k, end_k = (V_cols - k), V_cols
	
	
	V_red = (V.eigen_vector_matrix[:,start_k:end_k])
	train_red = (train_mat[start_k:end_k,:]) 
	test_red = (test_mat[start_k:end_k,:])
	mean_red = (V.mean_vector[start_k:end_k])

	print "V_red " + str(V_red.shape)
	print "train_red " + str(train_red.shape)
	print "test_red " + str(test_red.shape)
	print "mean_red " + str(mean_red.shape)

	# project each trainign image into the image space
	# DIMS train projection == (784 * k) dim * (k * numTrain)
	train_projection = np.mat(V_red)*np.mat(train_red)
	train_projection = np.mat(train_projection).transpose()

	# project each test image into the image space
	test_projection = np.mat(V_red)*np.mat((test_red - mean_red))
	

	# the "closest" training image determines the label of the test image
	for i in xrange(testStart,testEnd):
			
			#print test_projection.shape
			#print train_projection.shape
			
			curr_proj = np.array(test_projection[:,i])
			curr_proj = np.reshape(curr_proj, (1,784))
			
			#print "test_proj " + str(curr_proj.shape)
			#print "train_proj " + str(train_projection[:,0].shape)
	
			vals_L2 = distance.cdist(train_projection, curr_proj, metric='euclidean')
			vals_L1 = distance.cdist(train_projection, curr_proj, metric='cityblock')

			rows, cols = vals_L2.shape

			# reshape magic to get argsort to work
			vals_L2 = np.reshape(vals_L2, (cols, rows))
			vals_L1 = np.reshape(vals_L1, (cols, rows))

			sorted_indices_L2 = np.argsort(vals_L2)
			sorted_indices_L1 = np.argsort(vals_L1)

			print "sorted indices type: " + str(type(sorted_indices_L2))
			print "sorted_indices_L2: " + str(sorted_indices_L2[0])


			print "SANITY: " + str(sorted_indices_L2[-1] == np.argmin(vals_L2))

			top_k_L2 = [trainLabels[0,sorted_indices_L2[0][top]] for top in xrange(0,neighbors)]
			top_k_L1 = [trainLabels[0,sorted_indices_L1[0][top]] for top in xrange(0,neighbors)]

			#print "top K " + str(top_k_L2)
			#print "Len " + str(len(top_k_L2[0]))

			freq_L2 = np.array([0,0,0,0,0,0,0,0,0,0])
			for elem in top_k_L2:
				freq_L2[elem] += 1

			freq_L1 = np.array([0,0,0,0,0,0,0,0,0,0])
			for elem in top_k_L1:
				freq_L1[elem] += 1

			maj_L2 = np.argsort(freq_L2)
			maj_L1 = np.argsort(freq_L1)

			#print "maj_L2 " + str(maj_L2)


			classif_L2[i - testStart,0] = maj_L2[len(maj_L2) - 1]
			classif_L1[i - testStart,0] = maj_L1[len(maj_L1) - 1]


			classif_L2[i - testStart,1] = testLabels[:,i]
			classif_L1[i - testStart,1] = testLabels[:,i]

			print "iter test: " + str(i)

	acc_L1 = (1.0*sum([1 if classif_L1[i, 0] == classif_L1[i,1] else 0 for i in xrange(0, len(classif_L1))]))/(1.0 *len(classif_L1) )
	acc_L2 = (1.0*sum([1 if classif_L2[i, 0] == classif_L2[i,1] else 0 for i in xrange(0, len(classif_L2))]))/(1.0 *len(classif_L2) )
		
	return ClassifTuple(classif_L1,classif_L2)
	

if __name__ == '__main__':
	eigen_raw = sio.loadmat('digits.mat')

	trainImages = eigen_raw['trainImages']
	trainLabels = eigen_raw['trainLabels']
	testImages = eigen_raw['testImages']
	testLabels = eigen_raw['testLabels']
	
	# reshape the matrix
	train_mat = np.reshape(trainImages, (trainImages.shape[0]*trainImages.shape[1], trainImages.shape[3]))
	test_mat = np.reshape(testImages, (testImages.shape[0]*testImages.shape[1], testImages.shape[3]))

	#print "train "  + str(trainLabels.shape)
	#print "test " + str(testLabels.shape)
		
	#print train_mat.shape


	#ret = classify(0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)

	#pickle.dump(ret.acc_L1, open("classif_L1_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	#pickle.dump(ret.acc_L2, open("classif_L2_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))

	'''	
	knn2 = knn(2, 0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)

	pickle.dump(knn2.acc_L1, open("classif_L1_KNN2_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(knn2.acc_L2, open("classif_L2_KNN2_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))

	knn5 = knn(5, 0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)

	pickle.dump(knn5.acc_L1, open("classif_L1_KNN5_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(knn5.acc_L2, open("classif_L2_KNN5_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))

	knn10 = knn(10, 0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)

	pickle.dump(knn10.acc_L1, open("classif_L1_KNN10_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(knn10.acc_L2, open("classif_L2_KNN10_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	

	knn20 = knn(20, 0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)

	pickle.dump(knn20.acc_L1, open("classif_L1_KNN20_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(knn20.acc_L2, open("classif_L2_KNN20_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))

	knn50 = knn(50, 0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)

	pickle.dump(knn50.acc_L1, open("classif_L1_KNN50_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(knn50.acc_L2, open("classif_L2_KNN50_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))

	knn100 = knn(100, 0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)

	pickle.dump(knn100.acc_L1, open("classif_L1_KNN100_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(knn100.acc_L2, open("classif_L2_KNN100_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))

	knn200 = knn(200, 0, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)
	pickle.dump(knn200.acc_L1, open("classif_L1_KNN200_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(knn200.acc_L2, open("classif_L2_KNN200_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	

	for k_amm in xrange(100, 800, 100):
		for train_amm in xrange(5000, 60001, 5000):
			red = classify(k_amm, train_amm, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)
			pickle.dump(red.acc_L1, open("classif_L1_red_" + str(k_amm) + "k_" + str(train_amm) + "t_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
			pickle.dump(red.acc_L2, open("classif_L2_red_" + str(k_amm) + "k_" + str(train_amm) + "t_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	'''

	for train_amm in xrange(5000, 60001, 5000):
		red = classify(784, train_amm, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)
		pickle.dump(red.acc_L1, open("classif_L1_full_" + str(train_amm) + "t_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
		pickle.dump(red.acc_L2, open("classif_L2_full_" + str(train_amm) + "t_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))


	'''
	gen_test_projection(784, 60000, trainImages, testImages)
	gen_test_projection(600, 60000, trainImages, testImages)
	gen_test_projection(400, 60000, trainImages, testImages)
	gen_test_projection(200, 60000, trainImages, testImages)
	gen_test_projection(1000, 60000, trainImages, testImages)
	
	red = classify(784, 60000, int(sys.argv[1]), int(sys.argv[2]), trainImages, trainLabels, testImages, testLabels)
	pickle.dump(red.acc_L1, open("classif_L1_red_" + str(k_amm) + "k_" + str(train_amm) + "t_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	pickle.dump(red.acc_L2, open("classif_L2_red_" + str(k_amm) + "k_" + str(train_amm) + "t_" + sys.argv[1] + "_" + sys.argv[2] + ".p", "wb"))
	'''
