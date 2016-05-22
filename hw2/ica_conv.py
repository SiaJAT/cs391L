import sys
import pickle
import numpy as np
import scipy.io as sio
from scipy.spatial import distance
from scipy.special import expit

# generate a random matrix of [m, n] definitions
def gen_ran_mix_matrix(m, n):
        random_weightings = []
        for i in xrange(0, m):
            random_weightings.append(np.random.dirichlet(np.ones(n)))
        test = np.mat(np.vstack(random_weightings))
        print test
        return test


# INPUT:
# X, matrix of mixed signals
# n, number of source signals
# m, number of mixtures
# t, number of timesteps
# learn_rate, learning rate
# R_max, max number of learning interations
#
# OUTPUT: W, the recovered original n source signals
def ICA(X, n, m, t, learn_rate, R_max):
	# init weight Matrix
	W = np.random.randint(0,10,(n,m))/100.0 
        W_old = None
        
        counter = 0
	old_dist = float('inf')
        while (counter == 0 or abs(dist - old_dist) > R_max):
		# calculate current estimate of source signals
		Y = W*X

		# calculate Z matrix by vectorizing and applying sigmoid expit function to Y 
		vect_expit = np.vectorize(expit)
		Z = vect_expit(Y)

		# [n,T] * [T, n]
		delta_W = learn_rate*(np.matrix(np.identity(n)) + (1 - 2*Z)*Y.transpose())*W

		# update weights
	    	W_old = W
                W += delta_W

                #print W_old
                #print W
                
                W_old_reshape = 100*np.array(W_old)
                W_old_reshape = 100*np.reshape(W_old, (W_old.shape[0]*W_old.shape[1]))

                W_reshape =np.array(W)
                W_reshape = np.reshape(W_reshape, (W.shape[0]*W.shape[1]))
                
                
                if counter == 0:
                    old_dist = float("inf")
                    dist = distance.euclidean(W_old_reshape, W_reshape)
                else:
                    old_dist = dist
                    dist = distance.euclidean(W_old_reshape, W_reshape)

                print "iteration: " + str(counter) + ", learning at rate " + str(learn_rate) + ", m=" + str(m) + ", dist=" + str(dist) 

	        counter += 1


	return W

if __name__ == '__main__':
	raw_sounds = sio.loadmat('sounds.mat')
	sounds = raw_sounds['sounds']

	# (U) sounds is a [n, t] matrix, where n = number of source signals
	# (A) convmat is a [m, n] matrix, where m = number of mixtures (m>=n)
	# (X) mixed is a [m, t] matrix, where m = number of mixtures
	U = np.mat(sounds)
	n,t = sounds.shape	
	
	m = int(sys.argv[3])
	A = gen_ran_mix_matrix(m, n)

	X = A*U
  	print "mixed shape: " + str(X.shape)

  	learn_rate = float(sys.argv[1])/100.0
  	R_max = float(sys.argv[2])
	W = ICA(X, n, m, t, learn_rate, R_max)
    
	learn_str = ''
 	if float(sys.argv[1]) % 1 == 0:
		learn_str = str(learn_rate) + 'p0bp'
	else:
		print str(float(sys.argv[1]) % 1).split('.')[1] 
		learn_str = str(int(learn_rate/1)) + 'p' + str(float(sys.argv[1]) % 1).split('.')[1] + 'bp'
	
	recon = W*X

	pickle.dump(recon, open('recon_' + learn_str + '_' + str(R_max) + 'r' + str(m) + 'm.p', 'wb'))
        print W.shape
	
