import pickle
import os
import re
import numpy as np
import sys


def evaluate(classif, name):
	acc = (1.0*sum([1 if classif[i][0] == classif[i][1] else 0 for i in xrange(0, len(classif))]))/(1.0 *len(classif))
	
	print name + "," + str(acc)

def do_IO(which_norm):
	pickles = [f for f in os.listdir(os.getcwd()) if "classif_" + which_norm in f]

	master_list = []
	for f in pickles:
		curr_eval = pickle.load(open(f, "rb"))
		master_list.append(curr_eval)

	flat_master = [eval_pair for small in master_list for eval_pair in small]

	return flat_master 


if __name__ == '__main__':
	norm = sys.argv[1]
	 
        if int(sys.argv[2]) == 1:
		for k_amm in xrange(100, 800, 100):
			for train_amm in xrange(5000, 60001, 5000):
				red = do_IO(norm + "_red_" + str(k_amm) + "k_" + str(train_amm) + "t")
				evaluate(red, str(k_amm) + "," + str(train_amm))

		for train_amm in xrange(5000, 60001, 5000):
			full = do_IO(norm + "_full_" + str(train_amm) + "t")
			evaluate(full, "784," + str(train_amm))
	else:
		for k in [2,5,10,20,50,100,200]:
			clusters = do_IO(norm + "_KNN" + str(k) + "_")
			evaluate(clusters, str(k))


		
