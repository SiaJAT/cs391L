import matplotlib.pyplot as plt
from directed_node import DirectedNode
from directed_graph import DirectedGraph
import numpy as np
import random

def EM_run_0():
	print "BFS test"
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.all_nodes[1].count_all_predecessor()

'''
Experiments 1-3 test the sensitivity of initial estimates of the unknown variable
'''
def EM_run_1():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 1: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])

	graph.print_debug()

	samples = np.array([[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[1, 0, 0, 1, 1],
											[0, 0, 0, 0, 1],
											[0, 1, 0, 1, 0],
											[0, 0, 0, 0, 1],
											[1, 1, 0, 1, 1],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[0, 0, 0, 0, 1]])

	graph.EM(samples, 2, 1000)
	
def EM_run_2():
	
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 2: [tt, tf, ft, ff] = [0.1, 0.1, 0.1, 0.1]"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.1, 0.1, 0.1, 0.1])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])

	samples = np.array([[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[1, 0, 0, 1, 1],
											[0, 0, 0, 0, 1],
											[0, 1, 0, 1, 0],
											[0, 0, 0, 0, 1],
											[1, 1, 0, 1, 1],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[0, 0, 0, 0, 1]])

	graph.EM(samples, 2, 20)

def EM_run_3():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 3: [tt, tf, ft, ff] = [0.9, 0.9, 0.9, 0.9]"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"	
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.9, 0.9, 0.9, 0.9])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])

	samples = np.array([[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[1, 0, 0, 1, 1],
											[0, 0, 0, 0, 1],
											[0, 1, 0, 1, 0],
											[0, 0, 0, 0, 1],
											[1, 1, 0, 1, 1],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[0, 0, 0, 0, 1]])

	graph.EM(samples, 2, 100)

'''
Experiments 4-6 test how the goodness of the data affects the estimates
'''
def EM_run_4():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 4: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	print "GARBAGE DATA - ALL 0s"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
		
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])

	samples = np.array([[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0],
											[0, 0, 0, 0, 0]])

	graph.EM(samples, 2, 10)

def EM_run_5():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 5: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	print "GARBAGE DATA - ALL 1s"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])

	samples = np.array([[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1],
											[1, 1, 1, 1, 1]])
	graph.EM(samples, 2, 100)

def EM_run_6a():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 6: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	print "RANDOM DATA"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	
	iters = 0
	simulations = 1000
	runs = []

	for i in range(simulations):
		graph = DirectedGraph(5, [0.5 ,0.5])
		graph.draw_edge(0, 2)
		graph.draw_edge(1, 2)
		graph.draw_edge(2, 3)
		graph.draw_edge(2, 4)
		
		graph.init_CPT()
		graph.set_node_est_CPT(0, [0.1])
		graph.set_node_est_CPT(1, [0.2])
		graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
		graph.set_node_est_CPT(3, [0.2, 0.9])
		graph.set_node_est_CPT(4, [0.1, 0.8])

		samples = []
		for i in range(10):
			curr_sample = []
			for j in range(5):
				curr_sample.append(round(random.random(), 0))
			samples.append(curr_sample)
		
		samples = np.array(samples, dtype=np.int64)
		
		#print "RANDOM DATA"
		#print str(samples) 	

		ret = graph.EM(samples, 2, 1000)
		iters += ret
		runs.append(ret)
	
	print "Took approximately: " + str(iters/simulations)
	plt.figure()
	plt.hist(runs, bins=50)
	plt.title('Distribution of Iterations Until Converged')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Frequency')
	plt.savefig('dist_6.png')

def EM_run_6b():
	
	#print "\n Experiment run 6: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	#print "RANDOM DATA"
	
	iters = 0
	simulations = 10000
	

	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])

	samples = []
	for i in range(100):
		curr_sample = []
		for j in range(5):
			curr_sample.append(round(random.random(), 0))
		samples.append(curr_sample)
	
	samples = np.array(samples, dtype=np.int64)
	
	#print "RANDOM DATA"
	#print str(samples) 

	return graph.EM(samples, 2, 10000)

def EM_run_7():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 7: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	print "Recover leaf"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
	#graph.set_node_est_CPT(2, [0.1, 0.1, 0.2, 0.2])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])
	
	
	
	'''
	samples = []
	for i in range(100):
		curr_sample = []
		for j in range(5):
			curr_sample.append(round(random.random(), 0))
		samples.append(curr_sample)
	
	samples = np.array(samples, dtype=np.int64)
	'''
	samples = np.array([[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[1, 0, 0, 1, 1],
											[0, 0, 0, 0, 1],
											[0, 1, 1, 1, 0],
											[0, 0, 0, 0, 1],
											[1, 1, 0, 1, 1],
											[0, 0, 0, 0, 0],
											[0, 0, 1, 1, 0],
											[0, 0, 0, 0, 1]])
	#graph.EM(samples, 2, 10000)
	graph.EM(samples, 3, 10000)


def EM_run_8():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 8: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	print "Multiple recovery"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	graph = DirectedGraph(5, [0.5 ,0.5])
	graph.draw_edge(0, 2)
	graph.draw_edge(1, 2)
	graph.draw_edge(2, 3)
	graph.draw_edge(2, 4)
	
	graph.init_CPT()
	graph.set_node_est_CPT(0, [0.1])
	graph.set_node_est_CPT(1, [0.2])
	graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
	#graph.set_node_est_CPT(2, [0.1, 0.1, 0.2, 0.2])
	graph.set_node_est_CPT(3, [0.2, 0.9])
	graph.set_node_est_CPT(4, [0.1, 0.8])
	
	
	
	'''
	samples = []
	for i in range(100):
		curr_sample = []
		for j in range(5):
			curr_sample.append(round(random.random(), 0))
		samples.append(curr_sample)
	
	samples = np.array(samples, dtype=np.int64)
	'''
	samples = np.array([[0, 0, 0, 0, 0],
											[0, 0, 0, 1, 0],
											[1, 0, 0, 1, 1],
											[0, 0, 0, 0, 1],
											[0, 1, 1, 1, 0],
											[0, 0, 0, 0, 1],
											[1, 1, 0, 1, 1],
											[0, 0, 0, 0, 0],
											[0, 0, 1, 1, 0],
											[0, 0, 0, 0, 1]])
	graph.EM(samples, 2, 10000)
	graph.EM(samples, 3, 10000)

def EM_run_9():
	print "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	print "Experiment run 9: [tt, tf, ft, ff] = [0.9, 0.6, 0.3, 0.2]"
	print "RANDOM DATA - Varying Sample size (double)"
	print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	
	iters = 0
	simulations = 1000
	runs = []

	for i in range(simulations):
		graph = DirectedGraph(5, [0.5 ,0.5])
		graph.draw_edge(0, 2)
		graph.draw_edge(1, 2)
		graph.draw_edge(2, 3)
		graph.draw_edge(2, 4)
		
		graph.init_CPT()
		graph.set_node_est_CPT(0, [0.1])
		graph.set_node_est_CPT(1, [0.2])
		graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
		graph.set_node_est_CPT(3, [0.2, 0.9])
		graph.set_node_est_CPT(4, [0.1, 0.8])

		samples = []
		for i in range(100):
			curr_sample = []
			for j in range(5):
				curr_sample.append(round(random.random(), 0))
			samples.append(curr_sample)
		
		samples = np.array(samples, dtype=np.int64)
		
		#print "RANDOM DATA"
		#print str(samples) 	

		ret = graph.EM(samples, 2, 1000)
		iters += ret
		runs.append(ret)
	
	print "Took approximately: " + str(iters/simulations)
	plt.figure()
	plt.hist(runs, bins=50)
	plt.title('Distribution of Iterations Until Converged')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Frequency')
	plt.savefig('dist_9.png')
	print "Took approximately: " + str(iters/simulations)


if __name__ == "__main__":
	#EM_run_0()
	'''
	EM_run_1()
	EM_run_2()
	EM_run_3()
	EM_run_4()
	EM_run_5()
	EM_run_6a()	
	'''
	EM_run_7()
	EM_run_8()
	'''
	EM_run_9()
	'''
