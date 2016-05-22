import numpy as np
import itertools
import math
import random
import numpy as np

class DirectedNode:
	def __init__(self, distribution, id):
		# assert distriution is valid (i.e., sums to 1)
		assert sum(distribution) == 1 and len(distribution) == 2
		
		# smaller id == closer to the "root" of the DAG
		self.id = id

		# distribution for binary nodes (e.g, [0.5, 0.5]) 
		# if the node is a root
		self.root_distribution = distribution
		self.est_distribution = []

		# flags for incoming / outgoing nodes
		self.has_incoming = False
		self.has_outgoing = False
		
		# lists of DirectedNodes representing parent/child relationships
		self.incoming = []
		self.outgoing = []

		# all possible combinations of parent values before this node
		self.combos = []
		self.all_predecessor_count = 0

		# ids of the incoming nodes
		self.incoming_ids = []
		self.outgoing_ids = []

		# the true CPT
		self.CPT = {}

		# the estimated CPT
		self.est_CPT = {}

		# number of times a given sample (or subsample) occurs in the dataset
		self.string2count = {}

		# initialize the CPTS to random values
		#self.random_CPT()

		# global knowledge
		
	
	def count_all_predecessor(self):
		count = 0
		queue = []
		for node in self.incoming:
			queue.append(node)

		while len(queue) > 0:
			curr = queue[0]
			for curr_parent in curr.incoming:
				queue.append(curr_parent)
			count += 1
			queue.pop(0)
		
		return count

	# calculate the CPT probabilities, avoiding underflow 
	# CPT is a dictionary from binary strings to probabilities
	# semantics: n length string with n-1 parents and this node itself at n
	# e.g., 001
	# ...2 parents, parent_1 = 0, parent_2 = 0, this DirectedNode = 1
	def random_CPT(self):
		# flush the old CPTs
		self.CPT = {}
		self.est_CPT = {}
		
		'''
		FUCKING WARNING: THIS ASSUMES YOU INITIALIZED THE GRAPH
		'''
		self.all_predecessor_count = self.count_all_predecessor()

		#self.combos = self.gen_combos(self.all_predecessor_count)	
		self.combos = self.gen_combos(len(self.incoming))
		'''
		numerator, denominator = 0.0, 0.0
		for combination in self.combos:
			for var in combination[:-1]:
				numerator += math.log(self.incoming[combination.index(var)].distribution[int(var)])
				denominator += math.log(self.incoming[combination.index(var)].distribution[int(var)])
			numerator += math.log(self.distribution[int(combination[-1])])
			self.CPT[combination] = math.exp(numerator - denominator)
			numerator, denominator = 0.0, 0.0
		'''
		for combination in self.combos:
			self.CPT[combination] = round(random.random(), 2) 
			self.est_CPT[combination] = round(random.random(), 2)
	
	def set_est_CPT(self, values):
		if self.has_incoming == True:
			#print str(self.combos)
			#assert len(values) == len(self.combos)
			index = 0
			
			'''
			change of self.combos to filtered in loop
			'''
			filtered_combos = set([combo[-len(self.incoming):] for combo in self.combos])
			
			#print "filtered: " + str(filtered_combos)
			for combination in self.combos:
				self.est_CPT[combination] = values[index]
				index += 1
		if self.has_incoming == False:	
			self.est_CPT[''] = values[0]
	
	def set_CPT(self, values):
		if self.has_incoming == True:
			assert len(values) == len(self.combos)
			index = 0
			for combination in self.combos:
				self.CPT[combination] = values[index]
				index += 1
		if self.has_incoming == False:	
			self.CPT[''] = values[0]

	def expectation(self, index, samples):
		# calculate the expectation for this node
		curr_sample = samples[index, :]
		
		# create dictionary of sample values
		id2sample = {}
		for var_ind in range(0, len(curr_sample)):	
				id2sample[var_ind] = curr_sample[var_ind]
	
		numer, denom = 0.0, 0.0
		previous_samples = ""	

		# root nodes
		if self.has_incoming == False:
			if id2sample[var_ind] == 1.0:
				numer += math.log(self.est_CPT[''])
			else:
				numer += math.log(1.0 - self.est_CPT[''])

		ancestor_string = ""
		# compute probabilityincoming nodes 
		for curr_incoming in self.incoming:
			curr_val = id2samples[curr_incoming.id]
			ancestor_string += str(curr_val)

		numer += math.log(self.est_CPT[ancestor_string])
				
		'''
		# column index in samples corresponds to variable id
		parent_string = ""
		numerator, denominator = 0.0 ,0.0
		for var in xrange(0, self.id):
			# curr_val of var index variable
			curr_val = curr_sample[var]
			
			# get the probability values of the parents 
			# calculate the numerator values for the parents
			if self.incoming[var].id in self.incoming_ids:
				if curr_val == 0:
					numerator += math.log(self.incoming[var].est_CPT[str(var)])
			
			# keep a record of the previously observed parent values
			# front 
			parent_string += str(int(curr_sample[var]))


		for var in xrange(self.id + 1, len(curr_sample)):
			# do the same as previous loop but for children
			# back
			curr_val = curr_sample[var]
			numerator += math.log(self.outgoing[var].est_CPT[str(var)])
		
		# 2(front + back)
		denominator = 2*numerator 
		
		# add the probability of this node given the values of the parents 
		numerator += math.log(self.CPT[parent_string])
		denominator += math.log(self.CPT[parent_string]) + math.log(1.0 - self.CPT[parent_string])	
		
		# set the estiamted values 
		self.est_distribution[0] = math.exp(numerator - denominator)
		self.est_distribution[1] = 1.0 - math.exp(numerator - denominator) 

		return self.est_distribution[0]
		'''
	

	# maximize the expectation for a 2d array of samples, where 
 	# first dim is # of samples, second smaple is # of variables 
	def batch_expectation(self, samples):
		num_samples, num_vars = samples.shape
		est_probs = []
		for samp in xrange(0, num_samples):
			est_probs.append(self.expectation(samp, samples))
		return est_probs


	# call this function when you want to do EM for this node
	# (assumes this node is hidden)
	def batch_maximize(self, samples):
		# "samples" sample table semantics (numpy array)
		# all nodes in columns before self.id are potential parents
		# all nodes in columsn after self.id are potential children
		
		if self.has_incoming == False:
			self.est_CPT[""] = np.random.random

		# compute frequency counts 
		self.get_sample_freq_counts(samples)				
		
		# get the expectations (E step)
		estimates = self.batch_expectation(samples)
		
		# go through each possible parent combination
		for comb in self.combos:
			count, pos_list = self.string2count[comb]
			
			numerator = 0
			for ind in pos_list:
				numerator += estimates[ind]

			denominator = count

			self.est_CPT[comb] = numerator/denominator  
		
					


	def get_sample_freq_counts(self, samples):
		# list of tuples of binary strings (values of samples) and indices of occurence
		parent_strings = []
		# compute frequency counts and indices of the occurences 
		num_samps, num_vars = samples.shape	
		for samp in xrange(0, num_samps):
			curr_string = ""
			for var in xrange(0, self.id):
				if var in self.incoming_ids:
					curr_string += str(int(samples[samp, var]))
			parent_strings.append((curr_string, samp))
		
		# dictionary entry
		# sample string ---> ( count of occurrence of sample string, list of positions)
		for tuple in parent_strings:
			
			var_string = tuple[0]
			pos = tuple[1]

			if var_string not in self.string2count:
				self.string2count[var_string] = (1, [pos])
			else:
				old_tuple = self.string2count[var_string] 
				
				count = old_tuple[0] + 1
				pos_list = old_tuple[1]
				pos_list.append(var_string)

				self.string2count[var_string] = (count, pos_list)
				
	def get_node(self, node_id):
		for node in self.incoming:
			if node.id == node_id:
				return node
		return None


	def add_outgoing(self, node):
		self.has_outgoing = True
		self.outgoing.append(node)
		self.outgoing_ids.append(node.id)

	def add_incoming(self, node):
		self.has_incoming = True
		self.incoming.append(node)
		self.incoming_ids.append(node.id)
	
	def gen_combos(self, num_vars):
		combos = []
		for digit in itertools.product("01", repeat=num_vars):
				combos.append("".join(digit))
		return combos

