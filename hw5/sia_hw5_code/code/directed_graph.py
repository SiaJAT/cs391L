from directed_node import DirectedNode
import math
import operator
import numpy as np
import matplotlib.pyplot as plt

_STOPPING = 0.001

class DirectedGraph:
	def __init__(self, num_nodes, init_dist):
		self.all_nodes = []
		self.construct_graph(num_nodes, init_dist)
	
	# initialize a graph with "num_nodes" number of nodes with
	# no edges, each node having a "dist", where dist is a list
	# of values that sum to 1
	# "dist" semantics == [0.3 0.7]
	# ...index 0 with probability 0.3
	# ...index 1 with probability 0.7
	def construct_graph(self, num_nodes, init_dist):
		for i in xrange(0, num_nodes):
			node = DirectedNode(init_dist, i)		
			self.all_nodes.append(node)

	def print_distribution(self):
		print "["
		for i in xrange(0, len(self.all_nodes)):
			print "(" + str(self.all_nodes[i].distribution) + ") "
		print "]"

	# draw edge from node at "src" index in "all_nodes"  to "dst" node
	def draw_edge(self, src, dst):
		self.all_nodes[src].outgoing.append(self.all_nodes[dst])
		self.all_nodes[src].has_outgoing = True

		self.all_nodes[dst].incoming.append(self.all_nodes[src])
		self.all_nodes[dst].has_incoming = True
		self.init_CPT()

	def init_CPT(self):
		for node in self.all_nodes:
			node.random_CPT()
	
	def set_node_est_CPT(self, node_index, values):
		self.all_nodes[node_index].set_est_CPT(values)

	def set_node_CPT(self, node_index, values):
		self.all_nodes[node_index].set_CPT(values)

	def run_EM(self, samples, max_iter):
		for x in xrange(0, max_iter):
			for node in self.all_nodes:
				node.batch_maximize(samples)
			
	def print_debug(self):
		for node in self.all_nodes:
			print "node: " + str(node.id)
			print "CPT: " + str(node.CPT)
			print "Est CPT: " + str(node.est_CPT)
			print ""


	def expectation(self, index, curr_sample):		
		# create dictionary of sample values
		id2sample = {}
		for var_ind in range(0, len(curr_sample)):	
				id2sample[var_ind] = curr_sample[var_ind]
	
		top = self.numer(index, id2sample)	
		bottom = self.denom(index, id2sample)

		#print "top: " + str(top)
		#print "bottom: " + str(bottom)
	
		numer = math.exp(top)
		denom = math.exp(top) + math.exp(bottom)
			
		return numer/denom


	def numer(self, index, id2sample):
		numer = 0.0
		for node in self.all_nodes:
			# root nodes
			#print "\ncurrently processing: " + str(node.id)
			if node.has_incoming == False:
				if id2sample[node.id] == 1.0:
					#print "root prob: " + str(math.log(node.est_CPT['']))
					numer += math.log(node.est_CPT[''])
				else:
					numer += math.log(1.0 - node.est_CPT[''])
				
			# non-root nodes
			if node.has_incoming == True:
				#print type(node.incoming)
				#print str(node.incoming)
				sorted_in = sorted(node.incoming, key = lambda x : x.id)
				
				# get CPT of this node
				ancestor_string = ""
				for curr_incoming in sorted_in:
					# this parent node is beign estimated, so assume the positive case
					if curr_incoming.id == index:
						curr_val = 1	
					else:
						curr_val = id2sample[curr_incoming.id]
					ancestor_string += str(curr_val)
				
				# if we're estimating this node, assume postiive class
				# also if it's actually 1.0 in the sample set...
				#print "ancestors: " + ancestor_string
				'''
				FUDGE FACTOR 2
				'''
				if node.est_CPT[ancestor_string] == 0:
					return 0
				
				
				if index == node.id or id2sample[node.id] == 1.0:	
					numer += math.log(node.est_CPT[ancestor_string])
				else:
					#print str(node.est_CPT[ancestor_string])
					if 1.0 - node.est_CPT[ancestor_string] == 0:
						return 0
					
					numer += math.log(1.0 - node.est_CPT[ancestor_string])
				
		#print "numer: " + str(numer)
		return numer	
		
	def denom(self, index, id2sample):
		denom = 0.0
		for node in self.all_nodes:
			# root nodes
			#print "\ncurrently processing: " + str(node.id)
			if node.has_incoming == False:
				if id2sample[node.id] == 1.0:
					#print "root prob: " + str(math.log(node.est_CPT['']))
					denom += math.log(node.est_CPT[''])
				else:
					denom += math.log(1.0 - node.est_CPT[''])
				
			# non-root nodes
			if node.has_incoming == True:
				#print type(node.incoming)
				#print str(node.incoming)
				sorted_in = sorted(node.incoming, key = lambda x : x.id)
				
				# get CPT of this node
				ancestor_string = ""
				for curr_incoming in sorted_in:
					# this parent node is beign estimated, so assume the positive case
					if curr_incoming.id == index:
						curr_val = 0
					else:
						curr_val = id2sample[curr_incoming.id]
					ancestor_string += str(curr_val)
				
				# if we're estimating this node's denominator, assume negative class
				# also if it's actually 1.0 in the sample set...
				#print "ancestors: " + ancestor_string
				'''
				FUDGE FACTOR 2
				'''
				if 1.0 - node.est_CPT[ancestor_string] == 0:
					denom += 1
				else:	
				
					if index == node.id:	
						denom += math.log(1.0 - node.est_CPT[ancestor_string]) 
					elif id2sample[node.id] == 1.0:	
						denom += math.log(node.est_CPT[ancestor_string])
					else:
						# FUDGE FACTOR!!!!!!!!!! for case where den = 0
						if 1.0 - node.est_CPT[ancestor_string] == 0:
							return 1.0
						#print str(node.est_CPT[ancestor_string])
						denom += math.log(1.0 - node.est_CPT[ancestor_string])
				
		#print "denom " + str(denom)
		return denom
	

	def check_convergence(self, old_vals, new_vals, stopping):
		diffs = abs(np.array(new_vals) - np.array(old_vals))
		#print "old: " + str(old_vals)
		#print "new: " + str(new_vals)
		#print str(diffs)
		for d in diffs:
			if d > stopping:
				return False
		
		return True


	# main call to EM
	def EM(self, samples, index, max_iters):
		#print "> MAXIMIZING index node: " + str(index)
		#print "SAMPLES: " + str(samples)
		
		old_vals = [val for key, val in self.all_nodes[index].est_CPT.iteritems()]
			
		#print "old: " + str(old_vals)
		for i in xrange(0,max_iters):
			#print ">> Iteration: " + str(i)
			# stopping conditions
			
			for ancestor, val in self.all_nodes[index].est_CPT.iteritems():
				'''	
				#print "ancestor: " + ancestor + ", val: " + str(val)
				if i > 0 and val == 1.0 or val == 0.0:
					self.print_debug()
					print "EM has converged on something at iteration " + str(i)
					return i
				'''
			#self.batch_maximize(samples, index)	
			new_vals = [val for key, val in self.all_nodes[index].est_CPT.iteritems()]

			has_converged_normally = self.check_convergence(old_vals, new_vals, _STOPPING)

			if has_converged_normally and i > 0:
				self.print_debug()
				print "EM has NORMALLY converged on something at iteration " + str(i)
				return i
			
			old_vals = new_vals
			#print "pre max"
			#self.print_debug()
			self.batch_maximize(samples, index) 
			#print "post max"
			#self.print_debug()
		#self.print_debug()
		return max_iters

	# maximize the expectation for a var given a2d array of samples, where 
 	# first dim is # of samples, second smaple is # of variables 
	def batch_expectation(self, samples, index):
		num_samples, num_vars = samples.shape
		est_probs = []
		for samp_ind in range(num_samples):
			val = self.expectation(index, samples[samp_ind])
			est_probs.append(round(val, 4))
		return est_probs
	

	# call this function when you want to do EM for this node
	# (assumes this node is hidden)
	def batch_maximize(self, samples, index):
		# "samples" sample table semantics (numpy array)
		# all nodes in columns before self.id are potential parents
		# all nodes in columsn after self.id are potential children
	
		# node with which we're maximizing
		node = self.all_nodes[index]

		# maximize on root?
		if self.all_nodes[index].has_incoming == False:
			self.all_nodes[index].est_CPT[""] = np.random.random

		# compute frequency counts 
		ancestors = self.get_sample_freq_counts(samples, index)				
		
		#print "!!!!!!!!"
		#print str(ancestors)

		# get the expectations (E step)
		estimates = self.batch_expectation(samples, index)
		#print str(estimates)		
		
		#for k,v in node.string2count.iteritems():
		#	print "string2count " + str(k) + ", " + str(v)
		
		
		# go through each possible parent combination
		
		
		for comb in node.combos:
			#print "comb: " + str(comb)
			if comb in node.string2count:
				count, pos_list = node.string2count[comb]
			
				numerator = 0
				for ind in pos_list:
					numerator += estimates[ind]

				denominator = count
				#print "comb : " + str(comb) + ", num: " + str(numerator) +  ", den: " + str(denominator)

				node.est_CPT[comb] = numerator/denominator  
			else:
				node.est_CPT[comb] = 0
					


	def get_sample_freq_counts(self, samples, index):
		# list of ancestors for which i need freq counts
		#list_ancestors = [node.id for node in self.all_nodes[index].incoming]
		

		#return list_ancestors	
		
		#current node which we're maximizing
		node = self.all_nodes[index]
		
		# list of tuples of (binary strings (values of samples), indices of occurence)
		parent_strings = []
		# compute frequency counts and indices of the occurences 
		num_samps, num_vars = samples.shape	
		for samp in xrange(0, num_samps):
			curr_string = ""

			'''
			FUCKING CHANGE: GET THE FIRST OCCURRENCE of THE PARENT
			ASSUMPTION: all_nodes are sorted by id from root of DAG
			'''
			sorted_in = sorted(node.incoming, key = lambda x : x.id)
			first_parent_index = sorted_in[0].id

			for var in xrange(first_parent_index, node.id):
				curr_string += str(int(samples[samp, var]))
			parent_strings.append((curr_string, samp))
	
		#print "parent strings: " + str(parent_strings)
		# dictionary entry
		# sample string ---> ( count of occurrence of sample string, list of positions)
		for tuple in parent_strings:
			
			var_string = tuple[0]
			pos = tuple[1]

			if var_string not in node.string2count:
				node.string2count[var_string] = (1, [pos])
			else:
				old_tuple = node.string2count[var_string] 
				
				count = old_tuple[0] + 1
				pos_list = old_tuple[1]
				pos_list.append(pos)

				node.string2count[var_string] = (count, pos_list)

	def get_node(self, node_id):
		for node in self.incoming:
			if node.id == node_id:
				return node
		return None

	
