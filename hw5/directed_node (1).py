import numpy as np
import itertools
import math
import random

class DirectedNode:
	def __init__(self, distribution, id):
		# assert distriution is valid (i.e., sums to 1)
		assert sum(distribution) == 1 and len(distribution) == 2
		
		self.id = id
		self.distribution = distribution
		self.has_incoming = False
		self.has_outgoing = False
		self.incoming = []
		self.outgoing = []
		self.combos = []
		self.CPT = {}
		
	# calculate the CPT probabilities, avoiding underflow 
	# CPT is a dictionary from binary strings to probabilities
	# semantics: n length string with n-1 parents and this node itself at n
	# e.g., 001
	# ...2 parents, parent_1 = 0, parent_2 = 0, this DirectedNode = 1
	def build_CPT(self):
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
	
	def maximize(self, samples, CPT):
		# "samples" sample table semantics (numpy array)
		# all nodes in columns before self.id are potential parents
		# all nodes in columsn after self.id are potential children
		num_samps, num_vars = samples.shape
		for samp in xrange(0, num_samps):
			for var in xrange(0, num_vars):

		

	def add_outgoing(self, node):
		self.has_outgoing = True
		self.outgoing.append(node)

	def add_incoming(self, node):
		self.has_incoming = True
		self.incoming.append(node)
	
	def gen_combos(self, num_vars):
		combos = []
		for digit in itertools.product("01", repeat=num_vars):
				combos.append("".join(digit))
		return combos

