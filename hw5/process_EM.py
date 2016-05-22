import matplotlib.pyplot as plt
import numpy as np


sum = 0
lines = 0
nums = []

with open('random.txt','r') as read_file:
	for line in read_file:
		entries = line.split('iteration ')
		val = int(entries[1])
		sum += val
		lines += 1
		
		nums.append(val)
		
		

	print str(sum)
	print str(sum/lines)
	
	plt.figure()
	plt.hist(nums)
	plt.title("Distribution of Iterations Until Convergence")
	plt.xlabel("Simulation")
	plt.ylabel("Num Iterations")
	plt.savefig('dist.png')

