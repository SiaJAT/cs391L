from directed_node import DirectedNode
from directed_graph import DirectedGraph
import numpy as np
'''
a = DirectedNode([0.5, 0.5])
b = DirectedNode([0.5, 0.5])
c = DirectedNode([0.7, 0.3])

c.add_incoming(a)
c.add_incoming(b)

c.build_samples()

print str(c.samples)
'''
'''
a = DirectedNode([0.5,0.5], 2)
samples = np.zeros((4,4))
samples[0, 0] = 0
samples[0, 1] = 1
samples[0, 2] = 1
samples[0, 3] = 0

samples[1, 0] = 1
samples[1, 1] = 1
samples[1, 2] = 1
samples[1, 3] = 0

samples[2, 0] = 0
samples[2, 1] = 0
samples[2, 2] = 1
samples[2, 3] = 1

samples[3, 0] = 0
samples[3, 1] = 0
samples[3, 2] = 0
samples[3, 3] = 1

a.maximize(samples, 0)
'''

graph = DirectedGraph(5, [0.5 ,0.5])
graph.draw_edge(0, 2)
graph.draw_edge(1, 2)
graph.draw_edge(2, 3)
graph.draw_edge(2, 4)

print "0: " + str(graph.all_nodes[0].has_incoming)
print "1: " + str(graph.all_nodes[1].has_incoming)
print "2: " + str(graph.all_nodes[2].has_incoming)
print "3: " + str(graph.all_nodes[3].has_incoming)
print "4: " + str(graph.all_nodes[4].has_incoming)


graph.set_node_est_CPT(0, [0.1])
graph.set_node_est_CPT(1, [0.2])
#graph.set_node_est_CPT(2, [0.2, 0.3, 0.6, 0.9])
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

#graph.print_debug()

print samples[0,]

#print graph.expectation(2, samples[0,])
#graph.run_EM(samples, 1)
#ret = graph.batch_expectation(samples, 2)
#print str(ret)

graph.EM(samples, 2, 5)
#graph.print_debug()
