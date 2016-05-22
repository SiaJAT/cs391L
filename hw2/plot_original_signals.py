import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

raw = sio.loadmat('sounds.mat')

sounds = raw['sounds']
freq_full = range(0,44000)
freq_red = range(0,40)

pp_full = PdfPages('original_sounds_full.pdf')
pp_red = PdfPages('original_sounds_40.pdf')

for i in xrange(0, 5):
	plt.plot(freq_full, sounds[i,:])
	pp_full.savefig()	
	plt.clf()
	plt.plot(freq_red, sounds[i,0:40])
	pp_red.savefig()
	plt.clf()

pp_full.close()
pp_red.close()


