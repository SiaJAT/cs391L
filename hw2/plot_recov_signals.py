import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from os import path
import pickle

def scale(arr):
    max_val = max(arr)
    min_val = min(arr)
    return [((2.0*(num - min_val))/(1.0*(max_val - min_val)))-1.0 for num in arr]

if __name__ == '__main__':
    p_files = [f for f in os.listdir(os.getcwd()) if "recon" in f 
	and ".p" in f 
	and ".pdf" not in f  
	and ".py " not in f]

    print str(p_files)

    freq_full = range(0,44000)
    freq_red = range(0,40)

    for f in p_files:
            pp_full = PdfPages(f + '_sounds_full.pdf')
            pp_red = PdfPages(f + '_sounds_40.pdf')

            sounds = pickle.load(open(f, 'rb'))
            sounds = np.array(sounds)
            print sounds.shape
            print sounds[1,:].shape
            
            for i in xrange(0, 5):
                    curr = scale(sounds[i,:])
                    plt.plot(freq_full, curr)
                    pp_full.savefig()	
                    plt.clf()
                    plt.plot(freq_red, curr[0:40])
                    pp_red.savefig()
                    plt.clf()

            pp_full.close()
            pp_red.close()


