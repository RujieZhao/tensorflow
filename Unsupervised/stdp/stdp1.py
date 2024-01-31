
import numpy as np 
import matplotlib.cm as cmap
import time
import os.path
import scipy
import pickle
#import brain_no_units
import brain as b
from struct import unpack
from brain import *


MNIST_data_path = './MNIST_data/'

def get_labeled_data(picklename,bTrain = True):

	if os.path.isfile('%s.pickle' % picklename):
		data = pickle.load(open('%s.pickle' % picklename))
	else:
		if bTrain:
			images = open(MNIST_data_path+'train-images.idx3-ubyte','rb')
			labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
		else:
			images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
    images.read(4)
    number_of_images = unpack('>I',images.read(4))[0]
    rows = unpack('>I',images.read(4))[0]
    cols = unpack('>I',images.read(4))[0]
    labels.read(4)
    N = unpack('>I',labels.read(4))[0]

    if number_of_images != N:
    	raise Exception('number of labels did not match the number of images')

    x = np.zeros((N,rows,cols),dtype = np.unit8)
    y = np.zeros((N,1),dtype=np.unit8)
    for i in range(N):
    	if i%1000 = 0:
    		print('i:%i' % i )
    	x[i] = [[unpack('>B',images.read(1))[0] for unused_col in range(cols)] for unused_row in range(rows)]
    	y[i] = unpack('>B',labels.read(1))[0]

    data = {'x':x, 'y':y, 'rows':rows, 'cols':cols}
    pickle.dump(data,open('%s.pickle'%picklename,'wb'))
return data

def get_matrix_from_file(fileName):
	offset = len(ending) +4
	if fileName[-4-offset] == 'X':
		n_src = n_input
	else:
		if fileName[-3-offset]=='e':
			n_src = n_e 
		else:
			n_src = n_i
	if fileName[-1-offset]=='e':
		n_tgt = n_e 
	else:
		n_tgt = n_i 
	readout = np.load(fileName)
	print (readout.shape, fileName)
	value_arr = np.zeros((n_src,n_tgt))
	if not readout.shape ==(0,):
		value_arr[np.int32(readout[:,0]),np.int32(readout[:,1])] = readout[:,2]
return value_arr























