import numpy as np
import pylab
import simulation_parameters
import sys
from scipy.optimize import leastsq
import os

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

n_bins = 20
bin_range = (0, 0.011)

#1
fn1 = sys.argv[1]
conn_list1 = np.loadtxt(fn1)
w = conn_list1[:, 2]
count1, bins1 = np.histogram(w, bins=n_bins, range=bin_range)
bin_width = .5 * (bins1[1] - bins1[0])
print 'debug1', count1, bins1

#2
fn2 = sys.argv[2]
conn_list2 = np.loadtxt(fn2)
w = conn_list2[:, 2]
count2, bins2 = np.histogram(w, bins=n_bins, range=bin_range)
print 'debug2', count2, bins2
bins2 += bin_width


fig = pylab.figure()
ax = fig.add_subplot(111)
ax.bar(bins1[:-1], count1, width=bin_width)
ax.bar(bins2[:-1], count2, width=bin_width, facecolor='g')

pylab.show()



