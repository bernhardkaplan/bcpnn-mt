import numpy as np
import utils
import pylab
import sys
import re
import os

import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

# get the not normalized connection probabilities and merge the files
cmd = 'cat '

folder = params['connections_folder']
fn_base = params['conn_list_ee_fn_base'].rsplit(folder)[1]
output_fn = folder + 'conn_probabilities.dat'

#for fn in os.listdir(folder):
#    to_match = "%spid(\d+)\.dat" % fn_base
#    m = re.match(to_match, fn)
#    if m:
#        print fn
#        cmd += ' %s%s' % (folder, fn)
#cmd += ' > %s' % (output_fn)
#print cmd
#os.system(cmd)

print 'Loading', output_fn
d = np.loadtxt(output_fn)
n_total = d[:, 2].size
n = int(n_total * 0.05)
p_sorted = np.copy(d[:, 2])
print 'Sorting', d[:, 2].size, n, n_total, params['n_exc'] ** 2
p_sorted.sort()
p_top = p_sorted[-n:]
#print p_sorted[:10]
#print p_sorted[-10:]
print 'p min max mean std', p_top.min(), p_top.max(), p_top.mean(), p_top.std(), p_top[-10:]

p_log = np.log(p_top)
#p_log.sort()
#p_log_top = p_log[-n:]
print 'log(p) min max mean std', p_log.min(), p_log.max(), p_log.mean(), p_log.std(), p_log[-10:]

w = 1 / abs(p_log)
print 'w min max mean std', w.min(), w.max(), w.mean(), w.std(), w[-10:]

w_max_bp = w.max()

#p_log_shifted = np.log(d[:, 2] + 1)
#p_log_shifted.sort()
#p_log_shifted = p_log_shifted[-n:]
#print 'log(p) shifted min max mean std', p_log_shifted.min(), p_log_shifted.max(), p_log_shifted.mean(), p_log_shifted.std()



n_bins = 200
n, bins = np.histogram(d[:, 2], bins=n_bins)
n_log, bins_log = np.histogram(p_log, bins=n_bins)
n_w, bins_w = np.histogram(w, bins=n_bins)

fig = pylab.figure()
ax = fig.add_subplot(311)
ax.bar(bins[:-1], n, width = bins[1] - bins[0], label='p_ij')

ax = fig.add_subplot(312)
ax.bar(bins_log[:-1], n_log, width = bins_log[1] - bins_log[0], label='log(p_ij)')

ax = fig.add_subplot(313)
ax.bar(bins_w[:-1], n_w, width = bins_w[1] - bins_w[0], label='weights')
pylab.legend()



w_min, w_max = 0.0001, 0.005
w = utils.linear_transformation(p_top, w_min, w_max)
fig = pylab.figure()
ax = fig.add_subplot(111)
ax.plot(p_top, w, label='p -> weight transformation')
pylab.show()
