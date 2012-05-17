import numpy as np
import pylab
import simulation_parameters

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

sim_cnt = 0
fn = params['conn_list_ee_fn_base'] + '%d.dat' % sim_cnt
d = np.loadtxt(fn)

weights = d[:, 2]
delays = d[:, 3]

n_w, bins_w = np.histogram(weights, bins=20)
print "bins_w", bins_w, n_w
n_d, bins_d = np.histogram(delays, bins=20)
print "bins_d", bins_d, n_d

fig = pylab.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.bar(bins_w[:-1], n_w, width=bins_w[1]-bins_w[0])
ax2.bar(bins_d[:-1], n_d, width=bins_d[1]-bins_d[0])

ax1.set_xlabel('Weights')
ax1.set_ylabel('Count')
ax2.set_xlabel('Delays')
ax2.set_ylabel('Count')

ax1.set_title(fn)
pylab.show()
