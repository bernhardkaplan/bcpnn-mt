import numpy as np
import utils
import simulation_parameters
import pylab
ps = simulation_parameters.parameter_storage()
params = ps.params

n_x = 200
n_y = 200
pos = np.linspace(0., 1., n_x, endpoint=True)
theta = np.linspace(0., 180., n_y, endpoint=False)
n_cells = n_x * n_y
tp = np.zeros((n_cells, 5))
idx = 0
for i_ in xrange(n_x):
    x = pos[i_]
    for j_ in xrange(n_y):
        y = theta[j_]
        tp[idx, 0] = x
        tp[idx, 1] = .5
        tp[idx, 4] = y
        idx += 1
rfs = np.zeros((n_cells, 5))
rfs[:, 0] = 0.03
rfs[:, 4] = 20.
predictor_params = [0.5, 0.5, 0., 0., 0.1]

L = utils.get_input(tp, rfs, params, predictor_params, motion='bar')

L_map = np.zeros((n_x, n_y))
idx = 0
for i_ in xrange(n_x):
    for j_ in xrange(n_y):
        L_map[i_, j_] = L[idx]
        idx += 1
#pylab.plot(L)

fig = pylab.figure()
ax = fig.add_subplot(111)
print "plotting ...."

L_map = L_map.transpose()
cax = ax.pcolormesh(L_map)#, edgecolor='k', linewidths='1')
#pylab.ylim(0, L_map.shape[0])
#pylab.xlim(0, L_map.shape[1])
pylab.colorbar(cax)

#new_yticks = ['%.2f' % (tp[int(y_), 4]) for y_ in ax.get_yticks()]
#n_yticks = 10
#ax.set_yticks(range(n_yticks))
new_yticks = ['%.2f' % (theta[int(y_)]) for y_ in ax.get_yticks()[:-1]]
#new_yticks = ['%.2f' % (theta[j_]) for j_ in xrange(n_y)]
print 'new_yticks:', new_yticks
ax.set_yticklabels(new_yticks)

new_xticks = ['%.2f' % (pos[x_]) for x_ in ax.get_xticks()[:-1]]
ax.set_xticklabels(new_xticks)

pylab.show()
