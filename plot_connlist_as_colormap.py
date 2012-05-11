import pylab
import numpy
import sys
import os
import utils
import simulation_parameters

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
sim_cnt = 0
if (len(sys.argv) < 2):
    fn = params['conn_list_ee_fn_base'] + '%d.dat' % sim_cnt
    print "Plotting default file:", fn
else:
    fn = sys.argv[1]

n_cells = params['n_exc']
data = utils.convert_connlist_to_matrix(fn, n_cells)

fig = pylab.figure()
ax = fig.add_subplot(111)
print "plotting ...."
#cax = ax.imshow(data[:,:12])
#cax = ax.pcolor(data, edgecolor='k', linewidths='1')

ax.set_title(fn)
cax = ax.pcolor(data)#, edgecolor='k', linewidths='1')
#cax = ax.pcolor(data, cmap='binary')
#cax = ax.pcolor(data, cmap='RdBu')

ax.set_ylim(0, data.shape[0])
ax.set_xlim(0, data.shape[1])

#cax = ax.pcolor(log_data)#, edgecolor='k', linewidths='1')


pylab.colorbar(cax)

#plot_fn = "testfig.png"
#print "saving ....", plot_fn
#pylab.savefig(plot_fn)

pylab.show()
