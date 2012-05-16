import pylab
import numpy as np
import re
import simulation_parameters
import utils

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

fn = params['inh_spiketimes_fn_base'] + '0.ras'
n_cells = params['n_inh']
nspikes, spiketimes = utils.get_nspikes(fn, n_cells, get_spiketrains=True)

fig = pylab.figure()
ax = fig.add_subplot(111)
for cell in xrange(int(len(spiketimes))):
    print cell, spiketimes[cell]

    ax.plot(cell * np.ones(nspikes[cell]), spiketimes[cell], '|', color='k')
    
pylab.show()

#folder = params['spiketimes_folder']
#fn_base = params['exc_spiketimes_fn_base'].rsplit(folder)[1]

#fn_to_plot = []
#gids = []

#spikes = []
#for fn in os.listdir(folder):
#    m = re.match("%s(\d+)\." % fn_base, fn)
#    print fn, fn_base
#    if m:
#        print "Found ", fn
#        try:
#            d = np.loadtxt(folder + fn)
#            mc_index = int(m.groups()[0])
#            for i in xrange(d[:, 0].size):
#                t = d[i, 0]
#                gid = mc_index * params['n_exc_per_mc'] + d[i, 1]
#                spikes.append((t, gid))
#        except:
#            pass

#data = np.array(spikes)
#pylab.plot(data[:,0], data[:,1], 'o', markersize=2, color='k')

#pylab.show()
