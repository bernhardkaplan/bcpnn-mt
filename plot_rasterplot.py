import pylab
import numpy
import sys
import os
import re
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

folder = params['spiketimes_folder']
fn_base = params['exc_spiketimes_fn_base'].rsplit(folder)[1]

fn_to_plot = []
gids = []

spikes = []
for fn in os.listdir(folder):
    m = re.match("%s(\d+)\." % fn_base, fn)
    print fn, fn_base
    if m:
        print "Found ", fn
        try:
            d = numpy.loadtxt(folder + fn)
            mc_index = int(m.groups()[0])
            for i in xrange(d[:, 0].size):
                t = d[i, 0]
                gid = mc_index * params['n_exc_per_mc'] + d[i, 1]
                spikes.append((t, gid))
        except:
            pass

data = numpy.array(spikes)
pylab.plot(data[:,0], data[:,1], 'o', markersize=2, color='k')

pylab.show()
