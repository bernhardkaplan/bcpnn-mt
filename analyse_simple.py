import os
import simulation_parameters
import numpy as np
import NetworkSimModuleNoColumns as simulation
import utils


# load simulation parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary



def sort_cells_by_distance_to_stimulus(n_cells):
    tp = np.loadtxt(params['tuning_prop_means_fn'])
    mp = params['motion_params']
    indices, distances = utils.sort_gids_by_distance_to_stimulus(tp , mp) # cells in indices should have the highest response to the stimulus
    print 'Motion parameters', mp
    print 'GID\tdist_to_stim\tx\ty\tu\tv\t\t'
    for i in xrange(n_cells):
        gid = indices[i]
        print gid, '\t', distances[i], tp[gid, :]



n_cells = 20
sort_cells_by_distance_to_stimulus(n_cells)

d = np.loadtxt(params['exc_spiketimes_fn_merged'] + '0.ras')
if d.size > 0:
    nspikes = utils.get_nspikes(params['exc_spiketimes_fn_merged'] + '0.ras', n_cells=params['n_exc'])
    spiking_cells = np.nonzero(nspikes)[0]
    fired_spikes = nspikes[spiking_cells]
   
    print '\nspiking cells', len(spiking_cells), spiking_cells, '\n percentage of spiking cells', float(len(spiking_cells))  / params['n_exc']
    print 'Number of spikes (sum, max, mean, std):\t%d\t%d\t%.2f\t%.2f' % (nspikes.sum(), nspikes.max(), nspikes.mean(), nspikes.std())
    print 'fired_spikes mean %.2f +- %.2f' % (fired_spikes.mean(), fired_spikes.std()), fired_spikes
else:
    print 'NO SPIKES'

#try:
#except:
#    d = np.loadtxt(params['exc_spiketimes_fn_merged'] + '0.ras')
#    print 'Number of spikes:', d.size
