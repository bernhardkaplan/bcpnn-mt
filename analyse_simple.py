import os
import simulation_parameters
import numpy as np
import NetworkSimModuleNoColumns as simulation
import utils


# load simulation parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

n_cells = params['n_gids_to_record']
idx, dist = utils.sort_cells_by_distance_to_stimulus(n_cells)

d = np.loadtxt(params['exc_spiketimes_fn_merged'] + '0.ras')
if d.size > 0:
    nspikes = utils.get_nspikes(params['exc_spiketimes_fn_merged'] + '0.ras', n_cells=params['n_exc'])
    spiking_cells = np.nonzero(nspikes)[0]
    fired_spikes = nspikes[spiking_cells]
   
    print 'Number of spikes (sum, max, mean, std):\t%d\t%d\t%.2f\t%.2f' % (nspikes.sum(), nspikes.max(), nspikes.mean(), nspikes.std())
    print '\nspiking cells', len(spiking_cells), spiking_cells, '\n fraction of spiking cells', float(len(spiking_cells))  / params['n_exc']
    print 'fired_spikes mean %.2f +- %.2f' % (fired_spikes.mean(), fired_spikes.std()), fired_spikes
else:
    print 'NO SPIKES'
    exit(1)

tp = np.loadtxt(params['tuning_prop_means_fn'])




