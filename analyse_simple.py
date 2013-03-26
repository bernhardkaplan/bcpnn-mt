import os
import simulation_parameters
import numpy as np
import utils
import sys


if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.info'
    import NeuroTools.parameters as NTP
    fn_as_url = utils.convert_to_url(param_fn)
    print 'Loading parameters from', param_fn
    params = NTP.ParameterSet(fn_as_url)

else:
    print '\nPlotting the default parameters given in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

tp = np.loadtxt(params['tuning_prop_means_fn'])
n_cells = params['n_gids_to_record']
idx, dist = utils.sort_cells_by_distance_to_stimulus(n_cells, verbose=False)

d = np.loadtxt(params['exc_spiketimes_fn_merged'] + '.ras')
if d.size > 0:
    nspikes = utils.get_nspikes(params['exc_spiketimes_fn_merged'] + '.ras', n_cells=params['n_exc'])
    spiking_cells = np.nonzero(nspikes)[0]
    fired_spikes = nspikes[spiking_cells]
   
    print 'tnspikes\tGID\tmin_dist_to_stim\ttp'
    for i in xrange(n_cells):
        gid = idx[i]
        print nspikes[gid], '\t', gid, '\t', dist[i], tp[gid, :]

    print 'Number of spikes (sum, max, mean, std):\t%d\t%d\t%.2f\t%.2f' % (nspikes.sum(), nspikes.max(), nspikes.mean(), nspikes.std())
    print '\nspiking cells', len(spiking_cells), spiking_cells, '\n fraction of spiking cells', float(len(spiking_cells))  / params['n_exc']
    print 'fired_spikes mean %.2f +- %.2f' % (fired_spikes.mean(), fired_spikes.std()), fired_spikes
else:
    print 'NO SPIKES'
    exit(1)





