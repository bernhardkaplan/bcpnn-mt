import os
import simulation_parameters
import numpy as np
import utils
import pylab

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

sim_cnt = 0
#fn = params['exc_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)
#nspikes = utils.get_nspikes(fn, n_cells=params['n_exc'])
#nspikes_normalized = nspikes / nspikes.sum()
#print "nspikes", nspikes
print "N_RF_X: %d\tN_RF_Y:%d\tn_exc: %d\tn_inh: %d\tn_cells:%d" % (params['N_RF_X'], params['N_RF_Y'], params['n_exc'], params['n_inh'], params['n_cells'])
tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
#particles = np.vstack((tuning_prop.transpose(), nspikes_normalized))

# parametrize the spatial layout
n_bins_x, n_bins_y = 20, 20
H, xedges, yedges = np.histogram2d(tuning_prop[:,0], tuning_prop[:, 1], bins=(n_bins_x, n_bins_y))
print "H", H
print "xedges", xedges
print "yedges", yedges


#N_X = params['N_RF_X'] + 10
#N_Y = params['N_RF_Y'] + 10
#hue = True
#hue_zoom = False
#fig_width = 1
#width = 1
#ywidth = 1
#fig, a = utils.spatial_readout(particles, N_X, N_Y, hue, hue_zoom, fig_width, width, ywidth, display=True)

#pylab.show()
