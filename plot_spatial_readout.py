import os
import simulation_parameters
import numpy as np
import utils
import pylab
import matplotlib
import os


sim_cnt = 0 # which run do you want to plot?
# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
output_fn_base = params['spatial_readout_fn_base'] + 'sim%d_' % sim_cnt
output_fn_movie = params['spatial_readout_movie']
bg_color = 'k'

# parameters
n_frames = 50    # number of output figures
n_bins_x, n_bins_y = 20, 20
output_arrays = [np.zeros((n_bins_x, n_bins_y)) for i in xrange(n_frames)]
time_grid = np.linspace(0, params['t_sim'], n_frames+1, endpoint=True)

# load tuning properties and activity
tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
n_cells = tuning_prop[:,0].size # = params['n_exc']
fn = params['exc_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)
nspikes, spiketrains = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
nspikes_normalized = nspikes / nspikes.sum()

print "nspikes", nspikes
print "N_RF_X: %d\tN_RF_Y:%d\tn_exc: %d\tn_inh: %d\tn_cells:%d" % (params['N_RF_X'], params['N_RF_Y'], params['n_exc'], params['n_inh'], params['n_cells'])
#particles = np.vstack((tuning_prop.transpose(), nspikes_normalized))

# parametrize the spatial layout
H, x_edges, y_edges = np.histogram2d(tuning_prop[:,0], tuning_prop[:, 1], bins=(n_bins_x, n_bins_y))
print "x_edges", x_edges, x_edges.size
print "y_edges", y_edges, y_edges.size


z_max = 0
for gid in xrange(n_cells):
    binned_spikes, time_bins = np.histogram(spiketrains[gid], time_grid)
    x_pos_cell, y_pos_cell = tuning_prop[gid, 0], tuning_prop[gid, 1] # cell properties
    x_pos_grid, y_pos_grid = utils.get_grid_pos(x_pos_cell, y_pos_cell, x_edges, y_edges) # cell's position in the grid
#    print "%d\t%.3e\t%.3e: x:%d\ty:%d" % (gid, x_pos_cell, y_pos_cell, x_pos_grid, y_pos_grid)
    z_max = max(binned_spikes.max(), z_max)
    for frame in xrange(n_frames): # put activity in right time bin (output figure)
        output_arrays[frame][x_pos_grid, y_pos_grid] = binned_spikes[frame]

for frame in xrange(n_frames):
    output_fn_dat = output_fn_base + 'frame%d.dat' % (frame)
    output_fn_fig = output_fn_base + 'frame%d.png' % (frame)
    print "Saving to file: ", output_fn_dat
    np.savetxt(output_fn_dat, output_arrays[frame])

    print "Plotting frame: ", frame
    fig = pylab.figure()
    ax = fig.add_subplot(111, axisbg=bg_color)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Spatial activity readout')

#    for gid in xrange(n_cells): # compute color
#     simple colormap
    norm = matplotlib.mpl.colors.Normalize(vmin=0, vmax=z_max)
    cax = ax.pcolor(output_arrays[frame], norm=norm)#, edgecolor='k', linewidths='1')
    pylab.colorbar(cax, cmap=pylab.bone())


    print "Saving figure: ", output_fn_fig
    pylab.savefig(output_fn_fig)


# let's make a movie!
print 'Creating the movie in file:', output_fn_movie
fps = 8     # frames per second
input_fn = output_fn_base + 'frame%d.png'
command = "ffmpeg -f image2 -r %f -i %s -b 72000 %s" % (fps, input_fn, output_fn_movie)
os.system("rm %s" % output_fn_movie) # remove old one
os.system(command)

show = False
if show:
    pylab.show()
    





#N_X = params['N_RF_X'] + 10
#N_Y = params['N_RF_Y'] + 10
#hue = True
#hue_zoom = False
#fig_width = 1
#width = 1
#ywidth = 1
#fig, a = utils.spatial_readout(particles, N_X, N_Y, hue, hue_zoom, fig_width, width, ywidth, display=True)

#pylab.show()
