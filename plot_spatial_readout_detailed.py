import os
import simulation_parameters
import numpy as np
import utils
import matplotlib
matplotlib.use('Agg')
import pylab
import os


sim_cnt = 0 # which run do you want to plot?
# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
output_fn_base = params['spatial_readout_fn_base'] + 'sim%d_' % sim_cnt
output_fn_movie = params['spatial_readout_movie']
bg_color = 'k'

# parameters
n_frames = 20    # number of output figures
n_bins_x, n_bins_y = 20, 20
output_arrays = [np.zeros((n_bins_x, n_bins_y)) for i in xrange(n_frames)]
time_grid = np.linspace(0, params['t_sim'], n_frames+1, endpoint=True)

# load tuning properties and activity
tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
n_cells = tuning_prop[:,0].size # = params['n_exc']
fn = params['exc_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)
nspikes, spiketrains = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
nspikes_binned = np.zeros((n_cells, n_frames))
nspikes_binned_normalized = np.zeros((n_cells, n_frames))

# spike time binning
for gid in xrange(n_cells):
    spiketimes = spiketrains[gid]
    if (len(spiketimes) > 0):
        count, bins = np.histogram(spiketimes, bins=time_grid)
        nspikes_binned[gid, :] = count

# normalization
for i in xrange(int(n_frames)):
    if (nspikes_binned[:, i].sum() > 0):
        nspikes_binned_normalized[:, i] = nspikes_binned[:, i] / nspikes_binned[:,i].sum()

scale = 20.
thetas = np.zeros(n_cells)
for gid in xrange(n_cells):
    thetas[gid] = np.arctan2(tuning_prop[gid, 3], tuning_prop[gid, 2])

l_max, l_offset = 127, 0
"""
High confidence --> lightness
Orientation (theta) --> hue
    h : [0, 360) degree
    s : [0, 1] 
    l : [0, 1]
"""
x_max = scale * np.max(tuning_prop[:, 0]) * 1.05
y_max = scale * np.max(tuning_prop[:, 1]) * 1.05
for frame in xrange(n_frames):
    print "Plotting frame: ", frame
    fig = pylab.figure()
    ax = fig.add_subplot(111, axisbg=bg_color)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Spatial activity readout')

    spiking_cells = nspikes_binned[:, frame].nonzero()[0]
    z_max = np.max(nspikes_binned_normalized[:, frame])
    
    print "n_spiking cells in time_bin %d-%d :" % (time_grid[frame], time_grid[frame+1]), spiking_cells.size
    for gid in spiking_cells:
        (x, y, u, v) = tuning_prop[gid, :]
        theta = thetas[gid]
        h = (theta + np.pi) / (2 * np.pi) * 360. # theta determines h, h must be [0, 360)
        l = (nspikes_binned_normalized[gid, frame] / z_max * l_max + l_offset) / 255. # [0, 1]
        s = 1. # saturation [0, 1]
        (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
#        print "color rgb", r, g, b , "hsl", h, s, l
        ax.plot((x*scale, x*scale+u), (y*scale, y*scale+v), c=(r,g,b))
    ax.set_xlim((-1.0, x_max))
    ax.set_ylim((-1.0, y_max))

    output_fn_fig = output_fn_base + 'frame%d.png' % (frame)
    print "Saving figure: ", output_fn_fig
    pylab.savefig(output_fn_fig)

# let's make a movie!
print 'Creating the movie in file:', output_fn_movie
os.system('rm %s' % output_fn_movie)
fps = 8     # frames per second
input_fn = output_fn_base + 'frame%d.png'
command = "ffmpeg -f image2 -r %f -i %s -b 200k %s" % (fps, input_fn, output_fn_movie)
os.system(command)

show = False
if show:
    pylab.show()
    
