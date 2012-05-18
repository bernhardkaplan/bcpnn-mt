import os
import simulation_parameters
import pylab
import re
import numpy as np
import matplotlib
import utils

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

folder = params['input_folder']
fn_base = params['input_st_fn_base'].rsplit(folder)[1]
n_cells = params['n_cells']

output_fn_base = params['input_fig_fn_base']
output_fn_movie = params['input_movie']
bg_color = 'k'

# parameters
n_frames = 24 # number of output figures
n_bins_x, n_bins_y = 20, 20
output_arrays = [np.zeros((n_bins_x, n_bins_y)) for i in xrange(n_frames)]
time_grid = np.linspace(0, params['t_sim'], n_frames+1, endpoint=True)
tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])

H, x_edges, y_edges = np.histogram2d(tuning_prop[:,0], tuning_prop[:, 1], bins=(n_bins_x, n_bins_y))
print "x_edges", x_edges, x_edges.size
print "y_edges", y_edges, y_edges.size

z_max = 0
for gid in xrange(n_cells):
    fn = params['input_st_fn_base'] + str(gid) + '.npy'
    try:
        spiketrain = np.load(fn)
        binned_spikes, time_bins = np.histogram(spiketrain, time_grid)
        x_pos_cell, y_pos_cell = tuning_prop[gid, 0], tuning_prop[gid, 1] # cell properties
        x_pos_grid, y_pos_grid = utils.get_grid_pos(x_pos_cell, y_pos_cell, x_edges, y_edges) # cell's position in the grid
        print "%d\t%.3e\t%.3e: x:%d\ty:%d  binned_spikes.max: %d" % (gid, x_pos_cell, y_pos_cell, x_pos_grid, y_pos_grid, binned_spikes.max())
        z_max = max(binned_spikes.max(), z_max)
        for frame in xrange(n_frames): # put activity in right time bin (output figure)
            output_arrays[frame][x_pos_grid, y_pos_grid] += binned_spikes[frame]
    except:
        pass # cell gets no input

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
    ax.set_title('Input spikes clustered by positions\nof target neurons in visual space')

#     simple colormap
    pylab.colorbar(cax, cmap=pylab.bone())
    norm = matplotlib.mpl.colors.Normalize(vmin=0, vmax=z_max)
    cax = ax.pcolor(output_arrays[frame], norm=norm)#, edgecolor='k', linewidths='1')

    print "Saving figure: ", output_fn_fig
    pylab.savefig(output_fn_fig)


# let's make a movie!
print 'Creating the movie in file:', output_fn_movie
fps = 12 # frames per second
input_fn = output_fn_base + 'frame%d.png'
command = "ffmpeg -f image2 -r %f -i %s -b 72000 %s" % (fps, input_fn, output_fn_movie)
os.system("rm %s" % output_fn_movie) # remove old one
os.system(command)

show = False
if show:
    pylab.show()
    


#fn_to_plot = []
#gids = []
#for fn in os.listdir(folder):
#    m = re.match("%s(\d+)\." % fn_base, fn)
#    if m:
#        print "Found file:", fn
#        try:
#            data = numpy.load(folder + fn)
#            fn_to_plot.append(fn)
#            gid = int(m.groups()[0])
#            gids.append(gid)
#            spiketrains[gid] = np.load(fn)
#        except:
#            pass

#fn_to_plot = sorted(fn_to_plot)
#gids = sorted(gids)
#n_plots = len(fn_to_plot)

#n_units = params['n_exc']
#x_max = int(round(numpy.sqrt(n_units)))
#y_max = int(round(numpy.sqrt(n_units)))
#if (n_units > x_max * y_max):
#    x_max += 1
#spike_count = numpy.zeros((x_max, y_max))

#fig = pylab.figure()
#ax = fig.add_subplot(111)
#pylab.title('Input spike trains')

#spikes = []
#gid = []
#n_spikes = 0
#for i in xrange(n_plots):
#    fn = fn_to_plot[i]
#    m = re.match("%s(\d+)\." % fn_base, fn)
#    mc_index = int(m.groups()[0])
#    data = numpy.load(folder + fn)
#    spike_count[mc_index % x_max, mc_index / x_max] = data.size
#    ax.plot(data, mc_index * numpy.ones(data.size), 'o', markersize=1, color='k')

#ax.set_ylim(-1, max(gids)+1)


#for gid in xrange(n_cells):
#    binned_spikes, time_bins = np.histogram(spiketrains[gid], time_grid)
#    x_pos_cell, y_pos_cell = tuning_prop[gid, 0], tuning_prop[gid, 1] # cell properties
#    x_pos_grid, y_pos_grid = utils.get_grid_pos(x_pos_cell, y_pos_cell, x_edges, y_edges) # cell's position in the grid

# spike count grid

#fig = pylab.figure()
#ax1 = fig.add_subplot(111)
#ax1.set_title('Input spikes integrated over time')
#pylab.show()

#cax1 = ax1.pcolor(spike_count)#, edgecolor='k', linewidths='1')
#cax1 = ax1.imshow(activity, interpolation='nearest')
#ax1.set_ylim((0, spike_count[:, 0].size))
#ax1.set_xlim((0, spike_count[0, :].size))


# plot histograms
#n_bins = 50
#fig = pylab.figure()
#pylab.subplots_adjust(bottom=0.05, top=0.9, hspace=-0.1)

#x_max = 1000
#xticks = numpy.arange(0, x_max, x_max/10.)
#for i in xrange(n_plots):
#    fn = fn_to_plot[i]
#    ax = fig.add_subplot(n_plots,1,n_plots-i)
#    data = numpy.load(folder + fn)
#    print fn, data.size
#    count, bins = numpy.histogram(data, n_bins)
#    gid = gids[i]
#    ax.bar(bins[:-1], count, width=bins[1]-bins[0], facecolor='blue')#, normed=1)
#    ax.set_yticks((0, max(count)))
#    ax.set_yticklabels(('0', str(max(count))))
#    ax.set_xlim((0, x_max))
#    ax.set_xticks(xticks)
#    ax.set_xticklabels(['' for i in xrange(len(xticks))])

#ax = fig.get_axes()[0]
#ax.set_xticks(xticks)
#ax.set_xticklabels(['%d' % i for i in xticks])
#ax.set_title('Input spike counts')

#pylab.show()
