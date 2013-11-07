import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import sys
import utils 

gid_axis = 0
time_axis = 1

# --------------------------------------------------------------------------
params2 = {'backend': 'png',
          'axes.labelsize': 12,
          'text.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': 12,
          'lines.markersize' : 0.1,
          'font.size': 12,
          'path.simplify': False,
          'figure.figsize': utils.get_figsize(800)}

def set_figsize(fig_width_pt):
    pylab.rcParams['figure.figsize'] = utils.get_figsize(fig_width_pt)

pylab.rcParams.update(params2)


def plot_histogram(d, fig, gids_to_plot=None, time_range=False):

    if time_range == False:
        time_range = (0, d[:, time_axis].max())
    print 'Plot histogram time_range:', time_range
    binsize = 50
    n_bins = (time_range[1] - time_range[0]) / binsize

    if gids_to_plot == None:
        gids_to_plot = np.unique(d[:, gid_axis])

    spikes = np.array([])
    for gid in gids_to_plot:
        idx = (d[:, gid_axis] == gid).nonzero()[0]
        spike_times = d[idx, time_axis]
        spikes = np.r_[spikes, spike_times]

    n, bins = np.histogram(spikes, bins=n_bins, range=time_range)

    # transform into rate
    n = n * (1000. / binsize)
    
    ax.bar(bins[:-1], n, width=binsize)
    ax.set_xlim(time_range)
    ax.set_ylabel('Network output rate (summed over %d cells) [Hz]' % len(gids_to_plot)) 
    ax.set_xlabel('Time [ms]')

    # print number of spikes and output rate for all gids
    for gid in np.unique(gids):
        idx = (d[:, gid_axis] == gid).nonzero()[0]
        spike_times = d[idx, time_axis]
        print 'Gid %d fired %d spikes' % (gid, spike_times.size)

#    print 'n, bins', n, bins

# --------------------------------------------------------------------------

fns = sys.argv[1:]

fig = pylab.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for fn in fns:
    try:
        d = np.loadtxt(fn)
    except:
        d = np.load(fn)

    gid = None

    if gid != None:
        spikes = d[d[:, gid_axis] == gid, time_axis]
        gids = d[d[:, gid_axis] == gid, gid_axis]
    else:
        gids_to_plot = np.unique(d[:, gid_axis])
        spikes = np.array([])
        gids = np.array([]) # must be same length as spikes
        for gid in gids_to_plot:
            idx = (d[:, gid_axis] == gid).nonzero()[0]
            spike_times = d[idx, time_axis]
            spikes = np.r_[spikes, spike_times]
            gids_ = d[idx, gid_axis]
            gids = np.r_[gids, gids_]

#    time_range = (0, 1600)
    if (d.ndim == 1):
        x_axis = np.arange(d.size)
        ax.scatter(x_axis, d)
    else:
        ax.plot(spikes, gids, 'o', markersize=1, color='k')
    ax.set_title(fn)
#    ax.set_xlim((0, 1000))
#    print 'xlim:', ax.get_xlim()
    ax.set_ylim((d[:, gid_axis].min()-1, d[:, gid_axis].max()+1))
    ax.set_xlim()#time_range)

    plot_histogram(d, fig, np.unique(gids))#, time_range=time_range)

pylab.show()

#output_fn = 'delme.dat'
#np.savetxt(output_fn, d)
#output_fn = 'delme.png'
#print output_fn
#pylab.savefig(output_fn)
pylab.show()
