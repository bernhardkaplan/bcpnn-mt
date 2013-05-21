import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import sys
# --------------------------------------------------------------------------
def get_figsize(fig_width_pt):
    inches_per_pt = 1.0/72.0                # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

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
          'figure.figsize': get_figsize(800)}

def set_figsize(fig_width_pt):
    pylab.rcParams['figure.figsize'] = get_figsize(fig_width_pt)

pylab.rcParams.update(params2)


def plot_histogram(d, fig, gid=None, time_range=(0, 1000)):
    binsize = 50
    n_bins = (time_range[1] - time_range[0]) / binsize

    if gid != None:
        spikes = d[d[:, 1] == gid, 0]
    else:
        spikes = d[:, 0]
    n, bins = np.histogram(spikes, bins=n_bins, range=time_range)

    # transform into rate
    n = n * (1000. / binsize)
    
    ax = fig.add_subplot(212)
    ax.bar(bins[:-1], n, width=binsize)
    ax.set_xlim(time_range)
    print 'n, bins', n, bins

# --------------------------------------------------------------------------

fns = sys.argv[1:]

for fn in fns:
    try:
        d = np.loadtxt(fn)
    except:
        d = np.load(fn)

    gid = 5444
    time_range = (0, 3000)

    if gid != None:
        spikes = d[d[:, 1] == gid, 0]
        gids = d[d[:, 1] == gid, 1]
    else:
        spikes = d[:, 0]
        gids = d[:, 1]

    fig = pylab.figure()
    ax = fig.add_subplot(211)
    if (d.ndim == 1):
        x_axis = np.arange(d.size)
        ax.scatter(x_axis, d)
    else:
        ax.plot(spikes, gids, 'o', markersize=1, color='k')
    ax.set_title(fn)
#    ax.set_xlim((0, 1000))
#    print 'xlim:', ax.get_xlim()
    ax.set_ylim((d[:, 1].min()-1, d[:, 1].max()+1))
    ax.set_xlim(time_range)

    plot_histogram(d, fig, gid, time_range=time_range)

pylab.show()

#output_fn = 'delme.dat'
#np.savetxt(output_fn, d)
#output_fn = 'delme.png'
#print output_fn
#pylab.savefig(output_fn)
pylab.show()
