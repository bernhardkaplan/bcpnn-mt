import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import utils 



def plot_spikes_colored(params):
    fn = params['exc_spiketimes_fn_merged']
    d = np.loadtxt(fn)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])
    
    feature_dimension = 4
    clim = (0., 180.)
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(tp[:, feature_dimension])
    colorlist= m.to_rgba(tp[:, feature_dimension])

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for gid in xrange(1, params['n_exc'] + 1):
        idx = np.where(d[:, 0] == gid)[0]
        spikes = d[idx, 1]
        ax.scatter(spikes, gid * np.ones(spikes.size), c=m.to_rgba(tp[gid-1, feature_dimension]), linewidths=0, s=2)


if __name__ == '__main__':

    params = utils.load_params(sys.argv[1])
    plot_spikes_colored(params)

    pylab.show()
