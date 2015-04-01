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

    clim2 = (0., 1.)
    norm2 = matplotlib.colors.Normalize(vmin=clim2[0], vmax=clim2[1])
    m2 = matplotlib.cm.ScalarMappable(norm=norm2, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m2.set_array(tp[:, feature_dimension])
    colorlist2 = m2.to_rgba(tp[:, 0])

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for gid in xrange(1, params['n_exc'] + 1):
        idx = np.where(d[:, 0] == gid)[0]
        spikes = d[idx, 1]
        ax.scatter(spikes, gid * np.ones(spikes.size), c=m.to_rgba(tp[gid-1, feature_dimension]), linewidths=0, s=3)

#     plot spike count histogram
    nspikes, bins = np.histogram(d[:, 0], bins=np.arange(1, params['n_exc'] + 2)) # + 2 because you always need 1 bin more than elements to be binned
    fig = pylab.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.scatter(tp[:, 0], nspikes, c=m.to_rgba(tp[:, 4]), linewidths=0, s=5)
    ax2.scatter(tp[:, 4], nspikes, c=m2.to_rgba(tp[:, 0]), linewidths=0, s=5)


def plot_nspikes_versus_stim_speed(params):

    fn = params['exc_spiketimes_fn_merged']
    mp = np.loadtxt(params['training_stimuli_fn'])
    stim_durations = np.loadtxt(params['stim_durations_fn'])
    d = np.loadtxt(fn)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    fig = pylab.figure()
    ax = fig.add_subplot(111)

    feature_dimension = 4
    clim = (0., 180.)
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(tp[:, feature_dimension])
    colorlist= m.to_rgba(tp[:, feature_dimension])


    for i_stim in xrange(params['n_stim'] - 1):
        t0, t1 = stim_durations[:i_stim].sum(), stim_durations[:i_stim+1].sum()
#        print 't0, t1 i_stim', i_stim, t0, t1, mp[i_stim, :]
        
        (spikes, gids) = utils.get_spikes_within_interval(d, t0, t1, time_axis=1, gid_axis=0)
        for gid in gids:
            n_ = np.where(gids == gid)[0].size
            ax.scatter(mp[i_stim, 2], n_, c=m.to_rgba(tp[gid - 1, 4]), s=5, linewidths=0)
#        print 'spikes', spikes
#        print 'gids', gids
#        ax.scatter(mp[i_stim, 2], 




if __name__ == '__main__':

    params = utils.load_params(sys.argv[1])
    plot_spikes_colored(params)
#    plot_nspikes_versus_stim_speed(params)
    pylab.show()
