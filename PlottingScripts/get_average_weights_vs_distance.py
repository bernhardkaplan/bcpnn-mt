import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import utils
import matplotlib
#matplotlib.use('Agg')
import pylab
from PlottingScripts.plot_spikes_sorted import plot_spikes_sorted
import json

plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 12,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 10,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 3,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.15,
              'figure.subplot.bottom':.15,
              'figure.subplot.right':.80,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}


def get_average_weights(params, conn_idx=0):

    pylab.rcParams.update(plot_params)
    conn_list_fn = params['merged_conn_list_ee']
    utils.merge_connection_files(params, 'ee')
    d = np.loadtxt(conn_list_fn)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    # deteremine which gids to use as source (target)
    feature_radius = 20.
    feature_tgt = [0.] # could be all values here --> no filtering
#    feature_tgt = np.linspace(0., 180, params['n_theta'], endpoint=False) # could be all values here --> no filtering
    filtered_gids = np.array([])
    for v_tgt in feature_tgt:
        gids = np.where(utils.torus_distance_array(tp[:, 4], v_tgt, w=180) < feature_radius)[0] + 1 # nest gids expected later --> + 1
        filtered_gids = np.r_[gids]
#        print 'gids size', gids.size
    filtered_gids = np.unique(filtered_gids)    
#    print 'filtered_gids:', filtered_gids.size, params['n_exc']

    all_distances = np.array([])
    all_weights = np.array([])
    #filtered_gids = np.array([161])
    #print tp[filtered_gids-1, :]

    for gid in filtered_gids:
        idx = np.where(d[:, conn_idx] == gid)[0]
        other_gids = np.array(d[idx, (conn_idx + 1) % 2], dtype=np.int)
        filtered_within_other_gids = np.where(utils.torus_distance_array(tp[other_gids - 1, 4], tp[gid-1, 4], w=180.) < feature_radius)[0]
        filtered_other_gids = other_gids[filtered_within_other_gids]
        d_ = utils.get_targets(d, gid)
        for i_, ogid in enumerate(filtered_other_gids):
            idx = np.where(d_[:, (conn_idx+1)%2] == ogid)[0]
            all_distances = np.r_[all_distances, tp[gid-1, 0] - tp[ogid-1,0]]
            all_weights = np.r_[all_weights, d_[idx, 2]]

    xbins = np.linspace(-1., 1., 81, endpoint=True)
    binwidth = 0.05
    avg_weight = np.zeros((xbins.size, 2))
    for i_, xbin in enumerate(xbins):
        idx = np.where(np.abs(all_distances - xbin) < binwidth)[0]
        if idx.size > 0:
            avg_weight[i_, 0] = all_weights[idx].mean()
            avg_weight[i_, 1] = all_weights[idx].std()
#            avg_weight[i_, 1] /= np.sqrt(idx.size)
        else:
            print xbin, 'none found'
                
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(xbins, avg_weight[:, 0], yerr=avg_weight[:, 1])

    xlabel = 'Distance between source and target cell'
    ylabel = 'Weight'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    output_fn = params['data_folder'] + 'average_weights_vs_distance.dat'
    output_array = np.array((xbins, avg_weight[:, 0], avg_weight[:, 1]))
    print 'Saving output data to:', output_fn
    np.savetxt(output_fn, output_array.transpose())

    output_fig = params['figures_folder'] + 'average_weight_vs_distance_taui%d_vtrain%.2f.png' % (params['bcpnn_params']['tau_i'], params['v_min_training'])
    print 'Saving figure to:', output_fig
    fig.savefig(output_fig, dpi=200)



if __name__ == '__main__':

#    folder_name = 'TrainingSim_Cluster__50x2x1_0-100_taui50_nHC20_nMC2_vtrain1.00-1.0'
    folder_name = sys.argv[1]
    params = utils.load_params(folder_name)
    source_perspective = True
    if source_perspective:
        conn_idx = 0
    else:
        conn_idx = 1
    get_average_weights(params, conn_idx)

    pylab.show()

