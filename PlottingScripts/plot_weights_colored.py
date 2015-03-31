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


def plot_weights_colored(params):

    conn_type = 'ee'
    conn_list_fn = params['merged_conn_list_%s' % conn_type]
    if not os.path.exists(conn_list_fn):
        print 'Merging connection files...'
        utils.merge_connection_files(params, conn_type, iteration=None)

    d = np.loadtxt(conn_list_fn)
    # source perspective
    conn_idx = 0

    # target perspective
#    conn_idx = 1

    limit_tp = False
    v_tolerance = 10
    v_target = 0

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

        if limit_tp and np.abs(tp[gid-1, feature_dimension] - v_target) < v_tolerance:

            # get the indices in the conn list of the current src/tgt cell
            idx_0 = np.where(d[:, conn_idx] == gid)[0]
            # get the tgt/src indices to/from which the current gid projects to/from
            other_gids = np.array(d[idx_0, (conn_idx + 1) % 2], dtype=np.int) # either source or target gids
            
            # filter the other src/tgt tp 
            idx_1 = np.where(np.abs(tp[other_gids-1, feature_dimension] - tp[gid-1, feature_dimension]) < v_tolerance)[0]
            distances = tp[gid - 1, 0] - tp[other_gids[idx_1] - 1, 0]
#            print 'debug', tp[gid -1, feature_dimension], np.max(np.abs(tp[other_gids[idx_1] - 1, feature_dimension]))
            
            weights = d[idx_0[idx_1], 2]
            ax.scatter(distances, weights, c=m.to_rgba(tp[gid-1, feature_dimension]), linewidths=0, s=2)
            if np.where(np.abs(weights) < 0.05)[0].size != 0:
                print 'gids with zero weights:', gid, d[idx_0[idx_1][np.where(np.abs(weights) < 0.05)[0]], (conn_idx + 1) % 2]

        else:
            idx_0 = np.where(d[:, conn_idx] == gid)[0]
            other_gids = np.array(d[idx_0, (conn_idx + 1) % 2], dtype=np.int) # either source or target gids
            distances = tp[gid - 1, 0] - tp[other_gids - 1, 0]
            weights = d[idx_0, 2]
            ax.scatter(distances, weights, c=m.to_rgba(tp[gid-1, feature_dimension]), linewidths=0, s=2)


if __name__ == '__main__':

    params = utils.load_params(sys.argv[1])
    plot_weights_colored(params)

    pylab.show()

