import sys
import os
import random
import pylab
import numpy as np
import utils
from matplotlib.mlab import griddata
import simulation_parameters
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

#ps = simulation_parameters.parameter_storage()
#params = ps.params

def plot_contour_connectivity(params, gid=None):
    random.seed(0)
    d = np.loadtxt(params['merged_conn_list_ee'])
    tp = np.loadtxt(params['tuning_prop_means_fn'])
    if gid == None:
        gids = np.unique(d[:, 0])
        gid = random.choice(gids)
#    gid = 29
    targets = utils.get_targets(d, gid)

    target_gids = targets[:, 1].astype(int)
    weights = targets[:, 2]
    x_tgt = tp[target_gids, 0]
    vx_tgt = tp[target_gids, 2]

    dx = 0.01 * 3
    dvx = 0.05 * 3
    x_min, x_max = .8 * x_tgt.min(), 1.2 * x_tgt.max()
    vx_min, vx_max = .8 * vx_tgt.min(), 1.2 * vx_tgt.max()
    x_grid, vx_grid = np.mgrid[slice(x_min, x_max, dx), 
                    slice(vx_min, vx_max, dvx)]
    print 'x_grid[0, :]', x_grid[:, 0]
    print 'vx_grid[0, :]', vx_grid[0, :]

    x_edges = x_grid[:, 0]
    vx_edges = vx_grid[0, :]
    z_data = np.zeros(x_grid.shape)
    for i_, w in enumerate(weights):
        x0, v0 = x_tgt[i_], vx_tgt[i_]
        (x_, y_) = utils.get_grid_pos(x0, v0, x_edges, vx_edges)
    #    print 'w %.3e x0 %.2e vx %.2e, x_ %d y_ %d' % (w, x0, v0, x_, y_)
        z_data[x_, y_] += w

    z_data = z_data[:-1, :-1]


#    fig = pylab.figure()
#    ax = fig.add_subplot(111)
#    im = ax.pcolormesh(x_grid, vx_grid, z_data)
#    ax.plot(x_tgt, vx_tgt, 'o', markeredgewidth=0, c='k', markersize=4)


    fig = pylab.figure()
    ax = fig.add_subplot(111)

    #levels = MaxNLocator(nbins=15).tick_values(z_data.min(), z_data.max())
    cmap = pylab.get_cmap('jet')
    ax.contourf(x_grid[:-1, :-1] + dx / 2.,
                vx_grid[:-1, :-1] + dvx / 2., z_data, 30, \
                  cmap=cmap)

    ax.plot(x_tgt, vx_tgt, 'o', markeredgewidth=0, c='k', markersize=4)
    ax.plot(tp[gid, 0], tp[gid, 2], '*', markersize=15, c='y', markeredgewidth=0)

    ax.set_ylim((vx_edges[1], vx_edges[-2]))
    ax.set_xlim((x_edges[1], x_edges[-2]))
    pylab.show()



if __name__ == '__main__':

#    print 'Running merge_connlists.py...'
#    os.system('python merge_connlists.py')
    np.random.seed(0)
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            gid = int(sys.argv[1])
            param_fn = sys.argv[2]
            if os.path.isdir(param_fn):
                param_fn += '/Parameters/simulation_parameters.json'
            import json
            f = file(param_fn, 'r')
            print 'Loading parameters from', param_fn
            params = json.load(f)
        else:
            param_fn = sys.argv[1]
            if os.path.isdir(param_fn):
                param_fn += '/Parameters/simulation_parameters.json'
            import json
            f = file(param_fn, 'r')
            print 'Loading parameters from', param_fn
            params = json.load(f)
            gid = np.int(np.loadtxt(params['gids_to_record_fn'])[0])
    else:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        try:
            gid = np.int(np.loadtxt(params['gids_to_record_fn'])[0])
        except:
            gid = None

    plot_contour_connectivity(params, gid)
