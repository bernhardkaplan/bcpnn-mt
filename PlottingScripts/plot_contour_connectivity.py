import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
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
    if not os.path.exists(params['merged_conn_list_ee']):
        print 'Running merge_connlists.py...'
        os.system('python merge_connlists.py')
    d = np.loadtxt(params['merged_conn_list_ee'])
    tp = np.loadtxt(params['tuning_prop_means_fn'])
    while gid == None:
        gid = utils.select_well_tuned_cells_1D(tp, params['motion_params'], params, 1)
        n_out = (d[:, 2] == gid).nonzero()
        if n_out == 0:
            print 'GID %d connects to NO CELLS' % gid
            gid = None

    print 'GID:', gid
    targets = utils.get_targets(d, gid)

    target_gids = targets[:, 1].astype(int)
    weights = targets[:, 2]
    x_tgt = tp[target_gids, 0]
    vx_tgt = tp[target_gids, 2]

    dx = 0.01 * 3
    dvx = 0.05 * 3
    autolimit = True
    print 'x_tgt min max', x_tgt.min(), x_tgt.max()
    print 'vx_tgt min max', vx_tgt.min(), vx_tgt.max()
    if autolimit:
        x_min, x_max = .6 * x_tgt.min(), 1.4 * x_tgt.max()
        if np.sign(vx_tgt.min()) == -1:
            vx_min = 1.5 * vx_tgt.min()
        else:
            vx_min = .5 * vx_tgt.min()
        vx_max = 1.3 * vx_tgt.max()
    else:
        ylim = (-3.5, 4)
        xlim = (0.0, 1.0)
        x_min, x_max = xlim[0], xlim[1]
        vx_min, vx_max = ylim[0], ylim[1]
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


    rcParams = { 'axes.labelsize' : 20,
                'axes.titlesize'  : 20,
                'label.fontsize': 20,
                'xtick.labelsize' : 18, 
                'ytick.labelsize' : 18, 
                'legend.fontsize': 16, 
                'figure.subplot.left':.15,
#                'figure.subplot.bottom':.10,
#                'figure.subplot.right':.95,
#                'figure.subplot.top':.90, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)
    fig = pylab.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

#    z_data = z_data[:-1, :-1]
    z_data = z_data
    #levels = MaxNLocator(nbins=15).tick_values(z_data.min(), z_data.max())
    cmap = pylab.get_cmap('jet')
#    ax.contourf(x_grid[:-1, :-1] + dx / 2.,
#                vx_grid[:-1, :-1] + dvx / 2., z_data, 30, \
#                  cmap=cmap)
    CS = ax.contourf(x_grid + dx / 2.,
                vx_grid + dvx / 2., z_data, 30, \
                  cmap=cmap)
    pylab.colorbar(CS)


    ax.plot(x_tgt, vx_tgt, 'o', markeredgewidth=0, c='k', markersize=4)
    ax.plot(tp[gid, 0], tp[gid, 2], '*', markersize=15, c='y', markeredgewidth=0, label='source')
    ax.legend(numpoints=1)

    # autoscale figure limits
    if autolimit:
        ylim = (vx_edges[1], vx_edges[-2])
        xlim = (x_edges[1], x_edges[-2])
        output_fig = params['figures_folder'] + 'contour_connectivity_%d_wsx%.2e_wsv%.2e_a.png' % (gid, params['w_sigma_x'], params['w_sigma_v'])
    else:
        ylim = (-3.5, 4)
        xlim = (0.6, 0.72)
        output_fig = params['figures_folder'] + 'contour_connectivity_%d_wsx%.2e_wsv%.2e.png' % (gid, params['w_sigma_x'], params['w_sigma_v'])

#    print 'x_grid[:-1, :-1]', x_grid[:-1, :-1]
#    print 'vx_grid[:-1, :-1]', vx_grid[:-1, :-1]
    ylim = (.5 * vx_edges[1], vx_edges[-2])
    xlim = (x_edges[1], x_edges[-2])
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    ax.set_xlabel('Receptive field position $x$')
    ax.set_ylabel('Preferred speed $v_x$')
    title = 'Distribution of outgoing connections for one example source cell\n'
    title += '$\sigma_X = %.2f\quad\sigma_V=%.2f$' % (params['w_sigma_x'], params['w_sigma_v'])
    ax.set_title(title)

    print 'Saving figure to:', output_fig
    pylab.savefig(output_fig, dpi=300)

    ######################
    # plot formula 
    ######################
    fig = pylab.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    n_pts = 1000
    x_min, x_max = 0., 1.
    vx_min, vx_max = -2, 2.
    x_sample = np.random.uniform(x_min, x_max, n_pts)
    vx_sample = np.random.uniform(vx_min, vx_max, n_pts)

    # compute the anisotropy in the formula
    tau_perception = params['neural_perception_delay'] / params['t_stimulus']
    d_ij = utils.torus_distance_array(x_sample, tp[gid, 0] * np.ones(n_pts))
    latency = d_ij / np.abs(tp[gid, 2])
    x_pred = tp[gid, 0] + tp[gid, 2] * (latency + tau_perception) # with delay-compensation
    d_pred_tgt = utils.torus_distance_array(x_pred, x_sample)
    z = np.exp(- (d_pred_tgt)**2 / (2 * params['w_sigma_x']**2)) \
            * np.exp(- ((tp[gid, 2] - vx_sample)**2/ (2 * params['w_sigma_v']**2)))
    x_grid = np.arange(x_min, x_max, dx) 
    vx_grid = np.arange(vx_min, vx_max, dvx)
    # grid the data
    z_grid = griddata(x_sample, vx_sample, z, x_grid, vx_grid, interp='linear')
    n_levels = 200
    CS = ax.contourf(x_grid, vx_grid, z_grid, n_levels, cmap=pylab.cm.jet,
                      vmax=abs(z_grid).max(), vmin=-abs(z_grid).max())
    ax.plot(x_tgt, vx_tgt, 'o', markeredgewidth=0, c='k', markersize=4)
    ax.plot(tp[gid, 0], tp[gid, 2], '*', markersize=15, c='y', markeredgewidth=0, label='source')
    ax.legend(numpoints=1)
    ax.set_xlabel('Receptive field position $x$')
    ax.set_ylabel('Preferred speed $v_x$')
    title = 'Distribution of connection probabilities'
    title += '\n$\sigma_X = %.2f\quad\sigma_V=%.2f$' % (params['w_sigma_x'], params['w_sigma_v'])
    ax.set_title(title)
    pylab.colorbar(CS)

#    ylim = (vx_min, vx_max)
#    xlim = (x_min, x_max)
    ylim = (vx_grid[1], vx_grid[-2])
    xlim = (x_grid[1], x_grid[-2])
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    output_fig = params['figures_folder'] + 'connection_probabilities_%d_wsx%.2e_wsv%.2e.png' % (gid, params['w_sigma_x'], params['w_sigma_v'])
    print 'Saving figure to:', output_fig
    pylab.savefig(output_fig, dpi=300)




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
#    pylab.show()
