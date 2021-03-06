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
import matplotlib
from matplotlib import cm

#ps = simulation_parameters.parameter_storage()
#params = ps.params

def plot_contour_connectivity(params, d, tp, gid):
    targets = utils.get_targets(d, gid)
    target_gids = targets[:, 1].astype(int)
    weights = targets[:, 2]
    delays = targets[:, 3]
    x_tgt = tp[target_gids, 0]
    vx_tgt = tp[target_gids, 2]

    autolimit = True
    if autolimit:
        x_min, x_max = .9 * x_tgt.min(), 1.1 * x_tgt.max()
        if np.sign(vx_tgt.min()) == -1:
            vx_min = 1.1 * vx_tgt.min()
        else:
            vx_min = .9 * vx_tgt.min()
        vx_max = 1.3 * vx_tgt.max()
    else:
        ylim = (-3.5, 4)
        xlim = (0.0, 1.0)
        x_min, x_max = xlim[0], xlim[1]
        vx_min, vx_max = ylim[0], ylim[1]

    dx = 0.04 * (x_max - x_min)
    dvx = 0.04 * (vx_max - vx_min)
    x_grid, vx_grid = np.mgrid[slice(x_min, x_max, dx), 
                    slice(vx_min, vx_max, dvx)]

    x_edges = x_grid[:, 0]
    vx_edges = vx_grid[0, :]
    z_data = np.zeros(x_grid.shape)
    for i_, w in enumerate(weights):
        x0, v0 = x_tgt[i_], vx_tgt[i_]
        (x_, y_) = utils.get_grid_pos(x0, v0, x_edges, vx_edges)
        z_data[x_, y_] += w


    rcParams = { 'axes.labelsize' : 24,
                'axes.titlesize'  : 24,
                'label.fontsize': 24,
                'xtick.labelsize' : 24, 
                'ytick.labelsize' : 24, 
                'legend.fontsize': 20, 
                'figure.subplot.left':.10,
                'figure.subplot.bottom':.08,
                'figure.subplot.right':.95,
                'figure.subplot.top':.90, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)
    fig = pylab.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    z_data = z_data
    cmap = pylab.get_cmap('jet')
    n_levels = 200
    CS = ax.contourf(x_grid + dx / 2.,
                vx_grid + dvx / 2., z_data, n_levels, \
                  cmap=cmap)

    # use weights as dot sizes
    markersize_cell = 15
    markersize_min = 4
    markersize_max = 10
    markersizes = utils.linear_transformation(weights, markersize_min, markersize_max)
    norm = matplotlib.colors.Normalize(vmin=delays.min(), vmax=delays.max())
#    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.bone) # large delays -- bright, short delays -- black
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.binary) # large delays -- black, short delays -- white
    m.set_array(delays)
    rgba_colors = m.to_rgba(delays)
    for i_, tgt in enumerate(targets):
        ax.plot(x_tgt[i_], vx_tgt[i_], 'o', markeredgewidth=0, c=rgba_colors[i_], markersize=markersizes[i_])
    ax.plot(tp[gid, 0], tp[gid, 2], '*', markersize=markersize_cell, c='y', markeredgewidth=0, label='source')
    ax.legend(numpoints=1)

    # set colorbars and labels
    cbar_prob = fig.colorbar(CS)
    cbar_delay = fig.colorbar(m)
    cbar_prob.set_label('Connection probability')
    cbar_delay.set_label('Delay [ms]')

    # autoscale figure limits
    if autolimit:
        ylim = (vx_edges[1], vx_edges[-2])
        xlim = (x_edges[1], x_edges[-2])
        output_fig = params['figures_folder'] + 'contour_connectivity_%d_wsx%.2e_wsv%.2e_a.png' % (gid, params['w_sigma_x'], params['w_sigma_v'])
    else:
        ylim = (-3.5, 4)
        xlim = (0.6, 0.72)
        output_fig = params['figures_folder'] + 'contour_connectivity_%d_wsx%.2e_wsv%.2e.png' % (gid, params['w_sigma_x'], params['w_sigma_v'])

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
    pylab.savefig(output_fig, dpi=200)



def get_pconn_source_perspective(params, tp, src_gid, x_tgt, vx_tgt):
    """To which cells should src_gid connect to?"""
    n_tgt = x_tgt.size
    tau_prediction = params['tau_prediction'] #/ params['t_stimulus']
    x_predicted = ((tp[src_gid, 0] + (tau_prediction) * tp[src_gid, 2]) % 1) * np.ones(n_tgt)
#    tau_shift = params['sensory_delay']
    # compute where the cell projects to (preferentially)
#    x_predicted = ((tp[src_gid, 0] + (tau_prediction + tau_shift) * tp[src_gid, 2]) % 1) * np.ones(n_tgt)
#    print 'debug', tp[src_gid, 0], x_predicted[0], tp[src_gid, 2] * (tau_prediction + tau_shift)
    d_pred_tgt = utils.torus_distance_array(x_predicted, x_tgt)
    z = np.exp(- d_pred_tgt**2 / (2 * params['w_sigma_x']**2)) \
            * np.exp(- ((tp[src_gid, 2] - vx_tgt)**2/ (2 * params['w_sigma_v']**2)))
    return z


def get_pconn_target_perspective(params, tp, gid, x_src, vx_src):
    """
    From which cells does cell gid get input from?
    This is as the simulation code works.
    """

    n_src = x_src.size
    tau_prediction = params['tau_prediction'] #/ params['t_stimulus']
    tau_shift = params['sensory_delay']
    x_predicted = (x_src + (tau_prediction + tau_shift) * vx_src) % 1.
    # calculate the distance between the predicted position and the target cell
    d_pred_tgt = utils.torus_distance_array(x_predicted, tp[gid, 0] * np.ones(n_src))
    # take the preferred speeds into account 
    v_tuning_diff = vx_src - tp[gid, 2] * np.ones(n_src)
    z = np.exp(- d_pred_tgt**2 / (2 * params['w_sigma_x']**2)) \
            * np.exp(- (v_tuning_diff**2 / (2 * params['w_sigma_v']**2)))
    # latency is computed based on src-tgt distance only --- not taking the tau_shift and tau_prediction into account
#    d_ij = utils.torus_distance_array(tp[:, 0], tp[gid, 0] * np.ones(n_src))
#    latency = d_ij / np.abs(vx_src)
    return z




def plot_formula(params, d, tp, gid, plot_source_perspective=False):

    print 'Plotting connectivity formula as contour plot ...'
    # get target cells and connection weights
    if plot_source_perspective:
        connections = utils.get_targets(d, gid)
        connection_gids = connections[:, 1].astype(int)
        print 'Cell %d is projecting to: ' % gid, connection_gids
        title = 'Distribution of outward connection probabilities'
    else:
        connections = utils.get_sources(d, gid)
        connection_gids = connections[:, 0].astype(int)
        print 'Cell %d is receiving input from: ' % gid, connection_gids
        title = 'Distribution of incoming connection probabilities'
    weights = connections[:, 2]
    delays = connections[:, 3]
    x_conn = tp[connection_gids, 0]
    vx_conn = tp[connection_gids, 2]

    print 'DEBUG weights:', weights, weights.sum()
    print 'DEBUG x_conn:', x_conn
    print 'DEBUG vx_conn:', vx_conn

    # # # # # # # # # # # # 
    # plot the formula 
    # # # # # # # # # # # # 
    n_pts = 500000
    autolimit = True
    if autolimit:
        x_min, x_max = .0, 0.6
#        x_min, x_max = .0, 1.
        vx_min = min(.7 * tp[gid, 2], np.min(.7 * vx_conn))
        vx_max = max(1.2 * tp[gid, 2], np.max(1.2 * vx_conn))

        print 'x_min', x_min
        print 'x_max', x_max
        print 'vx_min', vx_min
        print 'vx_max', vx_max

    dx = 0.04 * (x_max - x_min)
    dvx = 0.04 * (vx_max - vx_min)
    x_sample = np.random.uniform(x_min, x_max, n_pts)
    vx_sample = np.random.uniform(vx_min, vx_max, n_pts)

    if plot_source_perspective:
        z = get_pconn_source_perspective(params, tp, gid, x_sample, vx_sample)
        cell_label = 'source GID' #=%d' % (gid)
    else:
        # plot as the simulation code works
        z = get_pconn_target_perspective(params, tp, gid, x_sample, vx_sample)
        cell_label = 'target GID' #=%d' % (gid)
    
    clip_formula_at_connradius = False
    if clip_formula_at_connradius:
        # apply connection radius
        latency = d_ij / np.abs(tp[gid, 2])
        invalid_idx = np.nonzero(latency * params['delay_scale'] > params['delay_range'][1])[0]
        invalid_idx = np.nonzero(latency * params['delay_scale'] < params['delay_range'][0])[0]
        print 'DEBUG: # of neurons with a too short latency = ', invalid_idx.size, 'total number of source = ', tp_src[:, 0].size
        z[invalid_idx] = 0.
        invalid_idx = np.nonzero(latency * params['delay_scale'] > params['delay_range'][1])[0]
        print 'DEBUG: # of neurons with a too long latency = ', invalid_idx.size, 'total number of source = ', tp_src[:, 0].size
        z[invalid_idx] = 0.

    x_grid = np.arange(x_min, x_max, dx) 
    vx_grid = np.arange(vx_min, vx_max, dvx)
    # grid the data
    z_grid = griddata(x_sample, vx_sample, z, x_grid, vx_grid)#, interp='linear')
    n_levels = 300

    rcParams = { 'axes.labelsize' : 32,
                'axes.titlesize'  : 32,
                'label.fontsize': 20,
                'xtick.labelsize' : 24, 
                'ytick.labelsize' : 24, 
                'legend.fontsize': 18, 
                'figure.subplot.left':.15,
#                'figure.subplot.bottom':.10,
#                'figure.subplot.right':.95,
#                'figure.subplot.top':.90, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)
    fig = pylab.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    CS = ax.contourf(x_grid, vx_grid, z_grid, n_levels, cmap=pylab.cm.jet,
                      vmax=abs(z_grid).max(), vmin=-abs(z_grid).max())
    
    # # # # # # # # # # # # 
    # plot the connected cells
    # # # # # # # # # # # # 
    # use weights as dot sizes
    fontsize = 24
    markersize_cell = 25
    markersize_min = 4
    markersize_max = 10
    markersize_others = 1
    markersizes = utils.linear_transformation(weights, markersize_min, markersize_max)
    norm = matplotlib.colors.Normalize(vmin=delays.min(), vmax=delays.max())
    # cmap == bone: large delays -- bright, short delays -- black
    delay_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.binary) # cmap == binary: large delays -- black, short delays -- white
    delay_map.set_array(delays)
    rgba_colors = delay_map.to_rgba(delays)
    for i_, tgt in enumerate(connections):
        ax.plot(x_conn[i_], vx_conn[i_], 'o', markeredgewidth=0, c=rgba_colors[i_], markersize=markersizes[i_])

    # check where other cells are situated
    ax.plot(tp[:, 0], tp[:, 2], 'o', markeredgewidth=1, c='k', markersize=markersize_others)

    # plot the center of mass for all connections
    cms = utils.get_connection_center_of_mass(connection_gids, weights, tp)
    print 'CMS-x:', cms[0], 'cell x-pos:', tp[gid, 0]
    print 'CMS-vx:', cms[1], 'cell vx:', tp[gid, 2]
    ax.plot(cms[0], cms[1], 'D', markeredgewidth=1, markersize=np.int(markersize_cell/2 + 1), c='g', label='Connection center-of-mass')
    
    ax.plot(tp[gid, 0], tp[gid, 2], '*', markeredgewidth=1, markersize=markersize_cell, c='y', label=cell_label)
    ax.legend(numpoints=1)
    ax.set_xlabel('Receptive field position $x$')
    ax.set_ylabel('Preferred speed $v_x$')
    title += '\n%s $\sigma_X = %.2f\quad\sigma_V=%.2f$' % (params['IJCNN_code'], params['w_sigma_x'], params['w_sigma_v'])
    ax.set_title(title)
    cbar_prob = fig.colorbar(CS)
    cbar_prob.set_label('Connection probability')
#    cbar_delay = fig.colorbar(delay_map)
#    cbar_delay.set_label('Delay [ms]')

#    ylim = (vx_min, vx_max)
    ylim = (vx_grid[1], vx_grid[-1])
    xlim = (x_grid[1], x_grid[-1])
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    if clip_formula_at_connradius:
        output_fig = params['figures_folder'] + 'connection_probabilities_%d_wsx%.2e_wsv%.2e_clipped.png' % (gid, params['w_sigma_x'], params['w_sigma_v'])
    else:
        output_fig = params['figures_folder'] + 'connection_probabilities_%d_wsx%.2e_wsv%.2e.png' % (gid, params['w_sigma_x'], params['w_sigma_v'])
    print 'Saving figure to:', output_fig
    pylab.savefig(output_fig, dpi=200)




if __name__ == '__main__':

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
    else:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params

    # determine which cell to plot
    plot_source_perspective = True # if False --> it's like the simulation code
    random.seed(0)
    tp = np.loadtxt(params['tuning_prop_means_fn'])
    # load the data and merge files before if necessary
    if not os.path.exists(params['merged_conn_list_ee']):
        print 'Running merge_connlists.py...'
        os.system('python merge_connlists.py %s' % params['folder_name'])
    print 'Loading connection file ...', params['merged_conn_list_ee']
    d = np.loadtxt(params['merged_conn_list_ee'])
    gid = None
    i_, dx = 0, .05
    x_start = 0.2
    while gid == None:
        mp_for_cell_sampling = [(x_start + dx * i_) % 1., 0.0, 1.0, 0.]
        gid = utils.select_well_tuned_cells_1D(tp, mp_for_cell_sampling, 1)
        connections = utils.get_targets(d, gid)
        connection_gids = connections[:, 1].astype(int)
        print 'GID:', gid, i_, mp_for_cell_sampling

        if len(connection_gids) == 0:
            print 'GID %d connects to NO CELLS' % gid
            gid = None
        else:
            plot_formula(params, d, tp, gid, plot_source_perspective=plot_source_perspective) # plot the analytically expected weights and the actual connections in tuning space
        i_ += 1

#    plot_contour_connectivity(params, d, tp, gid) # connection weights laid out in the tuning space and put on a grid --> contour

#    pylab.show()
