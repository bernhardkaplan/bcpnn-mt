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

from Network_PyNEST_weight_analysis_after_training import WeightAnalyser

#ps = simulation_parameters.parameter_storage()
#params = ps.params

def plot_contour_connectivity_tgt(params, adj_list, tp, gid, plot_delay=False, clim=None):
    """
    the adj_list is assumed to be the target - indexe adjacency list
    """

    keys_are_int = False
    try:
        assert (gid in adj_list.keys()), 'ERROR: gid not found in the adjacency list provided'
        gid_key = gid
    except:
        assert (str(gid) in adj_list.keys()), 'ERROR: gid not found in the adjacency list provided'
        gid_key = str(gid)
    sources_weights = np.array(adj_list[gid_key])
    source_gids = sources_weights[:, 0].astype(int) - 1 # - 1 because it's NEST
    weights = sources_weights[:, 1]
    delays = np.ones(weights.size)
    x_tgt = tp[source_gids, 0]
    vx_tgt = tp[source_gids, 2]
    tp_cell = tp[int(gid_key) - 1, :]

    autolimit = False
    if autolimit == True:
        x_min, x_max = .9 * x_tgt.min(), 1.1 * x_tgt.max()
        if np.sign(vx_tgt.min()) == -1:
            vx_min = 1.1 * vx_tgt.min()
        else:
            vx_min = .9 * vx_tgt.min()
        vx_max = 1.3 * vx_tgt.max()
    else:
        ylim = (params['v_min_tp'], params['v_max_tp'])
        xlim = (0.0, 1.0)
        x_min, x_max = xlim[0], xlim[1]
        vx_min, vx_max = ylim[0], ylim[1]

    granularity = 0.06 # should not be between 0.02 and 0.1
    dx = granularity * (x_max - x_min)
    dvx = granularity * (vx_max - vx_min)
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
                'figure.subplot.left':.15,
                'figure.subplot.bottom':.08,
                'figure.subplot.right':.95,
                'figure.subplot.top':.90, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)
    fig = pylab.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

#    cmap = pylab.get_cmap('jet')
    cmap = pylab.get_cmap('bwr')
    if clim != None:
        levels = np.around(np.linspace(clim[0], clim[1], 200), decimals=1)
    else:
        levels = 100
    CS = ax.contourf(x_grid + dx / 2.,
                vx_grid + dvx / 2., z_data, levels, \
                extend='both',
                  cmap=cmap)

#    CS = ax.contourf(x_grid + dx / 2.,
#                vx_grid + dvx / 2., z_data, n_levels, \
#                  cmap=cmap)

    # use weights as dot sizes
    markersize_cell = 30
    markersize_min = 2
    markersize_max = 12
    if plot_delay:
#        markersizes = utils.transform_linear(abs(weights), markersize_min, markersize_max)
#        markersizes = utils.transform_linear(delays, markersize_min, markersize_max)
#        markersizes = utils.transform_linear(np.abs(weights), 0., abs_max)
        markersizes = utils.transform_linear(np.abs(weights), 0, markersize_max)
        print 'debug', markersizes, weights
        norm = matplotlib.colors.Normalize(vmin=-abs_max, vmax=abs_max)
        abs_max = max(abs(weights.min()), weights.max())
        norm = matplotlib.colors.Normalize(vmin=delays.min(), vmax=delays.max())
#        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.binary) # large delays -- black, short delays -- white
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.bwr) # large delays -- black, short delays -- white
#        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.bone) # large delays -- bright, short delays -- black
        m.set_array(delays)
        rgba_colors = m.to_rgba(delays)

    plot_weights = not plot_delay
    if plot_weights:
        markersizes = utils.transform_linear(np.abs(weights), markersize_min, markersize_max)

        abs_max = max(abs(weights.min()), weights.max())
        norm = matplotlib.colors.Normalize(vmin=-abs_max, vmax=abs_max)
#        norm = matplotlib.colors.Normalize(vmin=weights.min(), vmax=weights.max())

#        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.bone) # large weights -- bright, small weights -- black
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.bwr) # large weights -- black, small weights -- white
#        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.binary) # large weights -- black, small weights -- white
#        m.set_array(np.abs(weights))
        m.set_array(weights)
        rgba_colors = m.to_rgba(weights)

    for i_, tgt in enumerate(source_gids):
        ax.plot(x_tgt[i_], vx_tgt[i_], 'o', markeredgewidth=0, c=rgba_colors[i_], markersize=markersizes[i_])
    ax.plot(tp_cell[0], tp_cell[2], '*', markersize=markersize_cell, c='y', markeredgewidth=1, label='source', zorder=source_gids.size + 10)
    print 'DEBUG cell', tp_cell[0], tp_cell[2]
    ax.legend(numpoints=1)

    # set colorbars and labels
    cbar_prob = fig.colorbar(CS)
    cbar_prob.set_label('Connection strength')

    if plot_delay:
        cbar_delay = fig.colorbar(m)
        cbar_delay.set_label('Delay [ms]')

    if plot_weights:
        cbar_weight = fig.colorbar(m)
        cbar_weight.set_label('Weight [a.u.]')

    # autoscale figure limits
    if autolimit:
        ylim = (vx_edges[1], vx_edges[-2])
        xlim = (x_edges[1], x_edges[-2])
    else:
        ylim = (-3.5, 4)
        xlim = (0.6, 0.72)
    output_fig = params['figures_folder'] + 'contour_connectivity_gid%d_taui%d_gran%.2f.png' % (gid, params['taui_bcpnn'], granularity)

    ylim = (.45 * vx_edges[1], vx_edges[-2])
    xlim = (x_edges[1], x_edges[-2])
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    print 'DEBUG xlim ylim', ax.get_xlim(), ax.get_ylim()

    ax.set_xlabel('Receptive field position $x$')
    ax.set_ylabel('Preferred speed $v_x$')
    title = 'Distribution of incoming connections for \none example source cell, $\\tau_{z_{i}}= %d$ [ms]\n' % params['taui_bcpnn']
#    title += '$\sigma_X = %.2f\quad\sigma_V=%.2f$' % (params['w_sigma_x'], params['w_sigma_v'])
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







def run_contour_plot_tgt_perspective(params, mp=None):

    tp = np.loadtxt(params['tuning_prop_exc_fn']) #load the data and merge files before if necessary
    adj_list_fn = params['adj_list_tgt_fn_base'] + 'merged.json'
#    if not os.path.exists(adj_list_fn):
#        adj_list_src_index = utils.convert_adjacency_lists(params)

    WA = WeightAnalyser(params)
    adj_list = WA.load_adj_lists(src_tgt='tgt')

    gid = None
    if gid == None:
        gid = 1 + utils.select_well_tuned_cells_1D(tp, mp, 1, w_pos=1.)[0]
        print 'Plotting gid:', gid
    clim = (-20, 20)
#    clim = None
    plot_contour_connectivity_tgt(params, adj_list, tp, gid, clim=clim) # connection weights laid out in the tuning space and put on a grid --> contour




def run_contour_plot_src_perspective(params, mp=None):

    tp = np.loadtxt(params['tuning_prop_exc_fn']) #load the data and merge files before if necessary
    adj_list_fn = params['adj_list_src_fn_base'] + 'merged.json'
    if not os.path.exists(adj_list_fn):
        adj_list_src_index = utils.convert_adjacency_lists(params)
    WA = WeightAnalyser(params)
    adj_list = WA.load_adj_lists(src_tgt='src')

#    conn_list = utils.convert_adjacency_list_to_connlist(adj_list, src_tgt='tgt')

#    d = WA.get_weight_matrix_src_index(plot=False, output_fn=None)
    # determine which cell to plot
#    plot_source_perspective = False
#    gid = 1
    gid = None
    if gid == None:
        gid = 1 + utils.select_well_tuned_cells_1D(tp, mp, 1, w_pos=1.)[0]
        print 'Plotting gid:', gid
    clim = (-20, 20)
    #clim = None
    plot_contour_connectivity_src(params, adj_list, tp, gid, clim=clim) # connection weights laid out in the tuning space and put on a grid --> contour



"""

def plot_contour_connectivity_src(params, adj_list, tp, gid, plot_delay=False, clim=None):

    keys_are_int = False
    try:
        assert (gid in adj_list.keys()), 'ERROR: gid not found in the adjacency list provided'
        gid_key = gid
    except:
        assert (str(gid) in adj_list.keys()), 'ERROR: gid not found in the adjacency list provided'
        gid_key = str(gid)
    targets_weights = np.array(adj_list[gid_key])
    target_gids = targets_weights[:, 0].astype(int) - 1 # - 1 because it's NEST
    weights = targets_weights[:, 1]
    delays = np.ones(weights.size)
    x_tgt = tp[target_gids, 0]
    vx_tgt = tp[target_gids, 2]
    tp_cell = tp[gid_key, :]

    autolimit = False
    if autolimit == True:
        x_min, x_max = .9 * x_tgt.min(), 1.1 * x_tgt.max()
        if np.sign(vx_tgt.min()) == -1:
            vx_min = 1.1 * vx_tgt.min()
        else:
            vx_min = .9 * vx_tgt.min()
        vx_max = 1.3 * vx_tgt.max()
    else:
        ylim = (-.1, tp_cell[2] + 1.5)
#        xlim = (tp_cell[0] - 1.0, tp_cell[0] + 1.0]
        xlim = (0.0, 1.0)
        x_min, x_max = xlim[0], xlim[1]
        vx_min, vx_max = ylim[0], ylim[1]

    granularity = 0.08 # should not be too small, i.e. > 0.02
    dx = granularity * (x_max - x_min)
    dvx = granularity * (vx_max - vx_min)
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
                'figure.subplot.left':.15,
                'figure.subplot.bottom':.08,
                'figure.subplot.right':.95,
                'figure.subplot.top':.90, 
                'lines.markeredgewidth' : 0}
    pylab.rcParams.update(rcParams)
    fig = pylab.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    z_data = z_data

#    if clim != None:
#        cmap_name = 'bwr'
#        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])#, clip=True)
#        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
#        m.set_array(np.arange(clim[0], clim[1], 0.01))

#    cmap = pylab.get_cmap('jet')
    cmap = pylab.get_cmap('bwr')
    if clim != None:
        levels = np.arange(clim[0], clim[1], 0.01)
    else:
        levels = 100
    CS = ax.contourf(x_grid + dx / 2.,
                vx_grid + dvx / 2., z_data, levels, \
                extend='both',
                  cmap=cmap)
#    CS = ax.contourf(x_grid + dx / 2.,
#                vx_grid + dvx / 2., z_data, n_levels, \
#                  cmap=cmap)

    # use weights as dot sizes
    markersize_cell = 35
    markersize_min = 2
    markersize_max = 12
    if plot_delay:
        markersizes = utils.transform_linear(delays, markersize_min, markersize_max)
        norm = matplotlib.colors.Normalize(vmin=delays.min(), vmax=delays.max())
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.binary) # large delays -- black, short delays -- white
#        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.bone) # large delays -- bright, short delays -- black
        m.set_array(delays)
        rgba_colors = m.to_rgba(delays)

    plot_weights = not plot_delay
    if plot_weights:
#        markersizes = utils.transform_linear(abs(weights), markersize_min, markersize_max)
        abs_max = max(abs(weights.min()), weights.max())
        markersizes = utils.transform_linear(weights, 0., abs_max)
        print 'debug', markersizes, weights
        norm = matplotlib.colors.Normalize(vmin=-abs_max, vmax=abs_max)
#        norm = matplotlib.colors.Normalize(vmin=weights.min(), vmax=weights.max())

        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.bwr) # large weights -- black, small weights -- white
#        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.binary) # large weights -- black, small weights -- white
        m.set_array(np.abs(weights))
        rgba_colors = m.to_rgba(np.abs(weights))

    for i_, tgt in enumerate(target_gids):
        ax.plot(x_tgt[i_], vx_tgt[i_], 'o', markeredgewidth=0, c=rgba_colors[i_], markersize=markersizes[i_])
    ax.plot(tp_cell[0], tp_cell[2], '*', markersize=markersize_cell, c='y', markeredgewidth=1, label='source', zorder=target_gids.size + 10)
    print 'DEBUG cell', tp_cell[0], tp_cell[2]
    ax.legend(numpoints=1)

    # set colorbars and labels
    cbar_prob = fig.colorbar(CS)
    cbar_prob.set_label('Connection strength')

    if plot_delay:
        cbar_delay = fig.colorbar(m)
        cbar_delay.set_label('Delay [ms]')

    if plot_weights:
        cbar_weight = fig.colorbar(m)
        cbar_weight.set_label('Weight [a.u.]')

    # autoscale figure limits
    if autolimit:
        ylim = (vx_edges[1], vx_edges[-2])
        xlim = (x_edges[1], x_edges[-2])
    else:
        ylim = (-3.5, 4)
        xlim = (0.6, 0.72)
    output_fig = params['figures_folder'] + 'contour_connectivity_gid%d_taui%d.png' % (gid, params['taui_bcpnn'])

    ylim = (.45 * vx_edges[1], vx_edges[-2])
    xlim = (x_edges[1], x_edges[-2])
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    print 'DEBUG xlim ylim', ax.get_xlim(), ax.get_ylim()

    ax.set_xlabel('Receptive field position $x$')
    ax.set_ylabel('Preferred speed $v_x$')
    title = 'Distribution of outgoing connections for \none example source cell, $\\tau_{z_{i}}= %d$ [ms]\n' % params['taui_bcpnn']
#    title += '$\sigma_X = %.2f\quad\sigma_V=%.2f$' % (params['w_sigma_x'], params['w_sigma_v'])
    ax.set_title(title)

    print 'Saving figure to:', output_fig
    pylab.savefig(output_fig, dpi=200)

"""

def run_contour_plot(params, source_perspective=False, mp=None):
    if source_perspective:
        run_contour_plot_src_perspective(params, mp)
    else:
        run_contour_plot_tgt_perspective(params, mp)


if __name__ == '__main__':

    mp = [0.5, 0., 0.5, .0]
    plot_source_perspective = False
    np.random.seed(0)
    if len(sys.argv) == 1:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        run_contour_plot(params, source_perspective=plot_source_perspective, mp=mp)
    elif len(sys.argv) == 2:
        params = utils.load_params(sys.argv[1])
        run_contour_plot(params, source_perspective=plot_source_perspective, mp=mp)
    else:
        for folder in sys.argv[1:]:
            params = utils.load_params(folder)
            run_contour_plot(params, source_perspective=plot_source_perspective, mp=mp)



#    random.seed(0)
#    if not os.path.exists(params['merged_conn_list_ee']):
#        print 'Running merge_connlists.py...'
#        os.system('python merge_connlists.py %s' % params['folder_name'])
#    print 'Loading connection file ...', params['merged_conn_list_ee']
#    d = np.loadtxt(params['merged_conn_list_ee'])
#    gid = None
#    i_, dx = 0, .05
#    x_start = 0.2
#    while gid == None:
#        mp_for_cell_sampling = [(x_start + dx * i_) % 1., 0.0, 1.0, 0.]
#        gid = utils.select_well_tuned_cells_1D(tp, mp_for_cell_sampling, 1)
#        connections = utils.get_targets(d, gid)
#        connection_gids = connections[:, 1].astype(int)
#        print 'GID:', gid, i_, mp_for_cell_sampling

#        if len(connection_gids) == 0:
#            print 'GID %d connects to NO CELLS' % gid
#            gid = None
#        else:
#            plot_formula(params, d, tp, gid, plot_source_perspective=plot_source_perspective) # plot the analytically expected weights and the actual connections in tuning space
#        i_ += 1

#    pylab.show()
