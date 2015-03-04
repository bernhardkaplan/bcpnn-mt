import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np 
import utils
import json
import simulation_parameters
import functions

plot_params = {'backend': 'png',
              'axes.labelsize': 24,
              'axes.titlesize': 24,
              'text.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 14,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 1,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.15,
              'figure.subplot.bottom':.14,
              'figure.subplot.right':.90,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}

pylab.rcParams.update(plot_params)


def get_gids_to_mc(params, pyr_gid):
    """
    Return the HC, MC within the HC in which the cell with pyr_gid is in
    and the min and max gid of pyr cells belonging to the same MC.
    """
    mc_idx = pyr_gid / params['n_exc_per_mc']
    hc_idx = mc_idx / params['n_mc_per_hc']
    gid_min = mc_idx * params['n_exc_per_mc']
    gid_max = (mc_idx + 1) * params['n_exc_per_mc']  # here no +1 because it's used for randrange and +1 would include a false cell
    return (hc_idx, mc_idx, gid_min, gid_max)


def get_avg_xpos(params, xpos):
    """
    xpos is col 0 from the tuning prop file (containing all the xpos for all cells)
    return the average position for the cells belonging to one minicolumn
    """
    assert params['n_exc_per_mc'] * params['n_mc'] == xpos.size, 'Wrong dimensions in provided tuning properties and params dictionary'
    avg_pos = np.zeros(params['n_mc'])
    cnt_cells= np.zeros(params['n_mc'])
    for i_ in xrange(xpos.size):
        (hc_idx, mc_idx, gid_min, gid_max) = get_gids_to_mc(params, i_)
        avg_pos[mc_idx] += xpos[i_] # position
        cnt_cells[mc_idx] += 1
    for i_mc in xrange(params['n_mc']):
        # check if gid - mc mapping was correctly
        assert cnt_cells[i_mc] == params['n_exc_per_mc']
        avg_pos[i_mc] /= cnt_cells[i_mc]
    return avg_pos



def plot_connections_out_fixed_taui(argv, tau_i):
    """
    Plot weight between minicolumns versus position distance for a fixed tau_i.
    Speeds will be color-coded

    argv -- should be a list of Folders containing the training data
    """

    speed_range = [0.1, 1.0]

    fig2 = pylab.figure(figsize=(12, 12))
    ax2 = fig2.add_subplot(111)

    fig = pylab.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    vrange = np.arange(speed_range[0], speed_range[1], 0.01)
    clim = (np.min(vrange), np.max(vrange))
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(np.array(vrange))

    ms_min = 1
    ms_max = 10

    speed_str = ''
    for folder in argv:
        params = utils.load_params(folder)
        speed_str += '%.1f_' % params['v_min_tp']
        tp = np.loadtxt(params['tuning_prop_exc_fn'])
        conn_fn = params['merged_conn_list_ee']
        tau_i = params['bcpnn_params']['tau_i']
        v_plot_range = 0.1 # plot only connections whose src/tgt pairs have speeds not more dissimilar than v_plot_range
        v_range = (np.min(np.abs(tp[:, 2])), np.max(np.abs(tp[:, 2])))
#        v_range = (np.min(tp[:, 2]), np.max(tp[:, 2]))

        print 'Loading conn list:', conn_fn, 
        conn_list = np.loadtxt(conn_fn)
        print 'done'

        pos_idx = np.where(tp[:, 2] > 0)[0]
        d_out = ''
        cnt_ = 0
        for i_src, src in enumerate(pos_idx):
#        for i_src, src in enumerate(pos_idx[:10]):
            v_src = tp[src, 2]
            print 'v_src:', v_src, 'gid', src, i_src, ' / ', pos_idx.size
            conn_idx = np.where(conn_list[:, 0] == src + 1)[0]
            tgts = np.array(conn_list[conn_idx, 1] - 1, dtype=np.int)
            wijs = conn_list[conn_idx, 2]
            v_tgts = tp[tgts, 2]
            valid_conn_idx = np.where(np.abs(v_src - v_tgts) < v_plot_range)[0]
            for idx in valid_conn_idx:
                tgt = tgts[idx]
                wij = wijs[idx]
                d_out += '%.3e\t%.3e\t%.3e\t%.3e\t%d\n' % (tp[tgt, 0] - tp[src, 0], wij, tp[src, 2], tp[tgt, 2], tau_i)
                cnt_ += 1

        fn_out = 'xdiff_vs_wij_taui%d_vmintp%.1f.dat' % (tau_i, params['v_min_tp'])
        f = file(fn_out, 'w')
        print 'Writing to file:', fn_out
        f.write(d_out)
        f.flush()
        f.close()
        print 'Cnt:', cnt_
        print 'guess:', pos_idx.size **2
        d = np.loadtxt(fn_out)
        colors = m.to_rgba(d[:, 2])
        ax.scatter(d[:, 0], d[:, 1], c=colors, linewidths=0)

        # get the average wij for each xdiff --> bin the data

        n_bins = 20 + 1 # + 1 to get the 1.0 as last bin-edge
        bin_edges = np.linspace(-1, 1, n_bins, endpoint=True)
        binned_array = utils.bin_array(d[:, 0], bin_edges)
        average_wij_vs_xdiff = np.zeros((n_bins, 2))
        for i_bin in xrange(bin_edges.size):
            bin_idx = np.where(binned_array == i_bin)[0]
            if bin_idx.size > 0:
                average_wij_vs_xdiff[i_bin, :] = d[bin_idx, 0].mean(), d[bin_idx, 1].mean()
#        color = m.to_rgba(params['v_min_tp'])
        color = m.to_rgba(d[:, 2].mean())
        ax2.plot(average_wij_vs_xdiff[1:, 0], average_wij_vs_xdiff[1:, 1], '-o', lw=3, c=color)

    xlim = ax.get_xlim()
    ax.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=2)
    ylim = ax.get_ylim()
    ax.plot((0., 0.), (ylim[0], ylim[1]), '--', c='k', lw=2)
    title = 'Outgoing weights vs target position'
    title += '\n $\\tau_{i}=%d$' % (tau_i)
    ax.set_title(title)
    ax.set_ylabel('$w_{out}$')
    ax.set_xlabel('Distance between source and target neuron')
    cb = fig.colorbar(m)
    cb.set_label('$v_{src}$')
    output_fn = 'bcpnn_weights_vs_pos_taui%04d_%s.png' % (tau_i, speed_str)
    print 'Saving fig to:', output_fn
    fig.savefig(output_fn, dpi=200)

    xlim = ax2.get_xlim()
    ax2.plot((xlim[0], xlim[1]), (0., 0.), '--', c='k', lw=2)
    ylim = ax2.get_ylim()
    ax2.plot((0., 0.), (ylim[0], ylim[1]), '--', c='k', lw=2)
    title = 'Outgoing weights vs target position'
    title += '\n $\\tau_{i}=%d$' % (tau_i)
    ax2.set_title(title)
    ax2.set_ylabel('$w_{out}$')
    ax2.set_xlabel('Distance between source and target neuron')
    cb = fig2.colorbar(m)
    cb.set_label('$v_{src}$')
    output_fn = 'bcpnn_weights_vs_pos_taui%04d_%s_averaged.png' % (tau_i, speed_str)
    print 'Saving fig to:', output_fn
    fig2.savefig(output_fn, dpi=200)

if __name__ == '__main__':

    show = True
    colorcoded = 'tau_i' # plot a fixed speed tuning
    colorcoded = 'speed' # plot fixed taui

    folder_lists = {}
    # folders ordered after tau_i

    folder_lists[5] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    folder_lists[20] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    folder_lists[50] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    folder_lists[100] = [ 'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    folder_lists[150] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']
    
    folder_lists[200] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    folder_lists[250] = [
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']
    
    folder_lists[500] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    folder_lists[1000] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.20',
            'TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.10',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    # folders ordered after vmin_tp (cells only have this preferred speed)
    folder_lists['0.3'] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.30']

    folder_lists['0.4'] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.40']

    folder_lists['0.5'] = ['TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui10_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.50']

    folder_lists['0.6'] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.60']

    folder_lists['0.7'] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70', 
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.70']

    folder_lists['0.8'] = ['TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui10_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80',
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.80']

    folder_lists['0.9'] = ['TrainingSim_ClusternoSTP__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90',
        'TrainingSim_ClusternoSTP__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp0.90']

    folder_lists['1.0'] = ['TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui1000_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui100_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui10_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui150_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui200_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui20_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui250_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui500_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui50_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00', 
        'TrainingSim_ClusterwithSTP_unspecBlank__1x100x1_0-100_taui5_nHC20_nMC2_blurXV_0.00_0.00_pi1.0e-04_vmintp1.00']

    
    if colorcoded == 'speed':
        tau_i = 250
        plot_connections_out_fixed_taui(folder_lists[tau_i], tau_i)
    else:
        speed = 0.9
        plot_connections_out_fixed_speed(folder_list, speed)

    if show:
        pylab.show()



