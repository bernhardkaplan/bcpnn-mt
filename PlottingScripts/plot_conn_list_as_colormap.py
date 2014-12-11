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
#from matplotlib import cm

def plot_matrix(d, title=None, clim=None):

    cmap_name = 'bwr'
    if clim != None:
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])#, clip=True)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
        m.set_array(np.arange(clim[0], clim[1], 0.01))
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    print "plotting .... "
    if clim != None:
        cax = ax.pcolormesh(d, cmap=cmap_name, vmin=clim[0], vmax=clim[1])
    else:
        cax = ax.pcolormesh(d, cmap=cmap_name)

    ax.set_ylim((0, d.shape[0]))
    ax.set_xlim((0, d.shape[1]))
    if title != None:
        ax.set_title(title)
    pylab.colorbar(cax)
    return ax


def plot_conn_list(params, conn_list_fn, iteration=None, clim=None, conn_type='ee', compute_weights=False, sort_idx=None):
    """
    sort_idx -- 0 -> cells are sorted after x-position
                1 -> cells are sorted after y-position (only valid for 2D setup)
                2 -> cells are sorted after preferred x-speed
                3 -> cells are sorted after preferred y-speed
    """
    utils.merge_connection_files(params, conn_type, iteration=iteration)
    print 'Loading ', conn_list_fn
    data = np.loadtxt(conn_list_fn)
    src_min, src_max= np.min(data[:, 0]), np.max(data[:, 0])
    n_src = src_max - src_min + 1
    tgt_min, tgt_max= np.min(data[:, 1]), np.max(data[:, 1])
    n_tgt = tgt_max - tgt_min + 1
    conn_mat = np.zeros((n_src, n_tgt))
    print 'src_min, src_max', src_min, src_max
    print 'tgt_min, tgt_max', tgt_min, tgt_max
    for c in xrange(data[:,0].size):
        src = data[c, 0] - src_min
        tgt = data[c, 1] - tgt_min
        if compute_weights:
            conn_mat[src, tgt] = np.log(data[c, 5] / (data[c, 3] * data[c, 4]))
        else:
            conn_mat[src, tgt] = data[c, 2]
    
    if sort_idx != None:
        if conn_type[0] == 'e':
            tp = np.loadtxt(params['tuning_prop_exc_fn'])
        else:
            tp = np.loadtxt(params['tuning_prop_inh_fn'])
        idx = tp[:, sort_idx].argsort()
        conn_mat_sorted = conn_mat[idx, :]
        conn_mat = conn_mat_sorted

    if clim == 'symm':
        clim_max = max(np.abs(np.max(conn_mat)), np.abs(np.max(conn_mat)))
        clim = (-clim_max, clim_max)
        print 'clim:', clim

    ax = plot_matrix(conn_mat, clim=clim)
    print 'connmat min max', np.min(conn_mat), np.max(conn_mat)
    ax.set_title(conn_list_fn.rsplit("/")[-1])
    if params != None:
        if sort_idx != None:
            output_fn = params['figures_folder'] + conn_list_fn.rsplit("/")[-1].rsplit(".txt")[0] + "_cmap_sorted_by_tp.png"
        else:
            output_fn = params['figures_folder'] + conn_list_fn.rsplit("/")[-1].rsplit(".txt")[0] + "_cmap.png"
        title = output_fn.rsplit('/')[-1]
        ax.set_title(title)
        print "Saving fig to:", output_fn
        pylab.savefig(output_fn)
    return conn_mat


if __name__ == '__main__':

    """
    usage:
        python PlottingScripts/plot_conn_list_as_colormap.py Training_taup225_nStim1_nExcMpn800_nStates14_nActions15_it15-225_wMPN-BG3.50/Connections/merged_mpn_bg_d2_connection_dev_it*.txt
    OR 
        python PlottingScripts/plot_conn_list_as_colormap.py [FOLDER_NAME]
    """

    # TODO

    fns = sys.argv[1:] # 
    clim = None
#    clim = 'symm'
#    clim = (-0.5, .5)
#    clim = (-200., 200.)

    conn_type = 'ei'
#    if conn_type[0] == 'e':
#        sort_idx = 0 # or 2
#    else:
    sort_idx = None

    if len(sys.argv) > 2:
        print 'Case 1'
        for fn in fns:
            params = utils.load_params(fn)
            tgt_type = 'd1'
            conn_list_fn = params['merged_conn_list_%s' % conn_type]
            plot_conn_list(params, conn_list_fn, clim=clim, conn_type=conn_type, sort_idx=sort_idx)
    elif len(sys.argv) == 2:
        print 'Case 2'
#        params = utils.load_params(sys.argv[1])
        if sys.argv[1].endswith('.json') or os.path.isdir(sys.argv[1]):
            params = utils.load_params(sys.argv[1])
            conn_list_fn = params['merged_conn_list_%s' % conn_type]
            plot_conn_list(params, conn_list_fn, clim=clim, conn_type=conn_type, sort_idx=sort_idx)
        else:          
            print 'Please provide the folder / simulation_parameters.json file and not the conn_list.dat file!'
            exit(1)
    else:
        print 'Case 3: default parameters'
        import simulation_parameters
        GP = simulation_parameters.parameter_storage()
        params = GP.params
        conn_list_fn = params['merged_conn_list_%s' % conn_type]
        plot_conn_list(params, conn_list_fn, clim=clim, conn_type=conn_type, sort_idx=sort_idx)
    pylab.show()
