import numpy as np
import utils
import pylab
import sys
import os
import figure_sizes

params2 = { 'figure.figsize': (15, 10)}
pylab.rcParams.update(params2)

import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
pylab.rcParams['lines.markeredgewidth'] = 0


exc_cell = int(sys.argv[1])
if len(sys.argv) < 3:
    conn_mat_fn = params['weight_matrix_abstract']
else:
    conn_mat_fn = sys.argv[2]

if len(sys.argv) < 4:
    fig_fn = params['figures_folder'] + 'precomp_conn_profile_%d.png' % exc_cell
else:
    fig_fn = sys.argv[3]

if len(sys.argv) < 5:
    iteration = 0
else:
    iteration = int(sys.argv[4])

with_annotations = False
fig = pylab.figure()


x0, y0, u0, v0 = tp_exc[exc_cell, :]
print 'Loading connectivity matrix from:', conn_mat_fn
conn_mat = np.loadtxt(conn_mat_fn)
tgts_ee = np.arange(0, params['n_exc'], 1)
srcs_ee = np.arange(0, params['n_exc'], 1)

def plot_weights(subplot_code=111, relative=True):
    """
    relative means weights are scaled according to their group (exc / inh)
    if relative==False: weights are scaled in proportion to the maximum exc & inh weight
    """
    ax = fig.add_subplot(subplot_code)

    print "Plotting exc_cell -> E"

    w_abs_max = abs(conn_mat[exc_cell, :]).max()
#    w_min = conn_mat[exc_cell, :].min()
#    w_max = conn_mat[exc_cell, :].max()
    w_min = -1.
    w_max = 1.
    if w_max == 0:
        w_max = abs(w_min)
        print 'WARNING w_max == 0!'
        print 'Only negative weights!'

    print 'debug w_max:', w_max, 'w_mean', conn_mat[exc_cell, :].mean(), 'w_min', w_min
    ms_max = 4


    for i_, tgt in enumerate(tgts_ee):
        x_tgt, y_tgt, u_tgt, v_tgt = tp_exc[tgt, :]

        w = conn_mat[exc_cell, tgt]
        if w < 0:
            if relative==False:
                ms = abs(w) / w_abs_max * ms_max
            else:
                ms = w / w_min * ms_max
#            print 'w_inh ms:', ms
            color = 'b'
        else:
            if relative==False:
                ms = abs(w) / w_abs_max * ms_max
            else:
                ms = w / w_max * ms_max
            color = 'r'
#            print 'w_exc ms:', ms

        target_cell_exc = ax.plot(x_tgt, y_tgt, '%so' % color, markersize=ms)
    #    target_plot_ee = ax.plot((x0, x_tgt), (y0, y_tgt), '%s--' % color, lw=line_width)

        if with_annotations:
            ax.annotate('(%d, %.2e, %.2e)' % (tgt, w, d), (x_tgt, y_tgt), fontsize=8)


    direction = ax.plot((x0, x0+u0), (y0, (y0+v0)), 'yD-.', lw=1)

    #ax.legend((target_cell_exc[0], source_plot_ee[0], source_cell_exc[0], direction[0], target_plot_ei[0], source_plot_ie[0]), \
    #        ('exc target cell', 'incoming connections from exc', 'exc source cell', 'predicted direction', 'outgoing connections to inh', 'incoming connections from inh'))
    ax.quiver(x0, y0, u0, v0, angles='xy', scale_units='xy', scale=1, color='y', headwidth=6)

    title = 'Connectivity profile of cell %d' % (exc_cell)
    if relative:
        title += '\nWeights are scaled within group\n'
    else:
        title += '\nWeights are scaled on absolute scale\n'

    title += 'w_min%.2e   w_max=%.2e' % (w_min, w_max)

#    title += '\nw_sigma_x=%.2f w_sigma_v=%.2f' % (params['w_sigma_x'], params['w_sigma_v'])
    ax.set_title(title)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    return ax



def plot_cells_by_euclidean_tuning_distance(subplot_code=111, direction=False):
    def get_dist(x, y):
        d = 0.
        for i in xrange(len(x)):
            d += (x[i] - y[i])**2
        dist = np.sqrt(d)
        return dist

    ax2 = fig.add_subplot(subplot_code)
    dist = np.zeros(len(tgts_ee))
    for i_, tgt in enumerate(tgts_ee):
        x_tgt, y_tgt, u_tgt, v_tgt = tp_exc[tgt, :]
        if direction:
            x = (u_tgt, v_tgt)
            y = (u0, v0)
        else:
            x = (x_tgt, y_tgt, u_tgt, v_tgt)
            y = (x0, y0, u0, v0)
        dist[i_] = get_dist(x, y)
#        print 'debug', i_, tgt, get_dist(x, y), x, y
    #sorted_idx = np.argsort(dist)
    dist_min, dist_max = dist.min(), dist.max()
    print 'direction %s, dist_min=%.2e dist_max=%.2e' % (str(direction), dist_min, dist_max)

    for i in xrange(dist.size):
        x_tgt, y_tgt, u_tgt, v_tgt = tp_exc[i, :]
        h = 0
        l = (dist[i] - dist_min) / (dist_max - dist_min)
        s = 0. # saturation
        assert (0 <= h and h < 360)
        assert (0 <= l and l <= 1)
        assert (0 <= s and s <= 1)
        (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
        x, y, u, v = tp_exc[i, :]
        ax2.plot(x, y, 'o', c=(r,g,b), markersize=4)
        if with_annotations:
            ax2.annotate('(%d, %.2e, %.2e)' % (tgt, w, d), (x_tgt, y_tgt), fontsize=8)


    if direction:
        ax2.set_title('Distance to cell %d\nin direction space' % (exc_cell))
    else:
        ax2.set_title('Euclidean distance to cell %d\nin tuning_prop space' % (exc_cell))
    ax2.set_xlabel('x position')
    ax2.set_ylabel('y position')
    return ax2


def plot_stimulus(ax, motion_params_fn):
    stim_color = 'k'
    mp = np.loadtxt(motion_params_fn)
    ax.quiver(mp[0], mp[1], mp[2], mp[3], angles='xy', scale_units='xy', scale=1, color=stim_color, headwidth=4)
    ax.annotate('Stimulus', (mp[0]+.5*mp[2], mp[1]+0.1), fontsize=12, color=stim_color)



ax = plot_weights(111, True)
motion_params_fn = "%sTrainingInput_%d/input_params.txt" % (params['folder_name'], iteration)
plot_stimulus(ax, motion_params_fn)
#ax = plot_weights(122, False)
#plot_stimulus(ax, motion_params_fn)
#plot_cells_by_euclidean_tuning_distance(223, False)
#plot_cells_by_euclidean_tuning_distance(224, True)

pylab.subplots_adjust(top=0.85, hspace=0.35)
print "Saving fig to", fig_fn
pylab.savefig(fig_fn)
#pylab.show()



