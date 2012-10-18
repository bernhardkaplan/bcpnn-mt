import numpy as np
import utils
import pylab
import sys
import os

import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp_exc = np.loadtxt(params['tuning_prop_means_fn'])

lw_max = 10
with_annotations = False
fig = pylab.figure()
ax = fig.add_subplot(111)

exc_cell = int(sys.argv[1])
x0, y0, u0, v0 = tp_exc[exc_cell, 0], tp_exc[exc_cell, 1], tp_exc[exc_cell, 2], tp_exc[exc_cell, 3]

conn_mat = np.loadtxt(params['weight_matrix_abstract'])

tgts_ee = np.arange(0, params['n_exc'], 1)
srcs_ee = np.arange(0, params['n_exc'], 1)
print 'debug', tgts_ee, type(tgts_ee)

print "Plotting exc_cell -> E"
lws = utils.linear_transformation(conn_mat[exc_cell, tgts_ee], 1, lw_max)

w_max = abs(conn_mat[exc_cell, :].max())

print 'debug w:', conn_mat[exc_cell, :]
for i_, tgt in enumerate(tgts_ee):
    x_tgt = tp_exc[tgt, 0] 
    y_tgt = tp_exc[tgt, 1] 
#    ax.plot(x_tgt, y_tgt, 'o', c='b', markersize=1)
    w = conn_mat[exc_cell, tgt]
    if w < 0:
        color = 'b'
    else:
        color = 'r'
#    d = delays_ee[exc_cell, tgt]
    line_width = lws[i_]

    ms = abs(w) / w_max * 10
#    for i in xrange(n_cells):
#        h = 240.
#        l = 1. - 0.5 * input_sum[i] / input_max
#        s = 1. # saturation
#        assert (0 <= h and h < 360)
#        assert (0 <= l and l <= 1)
#        assert (0 <= s and s <= 1)
#        (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
#        x, y, u, v = tp[i, :]
#        ax.plot(x, y, 'o', c=(r,g,b), markersize=ms)

    target_cell_exc = ax.plot(x_tgt, y_tgt, '%so' % color, markersize=ms)

#    target_plot_ee = ax.plot((x0, x_tgt), (y0, y_tgt), '%s--' % color, lw=line_width)
    if with_annotations:
        ax.annotate('(%d, %.2e, %.2e)' % (tgt, w, d), (x_tgt, y_tgt), fontsize=8)

direction = ax.plot((x0, x0+u0), (y0, (y0+v0)), 'yD-.', lw=1)
#ax.legend((target_cell_exc[0], source_plot_ee[0], source_cell_exc[0], direction[0], target_plot_ei[0], source_plot_ie[0]), \
#        ('exc target cell', 'incoming connections from exc', 'exc source cell', 'predicted direction', 'outgoing connections to inh', 'incoming connections from inh'))
ax.quiver(x0, y0, u0, v0, angles='xy', scale_units='xy', scale=1, color='y', headwidth=6)

title = 'Connectivity profile of cell %d\ntp:' % (exc_cell) + str(tp_exc[exc_cell, :])
title += '\nw_sigma_x=%.2f w_sigma_v=%.2f' % (params['w_sigma_x'], params['w_sigma_v'])
ax.set_title(title)
ax.set_xlabel('x position')
ax.set_ylabel('y position')
#pylab.legend(('outgoing connections', 'incoming connections'))

#xgrid, ygrid = 0.01, 0.01
#x = np.arange(0, 1.2, xgrid)
#y = np.arange(0, 1.2, ygrid)
#X, Y = np.meshgrid(x, y)
#Z = connection_probability
#ax.pcolor(

fig_fn = params['figures_folder'] + 'precomp_conn_profile_%d.png' % exc_cell
print "Saving fig to", fig_fn
pylab.savefig(fig_fn)
pylab.show()



