import numpy as np
import utils
import pylab
import sys
import os

import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp_exc = np.loadtxt(params['tuning_prop_means_fn'])
tp_inh = np.loadtxt(params['inh_cell_pos_fn'])

lw_max = 10
with_annotations = False
fig = pylab.figure()
ax = fig.add_subplot(111)

exc_cell = int(sys.argv[1])
x0, y0, u0, v0 = tp_exc[exc_cell, 0], tp_exc[exc_cell, 1], tp_exc[exc_cell, 2], tp_exc[exc_cell, 3]

conn_mat_ee_fn = params['conn_mat_fn_base'] + 'ee.dat'
conn_mat_ei_fn = params['conn_mat_fn_base'] + 'ei.dat'
conn_mat_ie_fn = params['conn_mat_fn_base'] + 'ie.dat'
delay_mat_ee_fn = params['delay_mat_fn_base'] + 'ee.dat'
delay_mat_ei_fn = params['delay_mat_fn_base'] + 'ei.dat'
delay_mat_ie_fn = params['delay_mat_fn_base'] + 'ie.dat'
if os.path.exists(conn_mat_ee_fn):
    print 'Loading', conn_mat_ee_fn
    conn_mat_ee = np.loadtxt(conn_mat_ee_fn)
#    delays_ee = np.loadtxt(delay_mat_ee_fn)
else:
    conn_mat_ee, delays_ee = utils.convert_connlist_to_matrix(params['merged_conn_list_ee'], params['n_exc'])
    np.savetxt(conn_mat_ee_fn, conn_mat_ee)
#    np.savetxt(delay_mat_ee_fn, delay_mat_ee)

if os.path.exists(conn_mat_ei_fn):
    print 'Loading', conn_mat_ei_fn
    conn_mat_ei = np.loadtxt(conn_mat_ei_fn)
#    delays_ei = np.loadtxt(delay_mat_ei_fn)
else:
    conn_mat_ei, delays_ei = utils.convert_connlist_to_matrix(params['merged_conn_list_ei'], params['n_exc'])
    np.savetxt(conn_mat_ei_fn, conn_mat_ei)
#    np.savetxt(delay_mat_ei_fn, delay_mat_ei)

if os.path.exists(conn_mat_ie_fn):
    print 'Loading', conn_mat_ie_fn
    conn_mat_ie = np.loadtxt(conn_mat_ie_fn)
#    delays_ie = np.loadtxt(delay_mat_ie_fn)
else:
    conn_mat_ie, delays_ie = utils.convert_connlist_to_matrix(params['merged_conn_list_ie'], params['n_exc'])
    np.savetxt(conn_mat_ie_fn, conn_mat_ie)
#    np.savetxt(delay_mat_ie_fn, delay_mat_ie)

tgts_ee = conn_mat_ee[exc_cell, :].nonzero()[0]
srcs_ee = conn_mat_ee[:, exc_cell].nonzero()[0]
weights_ee = conn_mat_ee[exc_cell, tgts_ee]

print "Plotting exc_cell -> E"
lws = utils.linear_transformation(conn_mat_ee[exc_cell, tgts_ee], 1, lw_max)
for i_, tgt in enumerate(tgts_ee):
    x_tgt = tp_exc[tgt, 0] 
    y_tgt = tp_exc[tgt, 1] 
    ax.plot(x_tgt, y_tgt, 'o', c='b', markersize=1)
    w = conn_mat_ee[exc_cell, tgt]
#    d = delays_ee[exc_cell, tgt]
    line_width = lws[i_]
    target_cell_exc = ax.plot(x_tgt, y_tgt, 'bo', lw=line_width)
#    target_plot_ee = ax.plot((x0, x_tgt), (y0, y_tgt), 'b--', lw=line_width)
    if with_annotations:
        ax.annotate('(%d, %.2e, %.2e)' % (tgt, w, d), (x_tgt, y_tgt), fontsize=8)

print "Plotting E -> exc_cell"
lws = utils.linear_transformation(conn_mat_ee[srcs_ee, exc_cell], 1, lw_max)
for i_, src in enumerate(srcs_ee):
    x_src = tp_exc[src, 0] 
    y_src = tp_exc[src, 1] 
    w = conn_mat_ee[src, exc_cell]
#    d = delays_ee[src, exc_cell]
    line_width = lws[i_]
    source_cell_exc = ax.plot(x_src, y_src, 'b^', lw=line_width)
    source_plot_ee = ax.plot((x_src, x0), (y_src, y0), 'b:', lw=line_width)

print "Plotting exc_cell -> I"
tgts_ei = conn_mat_ei[exc_cell, :].nonzero()[0]
#lws = utils.linear_transformation(conn_mat_ei[tgts_ei, exc_cell], 1, lw_max)
lws = [2 for i in xrange(len(tgts_ei))]
for i_, tgt in enumerate(tgts_ei):
    x_tgt = tp_inh[tgt, 0] 
    y_tgt = tp_inh[tgt, 1] 
    w = conn_mat_ei[tgt, exc_cell]
#    d = delays_ei[tgt, exc_cell]
    line_width = lws[i_]
    target_plot_ei = ax.plot(x_tgt, y_tgt, 'ro', lw=line_width)

print "Plotting I -> exc_cell"
srcs_ie = conn_mat_ie[:, exc_cell].nonzero()[0]
#lws = utils.linear_transformation(conn_mat_ie[srcs_ie, exc_cell], 1, lw_max)
lws = [2 for i in xrange(len(srcs_ie))]
for i_, src in enumerate(srcs_ie):
    x_src = tp_inh[src, 0] 
    y_src = tp_inh[src, 1] 
    ax.plot(x_src, y_src, 'o', c='r', markersize=1)
    w = conn_mat_ie[src, exc_cell]
#    d = delays_ie[src, exc_cell]
    line_width = lws[i_]
    source_plot_ie = ax.plot(x_src, y_src, 'r^', lw=line_width)
    source_plot_ie = ax.plot((x_src, x0), (y_src, y0), 'r:', lw=line_width)

direction = ax.plot((x0, x0+u0), (y0, (y0+v0)), 'yD-.', lw=1)
ax.legend((target_cell_exc[0], source_plot_ee[0], source_cell_exc[0], direction[0], target_plot_ei[0], source_plot_ie[0]), \
        ('exc target cell', 'incoming connections from exc', 'exc source cell', 'predicted direction', 'outgoing connections to inh', 'incoming connections from inh'))
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



