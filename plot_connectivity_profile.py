import numpy as np
import utils
import pylab
import sys

import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp = np.loadtxt(params['tuning_prop_means_fn'])
mp = params['motion_params']
print "Motion parameters", mp

#sigma_x, sigma_v = params['w_sigma_x'], params['w_sigma_v'] # small sigma values let p and w shrink
#def connection_probability(x_src, y_src, u_src, v_src, x_tgt, y_tgt, u_tgt, v_tgt):
#    latency = np.sqrt((x_src - x_tgt)**2 + (y_src - y_tgt)**2) / np.sqrt(u_src**2 + v_src**2)
#    p = .5 * np.exp(-((x_src + u_src * latency - x_tgt)**2 + (y_src + v_src * latency - y_tgt)**2) / (2 * sigma_x**2)) \
#            * np.exp(-((u_src-u_tgt)**2 + (v_src - v_tgt)**2) / (2 * sigma_v**2))


conn_list_fn = params['conn_list_ee_fn_base'] + '0.dat'
#conn_list_fn = 'NoColumns_winit_precomputed/Connections/conn_list_ee_0.dat'
#conn_list_fn = 'NoColumns_winit_random/Connections/conn_list_ee_0.dat'
print "Loading connectivity data from ", conn_list_fn
conn_mat, delays = utils.convert_connlist_to_matrix(conn_list_fn, params['n_exc'])

src_cell = int(sys.argv[1])
tgts = conn_mat[src_cell, :].nonzero()[0]
srcs = conn_mat[:, src_cell].nonzero()[0]
weights1 = conn_mat[src_cell, tgts]
print "Target cells:", tgts
print "Weights:", weights1
weights2 = conn_mat[srcs, src_cell]
print "Source cells (projecting to the selected one):", srcs
print "Weights:", weights2
weights = weights1.tolist() + weights2.tolist()
w_max = np.max(weights)
w_min = np.min(weights)
lw_max = 6

print "Plotting..."
fig = pylab.figure()
ax1 = fig.add_subplot(111)
# plot the source cell position
x_src = tp[src_cell, 0] 
y_src = tp[src_cell, 1] 
ax1.plot(x_src, y_src, 'o', c='k', markersize=2)

for tgt in tgts:
    x_tgt = tp[tgt, 0] 
    y_tgt = tp[tgt, 1] 
    ax1.plot(x_tgt, y_tgt, 'o', c='b', markersize=1)
    w = conn_mat[src_cell, tgt]
    d = delays[src_cell, tgt]
    line_width = round(w / w_max * lw_max) + 1
    print "%d %d %.3e %d" % (src_cell, tgt, w, line_width)
    dx = (x_tgt - x_src)
    dy = (y_tgt - y_src)
    m = dy / dx
    ax1.plot((x_src, x_tgt), (y_src, y_tgt), 'b', lw=line_width)
    rnd = min(1., np.random.rand() + .5)
#    ax1.annotate('(%d, %.2e, %.2e)' % (tgt, w, d), (x_tgt, y_tgt), fontsize=6)

x_tgt = tp[src_cell, 0] 
y_tgt = tp[src_cell, 1] 
for src in srcs:
    x_src = tp[src, 0] 
    y_src = tp[src, 1] 
    ax1.plot(x_src, y_src, 'o', c='r', markersize=1)
    w = conn_mat[src, src_cell]
    d = delays[src, src_cell]
    line_width = round(w / w_max * lw_max) + 1
    print "%d %d %.3e %d" % (src_cell, src, w, line_width)
    dx = (x_tgt - x_src)
    dy = (y_tgt - y_src)
    m = dy / dx
    ax1.plot((x_src, x_tgt), (y_src, y_tgt), 'r', lw=line_width)
    rnd = max(.5, .5 * np.random.rand())
#    ax1.annotate('(%d, %.2e, %.2e)' % (src, w, d), (x_src, y_src), fontsize=6)

#xgrid, ygrid = 0.01, 0.01
#x = np.arange(0, 1.2, xgrid)
#y = np.arange(0, 1.2, ygrid)
#X, Y = np.meshgrid(x, y)
#Z = connection_probability
#ax1.pcolor(

fig_fn = 'precomp_conn_profile_%d.png' % src_cell
print "Saving fig to", fig_fn
pylab.savefig(fig_fn)



