import os
import simulation_parameters
import pylab
import numpy as np
import utils
import matplotlib
import sys
from matplotlib import cm
import plot_hexgrid
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def plot_scatter_with_histograms(x, y, fig):
#    from matplotlib.ticker import NullFormatter

#    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
#    fig2 = pylab.figure(figsize=(8,8))
#    ax = fig.add_subplot(111)
    
    axScatter = fig.add_axes(rect_scatter)
    axScatter.set_xlabel('v_x')
    axScatter.set_ylabel('v_y')
    axHistx = fig.add_axes(rect_histx)
    axHisty = fig.add_axes(rect_histy)

    # no labels
#    axHistx.xaxis.set_major_formatter(nullfmt)
#    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim( (-lim, lim) )
    axScatter.set_ylim( (-lim, lim) )

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

#    print 'xlim', axScatter.get_xlim(), axScatter.get_ylim()
    print 'xymax', xymax, np.max(np.fabs(x)), np.max(np.fabs(y))
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )




rcParams = { 'axes.labelsize' : 18,
            'label.fontsize': 20,
            'xtick.labelsize' : 16, 
            'ytick.labelsize' : 16, 
            'axes.titlesize'  : 16,
            'legend.fontsize': 9, 
            'lines.markeredgewidth' : 0}
pylab.rcParams.update(rcParams)


if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.json'
    print 'Trying to load parameters from', param_fn
    import json
    f = file(param_fn, 'r')
    params = json.load(f)
    re_calculate = False

else:
    print '\nPlotting the default parameters give in simulation_parameters.py\n'
    # load simulation parameters
    ps = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = ps.load_params()                       # params stores cell numbers, etc as a dictionary
    re_calculate = True # could be False, too, if you want


cell_type = 'exc'

if re_calculate: # load 
    print '\nCalculating the tuning prop'
    ps.create_folders()
    d = utils.set_tuning_prop(params, mode='hexgrid', cell_type=cell_type)        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
else:
    fn = params['tuning_prop_means_fn']
    print '\nLoading from', fn
    d = np.loadtxt(fn)



n_cells = d[:, 0].size
if cell_type == 'exc':
    n_rf = params['N_RF_X'] * params['N_RF_Y']
    n_units = params['N_RF_X'] * params['N_RF_Y'] * params['N_theta'] * params['N_V']
else:
    n_rf = params['N_RF_X_INH'] * params['N_RF_Y_INH']
    n_units = params['N_RF_X_INH'] * params['N_RF_Y_INH'] * params['N_theta_inh'] * params['N_V_INH']

ms = 2 # markersize for scatterplots

width = 8
fig = plt.figure(figsize=(12, 9))
plt.subplots_adjust(wspace=.3)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1, 1])
#ax1 = fig.add_subplot(121, autoscale_on=False, aspect='equal')
#ax2 = fig.add_subplot(122, aspect='equal')#, autoscale_on=False)
ax1 = plt.subplot(gs[0], aspect='equal', autoscale_on=False)
ax2 = plt.subplot(gs[1], aspect='equal', autoscale_on=False)


scale = 6. # scale of the quivers / arrows
# set the colorscale for directions
o_min = 0.
o_max = 360.
norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.hsv)#jet)
m.set_array(np.arange(o_min, o_max, 0.01))
rgba_colors = []

#ax4 = fig.add_subplot(224)

thetas = np.zeros(n_cells)
for i in xrange(n_cells):
    x, y, u, v = d[i, :]
    # calculate the color from tuning angle theta
    thetas[i] = np.arctan2(v, u)
    angle = ((thetas[i] + np.pi) / (2 * np.pi)) * 360. # theta determines h, h must be [0, 360)
    rgba_colors.append(m.to_rgba(angle))
    ax2.plot(u, v, 'o', color='k', markersize=ms)#, edgecolors=None)
    ax1.plot(x, y, 'o', color='k', markersize=ms)

# plot the hexgrid 
hexgrid_edge_lw = 2
N_RF = params['N_RF']
RF = plot_hexgrid.get_hexgridcell_midpoints(N_RF)
X = np.unique(RF[0, :])
#xdiff 
# plot the hexgrid edges
xdiff = X[1] - X[0]  # midpoitns x-difference 
edges = plot_hexgrid.get_hexgrid_edges(RF, xdiff)
for i_, edge in enumerate(edges):
    ax1.plot((edge[0][0], edge[0][1]), (edge[1][0], edge[1][1]), c='k', lw=hexgrid_edge_lw)


q = ax1.quiver(d[:, 0], d[:, 1], d[:, 2], d[:, 3], \
          angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='tail')
ax1.set_xlabel('$x$-position')#, fontsize=20)
ax1.set_ylabel('$y$-position')#, fontsize=16)
ax1.set_title('Spatial receptive fields')# for %s cells\n n_rf=%d, n_units=%d' % (cell_type, n_rf, n_units))
ax1.set_xlim((-.05, 1.1))
ax1.set_ylim((-.05, 1.1))
cb = fig.colorbar(m, ax=ax1)#, shrink=.43)
cb.set_label('Preferred angle of motion', fontsize=14)

ax2.set_xlabel('$u$')#, fontsize=16)
ax2.set_ylabel('$v$')#, fontsize=16)
ax2.set_ylim((d[:, 3].min() * 1.05, d[:, 3].max() * 1.05))
ax2.set_xlim((d[:, 2].min() * 1.05, d[:, 2].max() * 1.05))
ax2.set_title('Distribution of preferred directions')

#xticks = np.arange(-.5, .75, 0.25)
#xticks = [-.5, 0., .5]
#xtick_labels = ['%.2f' % xticks[i] for i in xrange(len(xticks))]
#print 'xtick_labels', xtick_labels
#ax2.set_xticks(xticks)
#ax2.set_xticklabels(xtick_labels)

#yticks = [-.5, 0., .5]
#ax2.set_yticks(yticks)
#ax2.set_yticklabels(['%.2f' % yticks[i] for i in xrange(len(yticks))])


"""
bbax1=ax1.get_position()
bbax2=ax2.get_position()
posax1 = bbax1.get_points()
posax2 = bbax2.get_points()
# change height
#posax2[0][1]=posax1[0][1]
posax2[0][1]=posax1[0][1]
posax2[1][1]=posax1[1][1]
# change width
#posax1[1][0]=posax1[1][0]
#posax2[1][0]=posax2[1][0]

bbax1.set_points(posax1)
bbax2.set_points(posax2)

print 'posax1', posax1
print 'posax2', posax2
#! Update axes with new position
posax1=ax1.set_position(bbax1)
posax2=ax2.set_position(bbax2)
"""

#ax3.set_xlabel('$x$')
#ax3.set_ylabel('$y$')
#ax3.set_title('Preferred directions')
#yticks = ax3.get_yticks()
#xticks = ax3.get_xticks()
#yticks = np.arange(
#xticks = []
#for i in xrange(len(yticks)):
#    yticks_rescaled.append(yticks[i] / scale)
#for i in xrange(len(xticks)):
#    xticks_rescaled.append(xticks[i] / scale)
#ax3.set_yticklabels(yticks_rescaled)
#ax3.set_xticklabels(xticks_rescaled)

output_fn = params['tuning_prop_fig_%s_fn' % cell_type]
output_fn = output_fn.rstrip('.png') + '_%.2f.png' % params['log_scale']
print "Saving to ... ", output_fn
fig.savefig(output_fn, dpi=200)


fig2 = pylab.figure(figsize=(width, width))
#ax2 = fig2.add_subplot(111, autoscale_on=False, aspect='equal')
#ax2 = fig.add_subplot(122, aspect='equal')
#ax2 = pylab.axes()
#plot_scatter_with_histograms(d[:, 0], d[:, 1])
plot_scatter_with_histograms(d[:, 2], d[:, 3], fig2)
output_fn = params['figures_folder'] + 'v_tuning_histogram_vmin%.2e_vmax%.2e.png' % (params['v_min_tp'], params['v_max_tp'])
print 'Saving to', output_fn
fig2.savefig(output_fn, dpi=200)

pylab.show()
