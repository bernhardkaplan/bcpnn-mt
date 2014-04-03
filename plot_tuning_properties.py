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

rcParams = { 'axes.labelsize' : 18,
            'label.fontsize': 20,
            'xtick.labelsize' : 16, 
            'ytick.labelsize' : 16, 
            'axes.titlesize'  : 16,
            'legend.fontsize': 9, 
            'lines.markeredgewidth' : 0}
pylab.rcParams.update(rcParams)


cell_type = 'exc'
#cell_type = 'inh'

if re_calculate: # load 
    print '\nCalculating the tuning prop'
    ps.create_folders()
    d = utils.set_tuning_prop(params, mode='hexgrid', cell_type=cell_type)        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
else:
    fn = params['tuning_prop_means_fn']
    print '\nLoading from', fn
    d = np.loadtxt(fn)


n_cells = d[:, 0].size

def plot_histogram(data, fig, xlabel='', ylabel='count', title='', n_bins=50):
    count, bins = np.histogram(data, bins=n_bins)
    binwidth = bins[1] - bins[0]
    ax = fig.add_subplot(111)
    ax.bar(bins[:-1], count, width=binwidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)



def plot_scatter_with_histograms(x, y, fig, title='', xv='x'):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    axScatter = fig.add_axes(rect_scatter)
    if xv == 'x':
        axScatter.set_xlabel('x')
        axScatter.set_ylabel('y')
    else:
        axScatter.set_xlabel('v_x')
        axScatter.set_ylabel('v_y')
    axHistx = fig.add_axes(rect_histx)
    axHisty = fig.add_axes(rect_histy)

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

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
    axScatter.set_title(title)
    return axScatter, axHistx, axHisty


def transform_orientation_to_color(orientation):
    o_min, o_max = 0, 180
    norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)
    m.set_array(np.arange(o_min, o_max, 0.01))
    angle = (orientation / (2 * np.pi)) * 360. # orientation determines h, h must be [0, 360)
    color = m.to_rgba(angle)
    return color

def plot_orientation_as_quiver(tp):
    """
    data -- tuning properties of the cells
    """

    o_min, o_max = 0, 180
    norm = matplotlib.mpl.colors.Normalize(vmin=o_min, vmax=o_max)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)
    m.set_array(np.arange(o_min, o_max, 0.01))

    theta_x = np.cos(tp[:, 4])
    theta_y = np.sin(tp[:, 4])

    rgba_colors = []
    for i in xrange(n_cells):
        x, y, u, v, orientation = d[i, :]
        # calculate the color from tuning angle orientation
        angle = (orientation / (2 * np.pi)) * 360. # orientation determines h, h must be [0, 360)
        rgba_colors.append(m.to_rgba(angle))

    fig_2 = pylab.figure()
    ax = fig_2.add_subplot(111)

    scale = 8.
    ax.quiver(tp[:, 0], tp[:, 1], theta_x, theta_y, \
              angles='xy', scale_units='xy', scale=scale, color=rgba_colors, headwidth=4, pivot='middle', width=0.007)
    ax.scatter(tp[:, 0], tp[:, 1])
    return ax 



def plot_predictor_sequence():
    # PLOT PREDICTOR SEQUENCE
    protocol = params['motion_protocol']
    record_gids = utils.select_well_tuned_cells(d, params['mp_select_cells'], params, params['n_gids_to_record'])
    ax = plot_orientation_as_quiver(d[record_gids, :])

    random_predictor_mp = np.loadtxt(params['all_predictor_params_fn'])
    scale = 8.
    # plot the arrows with orientation determining their color
    ax.quiver(random_predictor_mp[:, 0], random_predictor_mp[:, 1], random_predictor_mp[:, 2], random_predictor_mp[:, 3], \
                  angles='xy', scale_units='xy', scale=scale, color=transform_orientation_to_color(random_predictor_mp[:, 4]), \
                  linewidths=(1,), edgecolors=('k'), zorder=100000, headwidth=4, pivot='middle', width=0.007)
    # annotate the order of stimuli
    for i in xrange(random_predictor_mp[:, 0].size):
        x, y = random_predictor_mp[i, 0:2] 
        ax.annotate('%d' % i, (x * 1.1, y * 1.1))
    ax.set_xlim((-.1, 1.))
    ax.set_ylim((-.1, 1.))
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    ax.set_title('Recorded cells and predictor sequence for %s' % protocol)
    output_fn = params['figures_folder'] + 'predictor_sequence_%s.png' % (params['motion_protocol'])
    print 'Saving to:', output_fn
    pylab.savefig(output_fn)


def plot_spatial_and_direction_tuning_2D():

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
        x, y, u, v, theta = d[i, :]
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

    output_fn = params['tuning_prop_fig_%s_fn' % cell_type]
    output_fn = output_fn.rstrip('.png') + '_%.2f.png' % params['log_scale']
    print "Saving to ... ", output_fn
    fig.savefig(output_fn, dpi=200)


#plot_spatial_and_direction_tuning_1D()
#plot_spatial_and_direction_tuning_2D()

width = 8
fig2 = pylab.figure(figsize=(width, width))
axScatter, axHistx, axHisty = plot_scatter_with_histograms(d[:, 2], d[:, 3], fig2, 'Distribution of preferred directions')
output_fn = params['figures_folder'] + 'v_tuning_histogram_vmin%.2e_vmax%.2e.png' % (params['v_min_tp'], params['v_max_tp'])
print 'Saving to', output_fn
fig2.savefig(output_fn, dpi=200)


fig3 = pylab.figure(figsize=(width, width))
axScatter, axHistx, axHisty = plot_scatter_with_histograms(d[:, 0], d[:, 1], fig3, 'Distribution of spatial receptive fields')
axScatter.set_xlim((0, 1))
axScatter.set_ylim((0, 1))
axHistx.set_xlim((0, 1))
axHisty.set_ylim((0, 1))

fig4 = pylab.figure(figsize=utils.get_figsize(600))
plot_histogram(d[:, 0], fig4, xlabel='x-position', title='Distribution of x-positions')
fig5 = pylab.figure(figsize=utils.get_figsize(600))
plot_histogram(d[:, 2], fig5, xlabel='$v_x$', title='Distribution of preferred x-directions')

pylab.show()


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

