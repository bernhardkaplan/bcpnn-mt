import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import utils
import pylab
import matplotlib
from PlottingScripts.plot_spikes_sorted import plot_spikes_sorted
import json
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 12,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 10,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 1,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.25,
              'figure.subplot.bottom':.08,
              'figure.subplot.right':.80,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}
pylab.rcParams.update(plot_params)


def plot_currents(params):

    fn = params['data_folder'] + 'input_currents.json'
    f = file(fn, 'r')
    d = json.load(f)
    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    stim_params = np.loadtxt(params['test_sequence_fn'])
    stim_range = params['test_stim_range']
    if params['n_stim'] == 1 and stim_range[0] != 0:
        mp = stim_params[stim_range[0]:stim_range[1], :]
    elif params['n_stim'] == 1 and stim_range[0] == 0:
        mp = stim_params.reshape((1, 4))
#        mp = stim_params[0, :]
    print 'mp:', mp, mp.shape
    x_start_blank, x_stop_blank = utils.get_blank_pos(params, mp[0, :])
    print 'x_start_blank, x_stop_blank:', x_start_blank, x_stop_blank
    dv = 0.15 # only plot cells that are +- dv near the tested stimulus

    v_range = (-1.0, 1.)
    clim = (v_range[0], v_range[1])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m_v = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m_v.set_array(v_range)
#    m_v.set_array(tp[:, 2])
#    colors_v = m_v.to_rgba(d[:, 1])

    x_range = (.0, 1.)
    clim = (x_range[0], x_range[1])
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m_x = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m_x.set_array(tp[:, 0])
#    colors_x = m_x.to_rgba(d[:, 0])

    fig = pylab.figure(figsize=(12,12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plots = []
    plots2 = []
    labels = []

    measured = d.keys()
    print 'Observable:', measured
    i_ampa = d['AMPA']
    i_ampa_integral = d['AMPA']['full_integral']
    i_nmda_integral = d['NMDA']['full_integral']
    ratios = []
    gids = [int('%s' % i_) - 1 for i_  in i_ampa_integral.keys()]
    tp_x = gids
    for nest_gid_str in i_ampa_integral.keys():
        gid = int(nest_gid_str) - 1
        x, y, u, v = tp[gid, :]
        if np.abs(u - mp[0, 2]) < dv:
            p1, = ax1.plot(x, i_ampa_integral[nest_gid_str], 'o', ms=10, c=m_v.to_rgba(u))
            p2, = ax1.plot(x, i_nmda_integral[nest_gid_str], '^', ms=10, c=m_v.to_rgba(u))
            plots.append(p1)
            plots2.append(p2)

            if i_nmda_integral[nest_gid_str] != 0.:
                ratio_ampa_nmda = i_ampa_integral[nest_gid_str] / i_nmda_integral[nest_gid_str]
                ax2.plot(x, ratio_ampa_nmda, 'o', ms=10, c=m_v.to_rgba(u))
                ratios.append(ratio_ampa_nmda)

    print 'Measure AMPA/NMDA ratios:', ratios
    mean_ampa_nmda_ratio = np.array(ratios).mean()
    std_ampa_nmda_ratio = np.array(ratios).std()
    print 'Measure on average AMPA/NMDA ratio:',  mean_ampa_nmda_ratio, '+-', std_ampa_nmda_ratio

    ylim = ax1.get_ylim()
    ax1.plot((x_start_blank, x_start_blank), (ylim[0], ylim[1]), ls='--', c='k', lw=2)
    ax1.plot((x_stop_blank, x_stop_blank), (ylim[0], ylim[1]), ls='--', c='k', lw=2)

#    ax2.set_ylim((0., mean_ampa_nmda_ratio + 1 * std_ampa_nmda_ratio))
    ax2.set_ylim((0., 3.))
    ylim2 = ax2.get_ylim()
    xlim2 = ax2.get_xlim()
    ax2.plot((x_start_blank, x_start_blank), (ylim2[0], ylim2[1]), ls='--', c='k', lw=2)
    ax2.plot((x_stop_blank, x_stop_blank), (ylim2[0], ylim2[1]), ls='--', c='k', lw=2)
    
    color = 'grey'
#    color = 'grey'
    ax2.plot((xlim2[0], xlim2[1]), (mean_ampa_nmda_ratio, mean_ampa_nmda_ratio), '-', c=color, lw=5)
    ax2.plot((xlim2[0], xlim2[1]), (mean_ampa_nmda_ratio + std_ampa_nmda_ratio, mean_ampa_nmda_ratio + std_ampa_nmda_ratio), '--', c=color, lw=5)
    ax2.plot((xlim2[0], xlim2[1]), (mean_ampa_nmda_ratio - std_ampa_nmda_ratio, mean_ampa_nmda_ratio - std_ampa_nmda_ratio), '--', c=color, lw=5)
    ax2.plot((xlim2[0], xlim2[1]), (.2, .2), '--', c='k', lw=1)

    ax1.legend([p1, p2], ['AMPA', 'NMDA'], scatterpoints=1, loc='upper left', ncol=1, numpoints=1)
#    ax2.legend([p2], ['NMDA'], scatterpoints=1, loc='upper left', ncol=1, numpoints=1)

    ax1.xaxis.set_major_locator(MultipleLocator(.2))
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.15)

    cb_v = pylab.colorbar(m_v, cax=cax1)
    cb_v.set_label('$v_{tgt}$')
#    cb_x = pylab.colorbar(m_x, cax=cax2)
#    cb_x.set_label('$x_{tgt}$')

    ax1.set_ylabel('Sum of incoming currents')
    ax1.set_xlabel('Target cell position $x_j$')

    ax2.set_ylabel('Ratio AMPA/NMDA')
    ax2.set_xlabel('Target cell position $x_j$')

#    ax3 = ax2.twiny()
#    ax3 

#    ax2.set_xlabel('Target MC speed $v_j$')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        show = True
        plot_currents(params)
    elif len(sys.argv) == 2: 
        folder_name = sys.argv[1]
        params = utils.load_params(folder_name)
        show = True
        plot_currents(params)
    else:
        for folder_name in sys.argv[1:]:
            params = utils.load_params(folder_name)
            show = False
            plot_currents(params)
    if show:
        pylab.show()

