"""
This script plots the input spike trains in the top panel
and the output rasterplots in the middle and lower panel
"""
import sys
import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import re
import utils
import os
rcP= { 'axes.labelsize' : 24,
            'label.fontsize': 24,
            'xtick.labelsize' : 24, 
            'ytick.labelsize' : 24, 
            'axes.titlesize'  : 32,
            'legend.fontsize': 9}


if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.info'
    import NeuroTools.parameters as NTP
    fn_as_url = utils.convert_to_url(param_fn)
    print 'Loading parameters from', param_fn
    params = NTP.ParameterSet(fn_as_url)

else:
    print '\nPlotting the default parameters given in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary


def plot_input_spikes(ax, shift=0, m='o', c='k', ms=2):
    """
    Shift could be used when plotting in the same axis as the output spikes
    """
    n_cells = params['n_exc']
    for cell in xrange(n_cells):
        fn = params['input_st_fn_base'] + str(cell) + '.npy'
        spiketimes = np.load(fn)
        nspikes = len(spiketimes)
        ax.plot(spiketimes, cell * np.ones(nspikes) + shift, m, color=c, alpha=.1, markersize=ms)


tp = np.loadtxt(params['tuning_prop_means_fn'])
def plot_input_spikes_sorted_in_space(ax, shift=0., m='o', c='g', sort_idx=0, ms=2):
    n_cells = params['n_exc']
    sorted_idx = tp[:, sort_idx].argsort()

    if sort_idx == 0 or sort_idx == 1:
        ylim = (0, 1)
    else: # it's a velocity --> adjust the range to plot
        crop = .8
        ylim = (crop * tp[:, sort_idx].min(), crop * tp[:, sort_idx].max())
    ylen = (abs(ylim[0] - ylim[1]))
    for i in xrange(n_cells):
        cell = sorted_idx[i]
        fn = params['input_st_fn_base'] + str(cell) + '.npy'
        if os.path.exists(fn):
            spiketimes = np.load(fn)
            nspikes = len(spiketimes)
            if sort_idx == 0:
                y_pos = (tp[cell, sort_idx] % 1.) / ylen * (abs(ylim[0] - ylim[1]))
            else:
                y_pos = (tp[cell, sort_idx]) / ylen * (abs(ylim[0] - ylim[1]))
            ax.plot(spiketimes, y_pos * np.ones(nspikes) + shift, m, color=c, alpha=.1, markersize=2)
        # else: this cell gets no input, because not well tuned
#        ax.plot(spiketimes, i * np.ones(nspikes) + shift, m, color=c, markersize=2)

    if sort_idx == 0:
        ylabel_txt ='Neurons sorted by $x$-pos'
    elif sort_idx == 1:
        ylabel_txt ='Neurons sorted by $y$-pos'
    elif sort_idx == 2:
        ylabel_txt ='Neurons sorted by $v_x$, '
    elif sort_idx == 3:
        ylabel_txt ='Neurons sorted by $v_y$'

    ax.set_ylabel(ylabel_txt)

#    n_yticks = 8
#    y_tick_idx = np.linspace(0, n_cells, n_yticks)
#    y_ticks = np.linspace(tp[:, sort_idx].min(), tp[:, sort_idx].max(), n_yticks)
#    y_ticklabels = []
#    for i in xrange(n_yticks):
#        y_ticklabels.append('%.2f' % y_ticks[i])
#    ax.set_yticks(y_tick_idx)
#    ax.set_yticklabels(y_ticklabels)


def plot_output_spikes_sorted_in_space(ax, cell_type, shift=0., m='o', c='g', sort_idx=0, ms=2):
    n_cells = params['n_%s' % cell_type]
    fn = params['%s_spiketimes_fn_merged' % cell_type] + '.ras'
    nspikes, spiketimes = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
    sorted_idx = tp[:, sort_idx].argsort()

    if sort_idx == 0:
        ylim = (0, 1)
    else:
        crop = .8
        ylim = (crop * tp[:, sort_idx].min(), crop * tp[:, sort_idx].max())
    ylen = (abs(ylim[0] - ylim[1]))
    print '\n', 'sort_idx', sort_idx, ylim, 
    for i in xrange(n_cells):
        cell = sorted_idx[i]
        if sort_idx == 0:
            y_pos = (tp[cell, sort_idx] % 1.) / ylen * (abs(ylim[0] - ylim[1]))
        else:
            y_pos = (tp[cell, sort_idx]) / ylen * (abs(ylim[0] - ylim[1]))
        ax.plot(spiketimes[cell], y_pos * np.ones(nspikes[cell]), 'o', color='k', markersize=ms)

#    n_yticks = 6
#    y_tick_idx = np.linspace(0, n_cells, n_yticks)
#    y_ticks = np.linspace(tp[:, sort_idx].min(), tp[:, sort_idx].max(), n_yticks)
#    y_ticklabels = []
#    for i in xrange(n_yticks):
#        y_ticklabels.append('%.2f' % y_ticks[i])
#    ax.set_yticks(y_tick_idx)
#    ax.set_yticklabels(y_ticklabels)



def plot_spikes(ax, fn, n_cells):
    nspikes, spiketimes = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
    for cell in xrange(int(len(spiketimes))):
        ax.plot(spiketimes[cell], cell * np.ones(nspikes[cell]), 'o', color='k', markersize=2)



fn_exc = params['exc_spiketimes_fn_merged'] + '.ras'
fn_inh = params['inh_spiketimes_fn_merged'] + '.ras'


pylab.rcParams['lines.markeredgewidth'] = 0
pylab.rcParams.update(rcP)
# ax1 is if input spikes shall be plotted in a seperate axis  (from the output spikes)
#fig = pylab.figure(figsize=(14, 12))
#ax1 = fig.add_subplot(411)
#ax2 = fig.add_subplot(412)
#ax3 = fig.add_subplot(413)
#ax4 = fig.add_subplot(414)
fig = pylab.figure()#figsize=(14, 12))
pylab.subplots_adjust(bottom=.15, left=.15)#hspace=.03)
ax1 = fig.add_subplot(111)
#fig2 = pylab.figure(figsize=(14, 12))
#ax2 = fig2.add_subplot(111)
#ax3 = fig.add_subplot(413)
#ax4 = fig.add_subplot(414)





#ax1.set_title('Input spikes')

#ax3.set_ylabel('Exc ID')
#ax4.set_ylabel('Inh ID')


# x-position
plot_input_spikes_sorted_in_space(ax1, c='b', sort_idx=0, ms=3) 
plot_output_spikes_sorted_in_space(ax1, 'exc', c='k', sort_idx=0, ms=3) 

# sorted by velocity in direction x / y
#plot_input_spikes_sorted_in_space(ax2, c='b', sort_idx=2, ms=3)
#plot_output_spikes_sorted_in_space(ax2, 'exc', c='k', sort_idx=2, ms=3) 


#plot_spikes(ax3, fn_exc, params['n_exc'])
#plot_spikes(ax4, fn_inh, params['n_inh'])


xticks = [0, 500, 1000, 1500]
ax1.set_xticks(xticks)
ax1.set_xticklabels(['%d' % i for i in xticks])
ax1.set_yticklabels(['', '.2', '.4', '.6', '.8', '1.0'])
ax1.set_xlabel('Time [ms]')
#ax2.set_xlabel('Time [ms]')
#ax3.set_xlabel('Time [ms]')
#ax4.set_xlabel('Time [ms]')

#ax1.set_ylim((0, params['n_exc'] + 1))
#ax2.set_ylim((0, params['n_exc'] + 1))
#ax3.set_ylim((0, params['n_exc'] + 1))
#ax4.set_ylim((0, params['n_inh'] + 1))

ax1.set_xlim((0, params['t_sim']))
#ax2.set_xlim((0, params['t_sim']))
#ax2.set_ylim((-3, 3))

#ax3.set_xlim((0, params['t_sim']))
#ax4.set_xlim((0, params['t_sim']))

output_fn = params['figures_folder'] + 'rasterplot_sorted_by_tp.png'
print "Saving to", output_fn
pylab.savefig(output_fn, dpi=200)
#output_fn = params['figures_folder'] + 'rasterplot_sorted_by_tp.pdf'
#print "Saving to", output_fn
#pylab.savefig(output_fn, dpi=200)
#output_fn = params['figures_folder'] + 'rasterplot_sorted_by_tp.eps'
#print "Saving to", output_fn
#pylab.savefig(output_fn, dpi=200)


pylab.show()

