import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import utils
import pylab
import matplotlib
import json
from PlottingScripts.plot_spikes_sorted import plot_spikes_sorted_simple


"""
Anticipation data is written by PlottingScripts/PlotAnticipation plot_anticipation_cmap(params)

This script just loads the json files from params['data_folder'] + 'anticipation_data.json'
"""

def get_tanticipation_data(folder_names):

    
    d = {'taui_ampa_nmda' : [], 
            'g_exc_total_in': [], 
            'g_inh_total_in': [], 
            'w_noise_exc': [], 
            'w_noise_inh': [], 
            'ampa_nmda_ratio': [], 
            't_anticipation_vmem': [], 
            't_anticipation_spikes': []}


    for i_, folder_name in enumerate(folder_names):
        params = utils.load_params(folder_name)
        d['taui_ampa_nmda'].append((params['taui_ampa'], params['taui_nmda']))
        d['g_exc_total_in'].append(params['g_exc_total_in'])
        d['g_inh_total_in'].append(params['g_inh_total_in'])
        d['w_noise_exc'].append(params['w_noise_exc'])
        d['w_noise_inh'].append(params['w_noise_inh'])
        d['ampa_nmda_ratio'].append(params['ampa_nmda_ratio'])

        fn = params['data_folder'] + 'anticipation_spike_data.json'
        f = file(fn, 'r')
        da = json.load(f)
        d['t_anticipation_spikes'].append(da['t_anticipation_spikes_filtered'])

        fn = params['data_folder'] + 'anticipation_volt_data.json'
        f = file(fn, 'r')
        da = json.load(f)
        d['t_anticipation_vmem'].append(da['t_anticipation_vmem'])

        fn = params['data_folder'] + 'anticipation_spike_data.json'
        f = file(fn, 'r')
        da = json.load(f)
        d['t_anticipation_spikes'].append(da['t_anticipation_spikes_filtered'])

    return d


if __name__ == '__main__':
    """
    python PlottingScripts/baseline_anticipation.py TestSim_baseLine__tauiAMPA_5_NMDA_5_nExcPerMc32_Etgt0.00_Itgt0.00_ratio1.0_wei2.0_wie-10.0_wii-1.00_wNoiseExc*
    """

    folder_names = sys.argv[1:]

    d = get_tanticipation_data(folder_names)

    markers = ['o', 'v', 'D', '*', 's', '^', '<', 'x', '>', '+', '1']
    colors = ['b', 'g', 'r']
    linestyles = ['-', '--', ':', '_']

    tauis = [(5, 5), (5, 150), (150, 150)]
    plot_params = {'backend': 'png',
                  'axes.labelsize': 28,
                  'axes.titlesize': 28,
                  'text.fontsize': 18,
                  'xtick.labelsize': 22,
                  'ytick.labelsize': 22,
                  'legend.pad': 0.2,     # empty space around the legend box
                  'legend.fontsize': 20,
                   'lines.markersize': 1,
                   'lines.markeredgewidth': 0.,
                   'lines.linewidth': 2,
                  'font.size': 12,
                  'path.simplify': False,
                  'figure.subplot.left':.14,
                  'figure.subplot.bottom':.15,
                  'figure.subplot.right':.97,
                  'figure.subplot.top':.86,
                  'figure.subplot.hspace':.30,
                  'figure.subplot.wspace':.30}

    pylab.rcParams.update(plot_params)

    figsize = utils.get_figsize(800, portrait=False)
    fig = pylab.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

       
    for i_, taui_pair in enumerate(tauis):
        for j_, taui_value in enumerate(d['taui_ampa_nmda']):
#            print 'debug', taui_pair, taui_value
            if taui_pair == taui_value:
#                print 'data', d['taui_ampa_nmda'][i_], d['t_anticipation_vmem'][j_]
                x_value = d['w_noise_exc'][j_]
                y_value = -1. * d['t_anticipation_spikes'][j_]
                print 'debug x, y', x_value, y_value
                ax1.plot(x_value, y_value, markers[i_], c=colors[i_], ms=13)

                y_value = -1. * d['t_anticipation_vmem'][j_]
                ax2.plot(x_value, y_value, markers[i_], c=colors[i_], ms=13)

#    ax1.set_xlabel('w_noise_exc')
    ax2.set_xlabel('$w_{noise}$ [nS]')
    ax1.set_title('Baseline network (no recurrency)\nAnticipation measured in spike response')
    ax2.set_title('Anticipation measured in membrane potential')
    ax1.set_ylabel('$t_{anticipation}^{spikes}$')
    ax2.set_ylabel('$t_{anticipation}^{vmem}$')

    ylim1 = (10, 22)
    ax1.set_ylim(ylim1)
    ylim2 = (45, 75)
    ax2.set_ylim(ylim2)

    output_fn = 'baseline_anticiption_vs_noise.png'
    print 'output_fn:', output_fn
    fig.savefig(output_fn, dpi=200)

    pylab.show()

