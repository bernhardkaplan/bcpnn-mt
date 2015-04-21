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

    return d


def get_tanticipation_value(folder_names, taui_tuple):

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
        if params['taui_ampa'] == taui_tuple[0] and params['taui_nmda'] == taui_tuple[1]:
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

    return d



if __name__ == '__main__':
    """
    python PlottingScripts/collect_anticiaption_data [FOLDER_NAMES]
    """

    folder_names = sys.argv[1:]

    d = get_tanticipation_data(folder_names)

    markers = ['o', 'v', 'D', '*', 's', '^', '<', 'x', '>', '+', '1']
    colors = ['b', 'g', 'r']
    linestyles = ['-', '--', ':']


    plot_params = {'backend': 'png',
                  'axes.labelsize': 28,
                  'axes.titlesize': 28,
                  'text.fontsize': 18,
                  'xtick.labelsize': 22,
                  'ytick.labelsize': 22,
                  'legend.pad': 0.2,     # empty space around the legend box
                  'legend.fontsize': 18,
                  'lines.markersize': 1,
                  'lines.markeredgewidth': 0.,
                  'lines.linewidth': 2,
                  'font.size': 12,
                  'path.simplify': False,
                  'figure.figsize': utils.get_figsize(1000, portrait=False),
                  'figure.subplot.left':.17,
                  'figure.subplot.bottom':.15,
                  'figure.subplot.right':.90,
                  'figure.subplot.top':.85,
                  'figure.subplot.hspace':.45,
                  'figure.subplot.wspace':.30}
    pylab.rcParams.update(plot_params)

    fig = pylab.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

#    tauis = [(5., 5.), (5., 150.), (150., 150.)]
    tauis = []
    for j_, taui_value in enumerate(d['taui_ampa_nmda']): #            print 'debug', taui_pair, taui_value if taui_pair == taui_value:
        if taui_value not in tauis:
            tauis.append(taui_value)
    print 'found tauis:', tauis
    plots1 = []
    plots2 = []
    labels1 = []
    n_data = d['taui_ampa_nmda'].count(tauis[0])
    data = np.zeros((len(tauis), n_data, 3))
        
    for j_, taui_tuple in enumerate(tauis):  
        d = get_tanticipation_value(folder_names, taui_tuple)
        data[j_, :, 0] = d['g_exc_total_in']
        data[j_, :, 1] = -1. * np.array(d['t_anticipation_spikes'])
        data[j_, :, 2] = -1. * np.array(d['t_anticipation_vmem'])
    print 'data:', data
    
    for j_, taui_tuple in enumerate(tauis):  
        label='$\\tau_i^{AMPA}=%d\ \\tau_i^{NMDA}=%d$' % (taui_tuple[0], taui_tuple[1])
        p1, = ax1.plot(data[j_, :, 0], data[j_, :, 1], lw=4, ls=linestyles[j_], marker=markers[j_], ms=15, c=colors[j_])
        p2, = ax2.plot(data[j_, :, 0], data[j_, :, 2], lw=4, ls=linestyles[j_], marker=markers[j_], ms=15, c=colors[j_])
        plots1.append(p1)
        plots2.append(p2)
        labels1.append(label)

    ax1.legend(plots1, labels1, numpoints=1, loc='upper left')
    ax2.legend(plots2, labels1, numpoints=1, loc='upper left')


#    ax1.set_xlabel('w_noise_exc')
#    pylab.legend(scatterpoints=1, ncol=1)
    ax2.set_xlabel('$w_{exc, in}^{total}$')
    ax1.set_title('Anticipation measured in spike response')
    ax2.set_title('Anticipation measured in membrane potential')
    ax1.set_ylabel('$t_{anticipation}^{spikes}$')
    ax2.set_ylabel('$t_{anticipation}^{vmem}$')

    output_fn = 'anticiption_vs_gexc.png'
    print 'output_fn:', output_fn
    fig.savefig(output_fn, dpi=200)

    pylab.show()

