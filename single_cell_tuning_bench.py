import os
import sys
import matplotlib
import pylab
import numpy as np 
import utils
import json
import simulation_parameters
import functions
import nest



plot_params = {'backend': 'png',
              'axes.labelsize': 24,
              'axes.titlesize': 24,
              'text.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 11,
               'lines.markersize': 1,
               'lines.markeredgewidth': 0.,
               'lines.linewidth': 3,
              'font.size': 12,
              'path.simplify': False,
              'figure.subplot.left':.20,
              'figure.subplot.bottom':.14,
              'figure.subplot.right':.90,
              'figure.subplot.top':.90,
              'figure.subplot.hspace':.30,
              'figure.subplot.wspace':.30}
pylab.rcParams.update(plot_params)


class Experiment(object):

    def __init__(self, params):
        try:
            nest.sr('(/home/bernhard/workspace/BCPNN-Module/module-100725/sli) addpath')
            nest.Install('pt_module')
        except:
            nest.Install('pt_module')


        nest.SetKernelStatus({'data_path': params['spiketimes_folder'], 'resolution': params['dt_sim'], 'overwrite_files' : True})
        n_neurons = 2
        neurons = nest.Create(params['neuron_model'], n_neurons, params=params['cell_params_exc'])
        dt_volt = 0.5

#        record_from = ['V_m', 'I_AMPA', 'I_NMDA', 'I_NMDA_NEG', 'I_AMPA_NEG', 'I_GABA']
        record_from = ['V_m', 'I_AMPA', 'I_NMDA']
#        record_from = ['I_AMPA', 'I_NMDA']
        recorder = nest.Create('multimeter', params={'record_from': record_from, 'interval': dt_volt})
        nest.SetStatus(recorder, [{"to_file": True, "withtime": True}])
        nest.DivergentConnect(recorder, neurons)

        input_spikes = nest.Create('spike_generator', 1)
        spike_times = [50.]

        nest.SetStatus([input_spikes[0]], {'spike_times' : spike_times})
        weight_bcpnn = 2.
        ampa_nmda_ratio = 5.
        w_ampa = weight_bcpnn * params['bcpnn_gain'] / params['tau_syn']['ampa']
        nest.CopyModel('static_synapse', 'ampa_synapse', \
                {'weight': w_ampa, 'delay': 0.1, 'receptor_type': params['syn_ports']['ampa']})  # numbers must be consistent with cell_params_exc

        w_nmda = weight_bcpnn * params['bcpnn_gain'] / (ampa_nmda_ratio * params['tau_syn']['nmda'])
        nest.CopyModel('static_synapse', 'nmda_synapse', \
                {'weight': w_nmda, 'delay': 0.1, 'receptor_type': params['syn_ports']['nmda']})  # numbers must be consistent with cell_params_exc

        nest.Connect(input_spikes, [neurons[0]], model='ampa_synapse')
        nest.Connect(input_spikes, [neurons[1]], model='nmda_synapse')
        t_sim = 2000.
        nest.Simulate(t_sim)
        gid_vec = nest.GetStatus(recorder)[0]['events']['senders']
        gids = np.unique(gid_vec)
        time = nest.GetStatus(recorder, 'events')[0]['times']

        fig = pylab.figure(figsize=utils.get_figsize_A4(portrait=True))
        n_plots = len(record_from)
#        integrals = np.zeros((n_plots, 2))
        integrals = {}
        integrals['V_m'] = {}
        integrals['I_AMPA'] = {}
        integrals['I_NMDA'] = {}
        title = 'Effect of AMPA and NMDA currents (%.1f) \n$R(\\frac{w_{AMPA}}{w_{NMDA}})=\\frac{%.2e}{%.2e}=%.2f$' % (ampa_nmda_ratio, w_ampa, w_nmda, w_ampa/w_nmda)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.set_title(title)
        ax1.set_ylabel('$V_m(t)$ [mV]')
        ax2.set_ylabel('$I(t)$ [pA]')

        v = nest.GetStatus(recorder)[0]['events']['V_m']
        i_ampa = nest.GetStatus(recorder)[0]['events']['I_AMPA']
        i_nmda = nest.GetStatus(recorder)[0]['events']['I_NMDA']

        for j_, gid in enumerate(gids):
            print 'GID:', gid
            idx = np.where(gid_vec == gid)[0]
            integrals['I_AMPA'][gid] = i_ampa[idx].sum() * dt_volt
            integrals['I_NMDA'][gid] = i_nmda[idx].sum() * dt_volt
            if gid == neurons[0]:
                label_ampa = '$\int\ dt\ I_{AMPA}(t) = %.3e\ pA \cdot ms$' % (integrals['I_AMPA'][gid])
                ax2.plot(time[idx], i_ampa[idx], label=label_ampa)
            elif gid == neurons[1]:
                label_nmda = '$\int\ dt\ I_{NMDA}(t) = %.3e\ pA \cdot ms$' % (integrals['I_NMDA'][gid])
                ax2.plot(time[idx], i_nmda[idx], label=label_nmda)
            ax1.plot(time[idx], v[idx])


        print 'Integrals:', integrals
        ratio_nmda_ampa = integrals['I_NMDA'][2] / integrals['I_AMPA'][1] 
        info_txt = 'Ratio of integrated currents (NMDA / AMPA):\n%.2e / %.2e = %.3f' % (integrals['I_NMDA'][2], integrals['I_AMPA'][1], ratio_nmda_ampa)
#        ax1 = fig.get_axes()[0]
        ax2.set_title(info_txt)
        print info_txt
        print 'w_nmda:', w_nmda
        print 'w_ampa:', w_ampa

        ax2.set_xlabel('Time [ms]')
        ax1.set_xlim((0., 500.))
        ax2.set_xlim((0., 500.))
        pylab.legend()
        pylab.show()





if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'Case 1: default parameters'
        GP = simulation_parameters.parameter_storage()
        params = GP.params
    elif len(sys.argv) == 2:
        params = utils.load_params(sys.argv[1])

    E = Experiment(params)
    
