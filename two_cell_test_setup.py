import os
import numpy as np
import nest
import pylab
import utils
from PlottingScripts.AnalyseTraining import TrainingAnalyser


def setup():
    if (not 'bcpnn_synapse' in nest.Models('synapses')):
        nest.Install('pt_module')

    nest.SetKernelStatus({'data_path':'TwoCellTestSetup/', 'resolution': .1, 'overwrite_files' : True})

    params = {}
    params['t_sim'] =  120.
    params['figures_folder'] = 'TwoCellTestSetup/'
    if not os.path.exists(params['figures_folder']):
        cmd = 'mkdir %s' % params['figures_folder']
        print 'cmd:', cmd
        os.system(cmd)

    #initial_weight = np.log(nest.GetDefaults('bcpnn_synapse')['p_ij']/(nest.GetDefaults('bcpnn_synapse')['p_i']*nest.GetDefaults('bcpnn_synapse')['p_j']))
    #initial_bias = np.log(nest.GetDefaults('bcpnn_synapse')['p_j'])

    fmax_bcpnn = 60.0
    taup_bcpnn = 10000.
    epsilon = 1 / (fmax_bcpnn * taup_bcpnn)
    params['epsilon'] = epsilon
    params['bcpnn_params'] = {
                  'K': 1.0, \
                  'bias': 0., \
                  'delay': 1.0, \
                  'weight': 1.0, \
                  'epsilon': epsilon, \
                  'fmax': fmax_bcpnn, \
                  'gain': 0.0, \
                  'p_i': epsilon, \
                  'p_j': epsilon, \
                  'p_ij': epsilon, \
                  'tau_i': 2000., \
                  'tau_j': 10., \
                  'tau_e': 10., \
                  'tau_p': taup_bcpnn \
                  }

    nest.SetDefaults('bcpnn_synapse', params['bcpnn_params'])
    print 'Syn params:', params['bcpnn_params']
    return params

def create_neurons(params):
    w_input_exc = 3000.
    nest.CopyModel('static_synapse', 'input_exc_0', {'weight': w_input_exc, 'receptor_type': 0})  # numbers must be consistent with cell_params_exc

    neurons = nest.Create("iaf_psc_exp_multisynapse", 2) # GID 1 - 2
#    neuron2 = nest.Create("iaf_psc_exp_multisynapse", 1) # GID 2
    spike_shift = 50
    n_spikes = 3
    spike_times_1 = np.around(np.linspace(50., 60., n_spikes, endpoint=True), decimals=1)
    spike_times_2 = spike_times_1 + spike_shift
    spike_times_1 = np.r_[spike_times_1, np.array([params['t_sim'] - 5.])]
    print 'spike_times_1:', spike_times_1
    print 'spike_times_2:', spike_times_2
    neuron1_spike_gen = nest.Create('spike_generator', params={'spike_times': spike_times_1})
    neuron2_spike_gen = nest.Create('spike_generator', params={'spike_times': spike_times_2})
    voltmeter = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval' :0.5, "to_file": True, "withtime": True, 'label' : 'exc_volt'})
    nest.ConvergentConnect(voltmeter, neurons)

    exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
    nest.ConvergentConnect(neurons, exc_spike_recorder)
    nest.Connect(neuron1_spike_gen, [neurons[0]], model='input_exc_0')
    nest.Connect(neuron2_spike_gen, [neurons[1]], model='input_exc_0')

    nest.Connect([neurons[0]], [neurons[1]], model="bcpnn_synapse", params=params['bcpnn_params'])

    return neurons
    #nest.DivergentConnect(neuron1, neuron2, model="bcpnn_synapse")


def plot_bcpnn_traces(params, spike_fn):
    exc_spike_data = np.loadtxt(spike_fn)
    Plotter = TrainingAnalyser(params, it_max=1)
    pre_gid = 1
    post_gid = 2
    Plotter.plot_bcpnn_traces(exc_spike_data, pre_gid, post_gid, params['bcpnn_params'])


def plot_volt(volt_fn):
    set_rcParams()
    print 'Loading', volt_fn
    d = np.loadtxt(volt_fn)
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    gids = np.unique(d[:, 0])
    for gid in gids:
        time_axis, volt = utils.extract_trace(d, gid)
        ax.plot(time_axis, volt, label='%d' % gid, lw=2)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Voltage [mV]')
    output_fn = params['figures_folder'] + 'voltage_traces.png'
    print 'Saving to:', output_fn
    pylab.savefig(output_fn, dpi=300)



def get_weight(neurons):
    conns = nest.GetConnections([neurons[0]], [neurons[1]])
    cp = nest.GetStatus([conns[0]])
    w = cp[0]['weight']
    print 'cp', cp
    pi = cp[0]['p_i']
    pj = cp[0]['p_j']
    pij = cp[0]['p_ij']
    print 'NEST weight:', w, 'log weight:', np.log(pij / (pi * pj))



def get_p_values(neurons):
    conns = nest.GetConnections([neurons[0]], [neurons[1]])
    cp = nest.GetStatus([conns[0]])
    pi = cp[0]['p_i']
    pj = cp[0]['p_j']
    pij = cp[0]['p_ij']
    wij = cp[0]['weight']
    return pi, pj, pij, wij


def plot_traces(t_axis, pi, pj, pij, wij_nest):

    wij = np.log(pij / (pi * pj))
    bias = np.log(pj)
    fig = pylab.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title('Traces retrieved from NEST module')
    plots = []
    p1, = ax1.plot(t_axis, pi)
    p2, = ax1.plot(t_axis, pj)
    p3, = ax1.plot(t_axis, pij)
    plots += [p1, p2, p3]
    labels = ['$p_i$', '$p_j$', '$p_{ij}$']
    ax1.legend(plots, labels, loc='upper left')
    ax1.set_ylabel('p values')


    plots = []
    p1, = ax2.plot(t_axis, wij)
    p2, = ax2.plot(t_axis, wij_nest)
    plots += [p1, p2]
    labels = ['$w=\log(p_{ij} / (p_i \cdot p_j))$', '$w_{NEST}$']
    ax2.set_ylabel('Weight')
    ax2.legend(plots, labels, loc='upper left')


    plots = []
    p1, = ax3.plot(t_axis, bias)
    plots += [p1]
    ax3.set_ylabel('Bias')
    ax3.set_xlabel('Time [ms]')
    ax3.legend(plots, ['Bias'], loc='upper left')

    output_fn = params['figures_folder'] + 'nest_traces.png'
    print 'Saving to:', output_fn
    pylab.savefig(output_fn, dpi=300)


def set_rcParams():


    plot_params = {'backend': 'png',
                  'axes.labelsize': 16,
                  'axes.titlesize': 16,
                  'text.fontsize': 16,
                  'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'legend.pad': 0.2,     # empty space around the legend box
                  'legend.fontsize': 14,
                   'lines.markersize': 0,
                   'lines.linewidth': 3,
                  'font.size': 12,
                  'path.simplify': False,
                  'figure.subplot.hspace':.40,
                  'figure.subplot.wspace':.10,
                  'figure.subplot.left':.15,
                  'figure.subplot.bottom':.10, 
                  'figure.subplot.right':.90,
                  'figure.subplot.top':.90}
    #              'figure.figsize': get_fig_size(800)}

    pylab.rcParams.update(plot_params)

if __name__ == '__main__':

    params = setup()
    neurons = create_neurons(params)

    tracking = False
    if tracking:
        # get the p values from the nest module at t_step
        t = 0
        t_step = .1
        time = np.arange(0, params['t_sim'], t_step)
        pi_nest = np.ones(time.size) * params['epsilon']
        pj_nest = np.ones(time.size) * params['epsilon']
        pij_nest = np.ones(time.size) * params['epsilon'] ** 2
        wij_nest = np.log(pij_nest / (pi_nest * pj_nest))
        for i_, t in enumerate(time):
            nest.Simulate(t_step)
            pi, pj, pij, wij = get_p_values(neurons)
            pi_nest[i_] = pi
            pj_nest[i_] = pj
            pij_nest[i_] = pij
            wij_nest[i_] = wij
        plot_traces(time, pi_nest, pj_nest, pij_nest, wij_nest)
        get_weight(neurons)

    else:
        # calculate the traces 'offline' based on the spikes
        nest.Simulate(params['t_sim'])
        get_weight(neurons)
        volt_fn = 'TwoCellTestSetup/exc_volt-5-0.dat'
        spike_fn = 'TwoCellTestSetup/exc_spikes-6-0.gdf'
        plot_bcpnn_traces(params, spike_fn)
        plot_volt(volt_fn)

#    pylab.show()

