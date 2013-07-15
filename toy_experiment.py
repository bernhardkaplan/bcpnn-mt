
import numpy as np
import numpy.random as nprnd
import sys
import os
import utils
import nest
import CreateStimuli
import json
import pylab
import re
import Bcpnn


class ToyExperiment(object):

    def __init__(self, params, output_folder, selected_gids, bcpnn_params=None):

        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.system('mkdir %s' % output_folder)
        nest.SetKernelStatus({'data_path': output_folder, 'resolution': .1, 'overwrite_files': True})
        self.selected_gids = selected_gids
        self.params = params
        nprnd.seed(0)
        if bcpnn_params == None:
            self.bcpnn_params = {'tau_i': 10., 'tau_j': 10.0, 'tau_e': 100., 'tau_p': 1000., 'fmax':50.}
        else:
            self.bcpnn_params = bcpnn_params
        self.dx = None


    def run_sim(self, t_sim):
        self.t_sim = t_sim
        
        # create two cells with matching ot not matching tuning properties
        x0, y0, u0, v0 = .3, .0, .5, .5

        if self.dx == None:
            self.dx = u0

        tp_matching = np.array([[x0, y0, u0, v0, .0], \
                    [x0 + self.dx, y0, u0, v0, .0]])
        tp_not_matching = np.array([[x0, y0, u0, v0, .0], \
                    [x0 + self.dx, y0, -u0, v0, .0]])
        tp_s = tp_matching

        print 'Cells\' tuning properties:', tp_s


        # choose the motion - parameters for the test stimulus
        stim_id = 0
#        mp = np.loadtxt(self.params['training_sequence_fn'])[stim_id, :]  # (x, v)
        mp = [.2, .5]


        n_cells = len(self.selected_gids)
        time = np.arange(0, self.t_sim, self.params['dt_rate'])
        self.L_input = np.zeros((n_cells, time.shape[0]))  # the envelope of the Poisson process
        input_spiketrains = []
        
        load_input = False
        # get the input signal
        if not load_input:
            # create the input
            time = np.arange(0, self.t_sim, self.params['dt_rate'])
            idx_t_stop = int(self.t_sim / self.params['dt_rate'])
#            idx_t_stop = int(self.params['t_training_stim'] / self.params['dt_rate'])
            for i_time in xrange(0, idx_t_stop):
                time_ = (i_time * self.params['dt_rate']) / self.params['t_stimulus']
                x_stim = mp[0] + time_ * mp[1]
                self.L_input[:, i_time] = utils.get_input(tp_s, self.params, (x_stim, 0, mp[1], 0, 0))
                self.L_input[:, i_time] *= self.params['f_max_stim']
                if (i_time % 10000 == 0):
                    print "t: %.2f [ms]" % (time_ * self.params['t_stimulus'])

            for i_ in xrange(n_cells):
                rate_of_t = np.array(self.L_input[i_, :])
                n_steps = rate_of_t.size
                spike_times = []
                for i in xrange(n_steps):
                    r = nprnd.rand()
                    if (r <= ((rate_of_t[i]/1000.) * self.params['dt_rate'])): # rate is given in Hz -> 1/1000.
                        spike_times.append(i * self.params['dt_rate'])
                input_spiketrains.append(spike_times)
            

        else: 
            for i_, gid in enumerate(self.selected_gids):
                input_fn = self.params['input_st_fn_base'] + str(gid) + '.npy'
                input_spiketrains.append(np.load(input_fn))
                input_fn = self.params['input_rate_fn_base'] + str(gid) + '.npy'
                self.L_input[i_, :] = np.load(input_fn)


        
        # NEST - code 
        # Setup 
        self.params['w_input_exc'] = 1.  # [nS]
        nest.CopyModel('static_synapse', 'input_exc_0', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
        nest.CopyModel('static_synapse', 'input_exc_1', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 1})

        if (not 'bcpnn_synapse' in nest.Models('synapses')):
            nest.Install('pt_module')


        # Create cells
        cell_params = self.params['cell_params_exc'].copy()
        cells = nest.Create(self.params['neuron_model'], n_cells, params=cell_params)


        # Create stimulus
        stimulus = nest.Create('spike_generator', n_cells)
        for i_ in xrange(n_cells):
            spike_times = input_spiketrains[i_]
            nest.SetStatus([stimulus[i_]], {'spike_times' : spike_times})
            nest.Connect([stimulus[i_]], [cells[i_]], model='input_exc_0')
            nest.Connect([stimulus[i_]], [cells[i_]], model='input_exc_1')
        

        # Record
        voltmeter = nest.Create('voltmeter')
        nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True, 'label' : 'volt'}])
        exc_spike_recorder = nest.Create('spike_detector', params={'to_file':True, 'label':'exc_spikes'})
        nest.ConvergentConnect(cells, exc_spike_recorder)
        nest.ConvergentConnect(voltmeter, cells)

        
        # Connect cells in a chain:
        initial_weight = np.log(nest.GetDefaults('bcpnn_synapse')['p_ij']/(nest.GetDefaults('bcpnn_synapse')['p_i']*nest.GetDefaults('bcpnn_synapse')['p_j']))
        print '\n\nDebug initial weight NEST BCPNN: \t', initial_weight
        initial_bias = np.log(nest.GetDefaults('bcpnn_synapse')['p_j'])

        self.syn_params = {'weight': initial_weight, 'bias': initial_bias, 'K': 1.0, 'delay': 1.0,\
                'tau_i': self.bcpnn_params['tau_i'], 'tau_j': self.bcpnn_params['tau_j'], \
                'tau_e': self.bcpnn_params['tau_e'], 'tau_p': self.bcpnn_params['tau_p'], \
                'p_i': 0.01, 'p_j': 0.01, 'p_ij': 0.0001}

        for i_ in xrange(n_cells - 1):
            nest.Connect([cells[i_]], [cells[i_ + 1]], model='bcpnn_synapse')

            # modify the parameters
            nest.SetStatus(nest.GetConnections([cells[i_]], [cells[i_ + 1]]), self.syn_params)


        # Simulate
        nest.Simulate(self.t_sim)

        # Get weights
        for i_ in xrange(n_cells - 1):
            cp = nest.GetStatus(nest.GetConnections([cells[i_]], [cells[i_ + 1]]))[0]
            print 'cp:', cp
            w_ij = cp['weight']
            print 'Weight %d - %d after sim = %.5e'  % (i_, i_ + 1, w_ij)



    def plot_input(self, ax):

        time_axis = np.arange(0, self.t_sim, self.params['dt_rate'])
        plots = []
        for i_, gid in enumerate(self.selected_gids):
            p, = ax.plot(time_axis, self.L_input[i_, :], label='%d', lw=2)
            plots.append(p)
        ax.legend(plots, self.selected_gids)



    def plot_voltages(self, ax):
        to_match = 'volt-'
        for fn in os.listdir(self.output_folder):
            m = re.match(to_match, fn)
            if m:
                volt_path = self.output_folder + fn

        d = np.loadtxt(volt_path)
        
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Voltage [mV]')
        gids = np.unique(d[:, 0])
        plots = []
        for gid in gids:
            time_axis, volt = utils.extract_trace(d, gid)
            p, = ax.plot(time_axis, volt, label='%d' % gid, lw=2)
            plots.append(p)
        ax.legend(plots, self.selected_gids)




    def get_bcpnn_traces_from_spiketrain(self, spike_fn=None):

        if spike_fn == None:
            fns = utils.get_spike_fns(output_folder, spike_fn_pattern='exc_spikes')
            spike_fn = fns[0]

        spike_train = np.loadtxt(self.output_folder + spike_fn)
        spike_trains = utils.get_spiketrains(spike_train)
        # spike_trains[0] is for gid == 0

        # add delay for test reasons
#        spike_trains[1] = np.array(spike_trains[1])
#        spike_trains[2] = np.array(spike_trains[2])
#        spike_trains[1] += 1
#        spike_trains[2] += 1

        # convert the spike trains to a binary trace
        dt = 0.1
        if (spike_train[:, 0] == 2).nonzero()[0].size == 0:
            post_trace = np.zeros(self.t_sim / dt)
        else:
            post_trace = utils.convert_spiketrain_to_trace(spike_trains[2], t_max=self.t_sim, dt=dt, spike_width=1. / dt)

        if (spike_train[:, 0] == 1).nonzero()[0].size == 0:
            pre_trace = np.zeros(self.t_sim / dt)
        else:
            pre_trace = utils.convert_spiketrain_to_trace(spike_trains[1], t_max=self.t_sim, dt=dt, spike_width=1. / dt)


        tau_dict = {'tau_zi' : self.bcpnn_params['tau_i'], 'tau_zj' : self.bcpnn_params['tau_j'], 
                    'tau_ei' : self.bcpnn_params['tau_e'], 'tau_ej' : self.bcpnn_params['tau_e'], 'tau_eij' : self.bcpnn_params['tau_e'],
                    'tau_pi' : self.bcpnn_params['tau_p'], 'tau_pj' : self.bcpnn_params['tau_p'], 'tau_pij' : self.bcpnn_params['tau_p'],
                    }
        fmax = self.bcpnn_params['fmax']

        # compute
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace, tau_dict=tau_dict, fmax=fmax, dt=dt)

        
        t_axis = dt * np.arange(zi.size)
        # save
        np.savetxt(self.output_folder + 'w_ij.txt', np.array((t_axis, wij)).transpose() )
        np.savetxt(self.output_folder + 'bias.txt', np.array((t_axis, bias)).transpose() )
        np.savetxt(self.output_folder + 'pi.txt', np.array((t_axis, pi)).transpose() )
        np.savetxt(self.output_folder + 'pj.txt', np.array((t_axis, pj)).transpose() )
        np.savetxt(self.output_folder + 'pij.txt', np.array((t_axis, pij)).transpose() )
        np.savetxt(self.output_folder + 'ei.txt', np.array((t_axis, ei)).transpose() )
        np.savetxt(self.output_folder + 'ej.txt', np.array((t_axis, ej)).transpose() )
        np.savetxt(self.output_folder + 'eij.txt', np.array((t_axis, eij)).transpose() )
        np.savetxt(self.output_folder + 'zi.txt', np.array((t_axis, zi)).transpose() )
        np.savetxt(self.output_folder + 'zj.txt', np.array((t_axis, zj)).transpose() )

        self.w_end = wij[-1]
        self.w_max = wij.max()
        self.t_max = wij.argmax() * dt
        print 'BCPNN offline computation:'
        print '\tw_ij : %.4e\tt_wmax: %d' % (self.w_end, self.t_max)
        print '\tp_i  :', pi[-1]
        print '\tp_j  :', pj[-1]
        print '\tp_ij :', pij[-1]

        plots = []

        pylab.rcParams.update({'figure.subplot.hspace': 0.30})
        fig = pylab.figure(figsize=utils.get_figsize(1200, portrait=False))

        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)

        self.title_fontsize = 24
        ax1.set_title('$\\tau_{z_i} = %d$ ms' % (self.bcpnn_params['tau_i']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c='k', lw=2)
        ax1.plot(t_axis, post_trace, c='k', lw=2)
        p1, = ax1.plot(t_axis, zi, c='b', label='$z_i$', lw=2)
        p2, = ax1.plot(t_axis, zj, c='g', label='$z_j$', lw=2)
        plots += [p1, p2]
        labels_z = ['$z_i$', '$z_j$']
        ax1.legend(plots, labels_z)
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')

        plots = []
        p1, = ax2.plot(t_axis, pi, c='b', lw=2)
        p2, = ax2.plot(t_axis, pj, c='g', lw=2)
        p3, = ax2.plot(t_axis, pij, c='r', lw=2)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax2.legend(plots, labels_p)
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c='b', lw=2)
        p2, = ax3.plot(t_axis, ej, c='g', lw=2)
        p3, = ax3.plot(t_axis, eij, c='r', lw=2)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.legend(plots, labels_p)
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')


        plots = []
        p1, = ax4.plot(t_axis, wij, c='b', lw=2)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w)
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')

        plots = []
        p1, = ax6.plot(t_axis, bias, c='b', lw=2)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_)
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Weight')


        ax5.set_yticks([])
        ax5.set_xticks([])
        ax5.annotate('Weight max: %.3e\nWeight end: %.3e\nt(w_max): %.1f [ms]' % (self.w_max, self.w_end, self.t_max * dt), (.1, .2), fontsize=20)

#        ax5.set_xticks([])


        output_fn = self.output_folder + 'traces_tauzi_%04d_tauzj_%04d.png' % (self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j'])
        print 'Saving traces to:', output_fn
        pylab.savefig(output_fn)


#        for gid in gids:
#            time_axis, volt = utils.extract_trace(d, gid)
#            p, = ax.plot(time_axis, volt, label='%d' % gid, lw=2)
#            plots.append(p)
#        ax.legend(plots, self.selected_gids)




if __name__ == '__main__':

    print '\nPlotting the default parameters give in simulation_parameters.py\n'
    import simulation_parameters
    ps = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = ps.load_params()                       # params stores cell numbers, etc as a dictionary

#    if len(sys.argv) > 1:
#        selected_gids = [int(sys.argv[i]) for i in xrange(1, 1 + len(sys.argv[1:]))]

#    else:
#        selected_gids = np.loadtxt(params['gids_to_record_fn'], dtype=int)[:2]
    selected_gids = np.loadtxt(params['gids_to_record_fn'], dtype=int)[:2]
    tau_zi = float(sys.argv[1])
#    tau_zj = float(sys.argv[2])
    tau_zj = 10.
    bcpnn_params = {'tau_i': tau_zi, 'tau_j': tau_zj, 'tau_e': 100., 'tau_p': 1000., 'fmax':50.}

    print 'Selected gids:', selected_gids
    output_folder = 'ToyExperiment/'
    t_sim = 3000
    TE = ToyExperiment(params, output_folder, selected_gids, bcpnn_params)
    dx = float(sys.argv[2])
    TE.dx = dx

    TE.run_sim(t_sim)

    fig1 = pylab.figure()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)
    TE.plot_input(ax1)
    TE.plot_voltages(ax2)
    TE.get_bcpnn_traces_from_spiketrain()

    output_fn = 'sweep_data_tsim%d_tauzj%d.dat' % (t_sim, bcpnn_params['tau_j'])
    f = file(output_fn, 'a')
    str_to_write = '%.4e\t%.4e\t%.4e\t%.4e\t%.1f\n' % (dx, bcpnn_params['tau_i'], TE.w_max, TE.w_end,TE.t_max)
    f.write(str_to_write)
#    pylab.show()

