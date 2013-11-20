import numpy as np
import numpy.random as nprnd
import sys
import os
import utils
import nest
import json
import matplotlib
matplotlib.use('Agg')
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
        self.n_cells = len(self.selected_gids)
        self.params = params
        nprnd.seed(0)
        if bcpnn_params == None:
            self.bcpnn_params = {'tau_i': 10., 'tau_j': 10.0, 'tau_e': 100., 'tau_p': 1000., 'fmax':200.}
        else:
            self.bcpnn_params = bcpnn_params

        self.dx = None
        self.v_stim = None
        self.dv = None


    def run_sim(self, t_sim=None, tp_0=None):

        # set parameters
        if self.dx == None:
            self.dx = u0
        if self.dv == None:
            self.dv = 0
        if self.v_stim == None:
            self.v_stim = .5
        if tp_0 == None:
            x0, y0, u0, v0 = .5, .0, self.v_stim, .0
        else:
            x0, y0, u0, v0 = tp_0[0], .0, tp_0[1], .0
        # create two cells with matching or not matching tuning properties
        tp_matching = np.array([[x0, y0, u0, v0, .0], \
                    [x0 + self.dx, y0, u0 + self.dv, v0, .0]])
        tp_not_matching = np.array([[x0, y0, u0, v0, .0], \
                    [x0 + self.dx, y0, -u0, v0, .0]])
        self.tp_s = np.array(tp_matching)
        self.mp = [.0, self.v_stim] # choose the motion - parameters for the test stimulus

        # define how long a simulation takes based on the stimulus duration and the speed of the stimulus
#        n_stim = 1
        n_stim = 1
        dt_stim = 3 * abs(self.dx) / self.v_stim * 1000.# time between cells see the stimulus
#        t_z_decay = 12 * max(self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j']) 
        t_z_decay = 8 * max(self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j']) 
        # t_z_decay is the time for the z_traces to decay between stimuli
        if t_sim == None:
            if (t_z_decay <= dt_stim):
                stim_duration = 4 * dt_stim 
            elif (t_z_decay > dt_stim):
                stim_duration = t_z_decay
            t_sim = n_stim * stim_duration + 1 # +1 to allow odd stim_durations and indexing errors due to rounding
        self.steps_for_wavg = int(stim_duration / .1)
        print 'Simulating:', t_sim
        print 'steps fo weight averaging:', self.steps_for_wavg
        self.t_sim = t_sim
        time = np.arange(0, self.t_sim, self.params['dt_rate'])
        self.L_input = np.zeros((self.n_cells, time.size))  # the envelope of the Poisson process
        self.input_spiketrains = []
        
        save_input = True
        self.build_input_filenames()
        load_input = self.check_input_files_exist()
        # get the input signal
        if not load_input:
#            stim_duration = int(10. * self.bcpnn_params['tau_i']) # how many time steps within one stimulus
            print 'stim_duration:', stim_duration
            print 't_sim:', self.t_sim
            print 'n_stim:', n_stim
            print 'v_stim:', v_stim

            global_time = 0.
            global_time_idx = 0
            for stim_cnt in xrange(n_stim):
                stim_time_axis = np.arange(0, stim_duration, self.params['dt_rate'])
                for t_ in stim_time_axis:
                    x_stim = self.mp[0] + t_ / self.params['t_stimulus'] * self.mp[1]
                    assert global_time_idx < self.L_input[0, :].size, 'invalid global_time_idx\ntime.size=%d' % time.size
                    self.L_input[:, global_time_idx] = utils.get_input(self.tp_s, self.params, (x_stim, 0, self.mp[1], 0, 0))
                    self.L_input[:, global_time_idx] *= self.params['f_max_stim']
                    if (global_time_idx % 10000 == 0):
                        print "t: %.2f [ms]" % (global_time)
                    global_time += self.params['dt_rate']
                    global_time_idx += 1

            for i_ in xrange(self.n_cells):
                rate_of_t = np.array(self.L_input[i_, :])
                n_steps = rate_of_t.size
                spike_times = []
                for i in xrange(n_steps):
                    r = nprnd.rand()
                    if (r <= ((rate_of_t[i]/1000.) * self.params['dt_rate'])): # rate is given in Hz -> 1/1000.
                        spike_times.append(i * self.params['dt_rate'])
                self.input_spiketrains.append(spike_times)
            
        else: 
            for i_, gid in enumerate(self.selected_gids):
                input_fn = self.input_st_fns[i_]
                self.input_spiketrains.append(np.loadtxt(input_fn))
                input_fn = self.input_rate_fns[i_]
                self.L_input[i_, :] = np.loadtxt(input_fn)

        if save_input:
            for i_, gid in enumerate(self.selected_gids):
                input_fn = self.input_st_fns[i_]
                np.savetxt(input_fn, self.input_spiketrains[i_])
                input_fn = self.input_rate_fns[i_]
                np.savetxt(input_fn, self.L_input[i_, :])
        
        # NEST - code 
        # Setup 
        self.params['w_input_exc'] = 50.0  # [nS]
        nest.CopyModel('static_synapse', 'input_exc_0', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 0})  # numbers must be consistent with cell_params_exc
        nest.CopyModel('static_synapse', 'input_exc_1', \
                {'weight': self.params['w_input_exc'], 'receptor_type': 1})

        if (not 'bcpnn_synapse' in nest.Models('synapses')):
            nest.Install('pt_module')


        # Create cells
        cell_params = self.params['cell_params_exc'].copy()
        cells = nest.Create(self.params['neuron_model'], self.n_cells, params=cell_params)


        # Create stimulus
        stimulus = nest.Create('spike_generator', self.n_cells)
        for i_ in xrange(self.n_cells):
            spike_times = self.input_spiketrains[i_]
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
                'p_i': 0.01, 'p_j': 0.01, 'p_ij': 0.0001, 'epsilon': self.bcpnn_params['fmax'] / self.bcpnn_params['tau_p']}

        for i_ in xrange(self.n_cells - 1):
            nest.Connect([cells[i_]], [cells[i_ + 1]], model='bcpnn_synapse')

            # modify the parameters
            nest.SetStatus(nest.GetConnections([cells[i_]], [cells[i_ + 1]]), self.syn_params)


        # Simulate
        nest.Simulate(self.t_sim)

        # Get weights
        for i_ in xrange(self.n_cells - 1):
            cp = nest.GetStatus(nest.GetConnections([cells[i_]], [cells[i_ + 1]]))[0]
            print 'cp:', cp
            w_ij = cp['weight']
            print 'Weight %d - %d after sim = %.5e'  % (i_, i_ + 1, w_ij)



    def build_input_filenames(self):
        """
        Build the filenames which should contain the input spiketrains for the cells
        based on their tuning properties.
        """
        self.input_st_fns = []
        self.input_rate_fns = []
        for i_, gid in enumerate(self.selected_gids):
            self.input_st_fns.append('%svstim%.2f_tsim%d_tp%.2f_%.2f.dat' % \
                (self.params['input_st_fn_base'], self.v_stim, self.t_sim, self.tp_s[i_][0], self.tp_s[i_][2]))
            self.input_rate_fns.append('%svstim%.2f_tsim%d_tp%.2f_%.2f.dat' % \
                (self.params['input_rate_fn_base'], self.v_stim, self.t_sim, self.tp_s[i_][0], self.tp_s[i_][2]))


    def check_input_files_exist(self):
        """
        Returns true if all files exist and can be loaded.
        """
        all_files_exist = np.zeros((self.n_cells, 2))
        for i_, gid in enumerate(self.selected_gids):
            input_fn = self.input_st_fns[i_]
            all_files_exist[i_, 0] = os.path.exists(input_fn)
            input_fn = self.input_rate_fns[i_]
            all_files_exist[i_, 1] = os.path.exists(input_fn)

        n_existing = (all_files_exist == 1).nonzero()[0].size
#        print 'All_files_exist:', all_files_exist
#        print 'Files existing:', n_existing
        if n_existing == 2 * self.n_cells: # 2 because checking for rate and spiketrain files
            print 'All input files exist\t Loading input...'
            return True
        else:
            print 'Not all input files exist\t Computing input...'
            return False
        



    def plot_input(self, ax):

        time_axis = np.arange(0, self.t_sim, self.params['dt_rate'])
        plots = []
        color_list = ['b', 'g', 'r', 'y', 'c', 'm', 'k', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']
        for i_, gid in enumerate(self.selected_gids):
            p, = ax.plot(time_axis, self.L_input[i_, :], label='%d', lw=2, c=color_list[i_ % len(color_list)])
            plots.append(p)

        ylim = ax.get_ylim()
#        for i_, gid in enumerate(self.selected_gids):
#            print 'Input spikes:', self.input_spiketrains[i_]
#            y_ = (np.ones(len(self.input_spiketrains[i_])) * ylim[0], np.ones(len(self.input_spiketrains[i_])) * ylim[1])
#            ax.plot((self.input_spiketrains[i_], self.input_spiketrains[i_]), (y_[0], y_[1]), c=color_list[i_ % len(color_list)])

#            ax.plot((self.input_spiketrains[i_], y_[0]), (self.input_spiketrains[i_], y_[1]), c=color_list[i_ % len(color_list)])

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

        print 'DEBUG fns', self.output_folder, spike_fn
        print 'Loading spiketrain from:', self.output_folder + spike_fn
        spike_train = np.loadtxt(self.output_folder + spike_fn)
        if spike_train.size == 0:
            spike_trains = [[], [], []]
        else:
#            print 'DEBUG spike_train:', spike_train
            spike_trains = utils.get_spiketrains(spike_train, n_cells=self.n_cells + 1) # +1 because NEST GIDs are 1-aligned
        # spike_trains[0] is for gid == 0
#        print 'Debug: spike_trains:', spike_trains

        # convert the spike trains to a binary trace
        dt = 0.1
        if len(spike_trains[1]) > 0:
            pre_trace = utils.convert_spiketrain_to_trace(spike_trains[1], t_max=self.t_sim, dt=dt, spike_width=1. / dt)
        else:
            pre_trace = np.zeros(self.t_sim / dt)

        if len(spike_trains[2]) > 0:
            post_trace = utils.convert_spiketrain_to_trace(spike_trains[2], t_max=self.t_sim, dt=dt, spike_width=1. / dt)
        else:
            post_trace = np.zeros(self.t_sim / dt)
        tau_dict = {'tau_zi' : self.bcpnn_params['tau_i'], 'tau_zj' : self.bcpnn_params['tau_j'], 
                    'tau_ei' : self.bcpnn_params['tau_e'], 'tau_ej' : self.bcpnn_params['tau_e'], 'tau_eij' : self.bcpnn_params['tau_e'],
                    'tau_pi' : self.bcpnn_params['tau_p'], 'tau_pj' : self.bcpnn_params['tau_p'], 'tau_pij' : self.bcpnn_params['tau_p'],
                    }

        fmax = self.bcpnn_params['fmax']

        # compute the traces 
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace, \
                tau_dict=tau_dict, fmax=fmax, dt=dt)#, initial_value=.1)
        t_axis = dt * np.arange(zi.size)
#        w_avg, bias_avg = Bcpnn.comput

        # save
#        np.savetxt(self.output_folder + 'w_ij.txt', np.array((t_axis, wij)).transpose() )
#        np.savetxt(self.output_folder + 'bias.txt', np.array((t_axis, bias)).transpose() )
#        np.savetxt(self.output_folder + 'pi.txt', np.array((t_axis, pi)).transpose() )
#        np.savetxt(self.output_folder + 'pj.txt', np.array((t_axis, pj)).transpose() )
#        np.savetxt(self.output_folder + 'pij.txt', np.array((t_axis, pij)).transpose() )
#        np.savetxt(self.output_folder + 'ei.txt', np.array((t_axis, ei)).transpose() )
#        np.savetxt(self.output_folder + 'ej.txt', np.array((t_axis, ej)).transpose() )
#        np.savetxt(self.output_folder + 'eij.txt', np.array((t_axis, eij)).transpose() )
#        np.savetxt(self.output_folder + 'zi.txt', np.array((t_axis, zi)).transpose() )
#        np.savetxt(self.output_folder + 'zj.txt', np.array((t_axis, zj)).transpose() )
        self.w_end = wij[-1]
        self.w_max = wij.max()
        self.w_avg = wij[-self.steps_for_wavg:].mean()
        self.t_max = wij.argmax() * dt
        print 'BCPNN offline computation:'
        print '\tw_ij : %.4e\tt_wmax: %d' % (self.w_end, self.t_max)
        print '\tw_avg : %.4e' % (self.w_avg)
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
        self.title_fontsize = 18
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j']), fontsize=self.title_fontsize)
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
        ax2.set_title('$\\tau_{p} = %d$ ms' % \
                (self.bcpnn_params['tau_p']), fontsize=self.title_fontsize)
        ax2.legend(plots, labels_p)
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c='b', lw=2)
        p2, = ax3.plot(t_axis, ej, c='g', lw=2)
        p3, = ax3.plot(t_axis, eij, c='r', lw=2)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.set_title('$\\tau_{e} = %d$ ms' % \
                (self.bcpnn_params['tau_e']), fontsize=self.title_fontsize)
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
        ax6.set_ylabel('Bias')

        ax5.set_yticks([])
        ax5.set_xticks([])
        ax5.annotate('$v_{stim} = %.2f, v_{0}=%.2f, v_{1}=%.2f$\ndx: %.2f\
                \nWeight max: %.3e\nWeight end: %.3e\nWeight avg: %.3e\nt(w_max): %.1f [ms]' % \
                (self.v_stim, self.tp_s[0][2], self.tp_s[1][2], self.dx, self.w_max, self.w_end, self.w_avg, \
                self.t_max * dt), (.1, .1), fontsize=20)

#        ax5.set_xticks([])

#        output_fn = self.params['figures_folder'] + 'traces_tauzi_%04d_tauzj%04d_taue%d_taup%d_dx%.2e_dv%.2e_vstim%.1e.png' % \
#                (self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j'], self.bcpnn_params['tau_e'], self.bcpnn_params['tau_p'], self.dx, self.dv, self.v_stim)
        output_fn = self.params['figures_folder'] + 'traces_dx%.2e_dv%.2e_vstim%.1e_tauzi_%04d_tauzj%04d_taue%d_taup%d.png' % \
                (self.dx, self.dv, self.v_stim, self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j'], self.bcpnn_params['tau_e'], self.bcpnn_params['tau_p'])
        print 'Saving traces to:', output_fn
        pylab.savefig(output_fn)


#        for gid in gids:
#            time_axis, volt = utils.extract_trace(d, gid)
#            p, = ax.plot(time_axis, volt, label='%d' % gid, lw=2)
#            plots.append(p)
#        ax.legend(plots, self.selected_gids)




if __name__ == '__main__':



#    tau_zi = float(sys.argv[1])
#    v_stim = float(sys.argv[2])
#    dx = float(sys.argv[3])
#    dv = float(sys.argv[4])
#    tau_zj = float(sys.argv[5])
#    tau_e = float(sys.argv[6])
#    tau_p = float(sys.argv[7])
#    x0 = float(sys.argv[8])
#    u0 = float(sys.argv[9])
#    output_folder = os.path.abspath(sys.argv[10]) + '/'

    v_stim = 1.0
#    tau_zi = 1. / v_stim * 100
    tau_zi = 150.
    dx = .25
    dv = .0
    tau_zj = 10.
    tau_e = 10.
    tau_p = 50000.
    x0 = .5
    u0 = v_stim
    output_folder = 'TwoCellTest/'

    bcpnn_params = {'tau_i': tau_zi, 'tau_j': tau_zj, 'tau_e': tau_e, 'tau_p': tau_p, 'fmax':300.}

#    output_fn = '%sdetailed_sweep_tauiz%d_tauzj%d_taue%d_taup%d_x0%.2f_u0%.2f_vstim%.2f.dat' % \
#            (output_folder, tau_zi, tau_zj, tau_e, tau_p, x0, u0, v_stim)
    output_fn = '%stauzizje_sweep_taup%d_x0%.2f_u0%.2f_vstim%.2f.dat' % \
            (output_folder, tau_p, x0, u0, v_stim)

    import simulation_parameters
    ps = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    ps.set_filenames(output_folder)
    params = ps.load_params()                       # params stores cell numbers, etc as a dictionary
#    params['dt_rate'] = 1.0
    ps.create_folders()

    selected_gids = [0, 1]

    print 'Selected gids:', selected_gids
    TE = ToyExperiment(params, output_folder, selected_gids, bcpnn_params)
    TE.dx = dx
    TE.dv = dv
    TE.v_stim = v_stim

#    t_sim = TE.dx / v_stim * 1000. * 20.
#    t_sim = 1000.
#    t_sim = 1. * bcpnn_params['tau_p']
    TE.run_sim(tp_0=(x0, u0))#t_sim)

    fig1 = pylab.figure()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)
    TE.plot_input(ax1)
    TE.plot_voltages(ax2)
    TE.get_bcpnn_traces_from_spiketrain()

#    output_fn = '%s/sweep_vstim_tauzj%d_taue%d.dat' % \
#            (output_folder, bcpnn_params['tau_j'], bcpnn_params['tau_e'], TE.dv)
    f = file(output_fn, 'a')
                     # 0    1     2    3      4    5      6     7    8     9     10    11
    str_to_write = '%.2e\t%.2e\t%.2e\t%.1e\t%.1e\t%.1e\t%.1e\t%.1e\t%.4e\t%.4e\t%.4e\t%.1f\n' % \
            (TE.dx, TE.dv, v_stim, bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'], v_stim, TE.w_max, TE.w_end, TE.w_avg, TE.t_max)
            # 0        1       2        3                   4                       5                       6                   7           8       9       10          11
    f.write(str_to_write)
#    pylab.show()

#    if len(sys.argv) > 1:
#        selected_gids = [int(sys.argv[i]) for i in xrange(1, 1 + len(sys.argv[1:]))]
#    else:
#        selected_gids = np.loadtxt(params['gids_to_record_fn'], dtype=int)[:2]

