import pylab
import numpy as np
import simulation_parameters
from NeuroTools import signals as nts


class PlotPrediction(object):
    """
    Creating an instance of this class plots following figures:
    1) nspikes_binned 
        x: time bins
        y: gid
        color: nspikes within time_binsize

    2) nspikes_binned_normalized
        x: time bins
        y: gid
        color: nspikes within time_binsize

    # single neuron level
    3) vx_binned
        x: time bins
        y: vx 
        color: confidence of vx (based on normalized nspikes_binned_normalized)

    4) vy_binned
        x: time bins
        y: vy 
        color: confidence of vy (based on normalized nspikes_binned_normalized)

    # population level: evaluate confidences ('votes')
    5) vx - prediction
        x: time bins
        y: vx_prediction
        contains several plots:
        - linear_avg
        . trace (based on vx_binned_trace), with yerr=vx_binned_trace.std()
        x non-linear vote

    6) same as 5) for vy

    7) theta-prediction
        x: time
        y: arctan2(vy_prediction, vx_prediction)
        contains several plots, based on the different predictions
        - linear_avg
        . trace (based on vx_binned_trace), with yerr=vx_binned_trace.std()
        x non-linear vote

    # large time-scale, population level
    bar plots
    9) vx prediction based on whole population and full run
        vx_full_run_linear_avg
        vx_full_run_non_linear
        x: vx
        y: confidence 

    10) same as 9) for vy
    11) same as 9) for theta
    """

    def __init__(self, sim_cnt=0):

        self.network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
        self.params = self.network_params.load_params()                       # params stores cell numbers, etc as a dictionary

        # define parameters
        self.n_cells = self.params['n_exc']
        self.time_binsize = 50 # [ms]
        self.n_bins = int((self.params['t_sim'] / self.time_binsize) )
        self.time_bins = [self.time_binsize * i for i in xrange(self.n_bins)]
        self.t_axis = np.arange(0, self.n_bins * self.time_binsize, self.time_binsize)

        # create data structures
        self.nspikes = np.zeros(self.n_cells)                                   # summed activity
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        self.nspikes_binned_normalized = np.zeros((self.n_cells, self.n_bins))  # normalized so that for each bin, the sum of the population activity = 1
        self.nspikes_normalized = np.zeros(self.n_cells)                        # activity normalized, so that sum = 1

        # sort the cells by their tuning vx, vy properties
        tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        # vx
        self.vx_tuning = tuning_prop[:, 2].copy()
        self.vx_tuning.sort()
        self.sorted_indices_vx = tuning_prop[:, 2].argsort()
        # vy
        self.vy_tuning = tuning_prop[:, 3].copy()
        self.vy_tuning.sort()
        self.sorted_indices_vy = tuning_prop[:, 3].argsort()

        self.load_spiketimes(sim_cnt)


    def load_spiketimes(self, sim_cnt):
        """
        Fills the 4 arrays with data:
        self.nspikes = np.zeros(self.n_cells)                                   # summed activity
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        self.nspikes_binned_normalized = np.zeros((self.n_cells, self.n_bins))  # normalized so that for each bin, the sum of the population activity = 1
        self.nspikes_normalized = np.zeros(self.n_cells)                        # activity normalized, so that sum = 1
        """

        print(' Loading data .... ')
        folder = self.params['spiketimes_folder']
#        fn = self.params['exc_spiketimes_fn_merged'].rsplit(folder)[1] + '%d.dat' % (sim_cnt)
        fn = self.params['exc_spiketimes_fn_merged'] + '%d.ras' % (sim_cnt)
        
        # NeuroTools
#        spklist = nts.load_spikelist(fn)
#        spiketrains = spklist.spiketrains
        d = np.loadtxt(fn)
        spiketrains = [[] for i in xrange(self.n_cells)]
        for i in xrange(d[:, 0].size):
            spiketrains[int(d[i, 1])].append(d[i, 0])

        for gid in xrange(self.params['n_exc']):

#            spiketimes = spiketrains[gid+1.].spike_times
#            nspikes = spiketimes.size

            spiketimes = spiketrains[gid]
            nspikes = len(spiketimes)
            if (nspikes > 0):
                count, bins = np.histogram(spiketimes, bins=self.n_bins)
                self.nspikes_binned[gid, :] = count
            self.nspikes[gid] = nspikes

        # normalization
        for i in xrange(int(self.n_bins)):
            if (self.nspikes_binned[:, i].sum() > 0):
                self.nspikes_binned_normalized[:, i] = self.nspikes_binned[:, i] / self.nspikes_binned[:,i].sum()
        self.nspikes_normalized = self.nspikes / self.nspikes.sum()

    def compute_v_estimates(self):
        """
        This function combines activity on the population level to estimate vx, vy

         On which time scale shall the prediction work?
         There are (at least) 3 different ways to do it:
           Very short time-scale:
           1) Compute the prediction for each time bin - based on the activitiy in the respective time bin 
           Short time-scale:
           2) Compute the prediction for each time bin based on all activity in the past
           3) Non-linear 'voting' based on 1)
           Long time-scale:
           3) Compute the prediction based on the the activity of the whole run - not time dependent
           4) Non-linear 'voting' based on 3) 
        """
        # momentary result, based on the activity in one time bin
        self.vx_avg = np.zeros(self.n_bins) 
        self.vy_avg = np.zeros(self.n_bins)
        # ---> gives theta_avg 

        # based on the activity in several time bins
        self.vx_moving_avg = np.zeros((self.n_bins, 2))
        self.vy_moving_avg = np.zeros((self.n_bins, 2))

        # non linear transformation of vx_avg
        self.vx_non_linear = np.zeros(self.n_bins)
        self.vy_non_linear = np.zeros(self.n_bins)

        trace_length = 100 # [ms] window length for moving average 
        trace_length_in_bins = int(round(trace_length / self.time_binsize))
        # ---> gives theta_moving_avg

        # # # # # # # # # # # # # # # # # # # # # # 
        # S P E E D    P R E D I C T I O N 
        # # # # # # # # # # # # # # # # # # # # # # 
        self.vx_confidence_binned = self.nspikes_binned_normalized[self.sorted_indices_vx]
        self.vy_confidence_binned = self.nspikes_binned_normalized[self.sorted_indices_vy]
        vx_prediction_trace = np.zeros((self.n_cells, self.n_bins, 2))    # _trace: prediction based on the momentary and past activity (moving average, and std) --> trace_length
        vy_prediction_trace = np.zeros((self.n_cells, self.n_bins, 2))    # _trace: prediction based on the momentary and past activity (moving average, and std) --> trace_length
        for i in xrange(self.n_bins):

            # 1) momentary vote
            # take the weighted average for v_prediction (weight = normalized activity)
            vx_pred = self.vx_confidence_binned[:, i] * self.vx_tuning
            vy_pred = self.vy_confidence_binned[:, i] * self.vy_tuning
            self.vx_avg[i] = np.sum(vx_pred)
            self.vy_avg[i] = np.sum(vy_pred)

            # 2) moving average
            past_bin = max(0, min(0, i-trace_length_in_bins))
            for cell in xrange(self.n_cells):
                vx_prediction_trace[cell, i, 0] = self.vx_confidence_binned[cell, past_bin:i].mean()
                vx_prediction_trace[cell, i, 1] = self.vx_confidence_binned[cell, past_bin:i].std()
                vy_prediction_trace[cell, i, 0] = self.vy_confidence_binned[cell, past_bin:i].mean()
                vy_prediction_trace[cell, i, 1] = self.vy_confidence_binned[cell, past_bin:i].std()
            self.vx_moving_avg[i, 0] = np.sum(vx_prediction_trace[:, i, 0] * self.vx_tuning)
            self.vx_moving_avg[i, 1] = np.sum(vx_prediction_trace[:, i, 1] * self.vx_tuning)

            # 3)
            # rescale activity to negative values
            vx_shifted = self.nspikes_binned[self.sorted_indices_vx, i] - self.nspikes_binned[self.sorted_indices_vx, i].max()
            vy_shifted = self.nspikes_binned[self.sorted_indices_vy, i] - self.nspikes_binned[self.sorted_indices_vy, i].max()
            # exp --> mapping to range(0, 1)
            vx_exp = np.exp(vx_shifted)
            vy_exp = np.exp(vy_shifted)
            # normalize and vote
            vx_votes = (vx_exp / vx_exp.sum()) * self.vx_tuning
            vy_votes = (vy_exp / vy_exp.sum()) * self.vy_tuning
            self.vx_non_linear[i] = vx_votes.sum()
            self.vy_non_linear[i] = vy_votes.sum()

        # in the first step the trace can not have a standard deviation --> avoid NANs 
        self.vx_moving_avg[0, 0] = np.sum(self.vx_confidence_binned[self.sorted_indices_vx, 0].mean() * self.vx_tuning)
        self.vy_moving_avg[0, 0] = np.sum(self.vy_confidence_binned[self.sorted_indices_vy, 0].mean() * self.vy_tuning)
        self.vx_moving_avg[0, 1] = 0
        self.vy_moving_avg[0, 1] = 0

        # ---> time INdependent estimates: based on activity of the full run
#        self.vx_avg_fullrun = np.zeros(n_cells)             # time independent prediction based on the whole run --> voting histogram
#        self.vx_nonlinear_fullrun = np.zeros(n_cells)       # prediction based on non-linear transformation of output rates
#        self.vy_avg_fullrun = np.zeros(n_cells)             # same for v_y
#        self.vy_nonlinear_fullrun = np.zeros(n_cells)         
#        self.theta_avg_fullrun = np.zeros(n_cells)            
#        self.theta_nonlinear_fullrun = np.zeros(n_cells)      

        self.vx_avg_fullrun = self.nspikes_normalized[self.sorted_indices_vx]# * self.vx_tuning
        self.vy_avg_fullrun = self.nspikes_normalized[self.sorted_indices_vy]# * self.vy_tuning

        # vx nonlinear
        nspikes_shifted = self.nspikes - self.nspikes.max()
        nspikes_exp = np.exp(nspikes_shifted)
        self.vx_nonlinear_fullrun = nspikes_exp / nspikes_exp.sum()
        # vx nonlinear
        self.vy_avg_fullrun = self.nspikes_normalized[self.sorted_indices_vy]# * self.vy_tuning
        nspikes_shifted = self.nspikes - self.nspikes.max()
        nspikes_exp = np.exp(nspikes_shifted)
        self.vy_nonlinear_fullrun = nspikes_exp / nspikes_exp.sum()




    def compute_theta_estimates(self):

        self.theta_avg = np.arctan2(self.vy_avg, self.vx_avg)
        self.theta_moving_avg = np.zeros((self.n_bins, 2))
        self.theta_moving_avg[:, 0] = np.arctan2(self.vy_moving_avg[:, 0], self.vx_moving_avg[:, 0])
        self.theta_moving_avg[:, 1] = self.theta_uncertainty(self.vx_moving_avg[:, 0], self.vx_moving_avg[:, 1], self.vy_moving_avg[:, 0], self.vy_moving_avg[:, 1])
        self.theta_non_linear = np.arctan2(self.vy_non_linear, self.vx_non_linear)
        # theta: take the average confidence for vote
        self.theta = np.arctan2(self.vy_tuning, self.vx_tuning)
        self.theta_avg_fullrun = .5 * (self.vx_avg_fullrun + self.vy_avg_fullrun)
        self.theta_nonlinear_fullrun = .5 * (self.vx_nonlinear_fullrun + self.vy_nonlinear_fullrun)



    def plot(self):
        print "plotting ...."
        self.fig1 = pylab.figure()
        pylab.subplots_adjust(hspace=0.95)
        pylab.subplots_adjust(wspace=0.3)

    def plot_nspikes_binned(self):

        self.ax1 = self.fig1.add_subplot(421)
        self.ax1.set_title('Spiking activity over time')
        self.cax1 = self.ax1.pcolor(self.nspikes_binned)
        self.ax1.set_ylim((0, self.nspikes_binned[:, 0].size))
        self.ax1.set_xlim((0, self.nspikes_binned[0, :].size))
        self.ax1.set_xlabel('Time [ms]')
        self.ax1.set_ylabel('GID')
        self.ax1.set_xticks(range(self.n_bins)[::2])
        self.ax1.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax1)

    def plot_nspikes_binned_normalized(self):

        self.ax2 = self.fig1.add_subplot(422)
        self.ax2.set_title('Normalized activity over time')
        self.cax2 = self.ax2.pcolor(self.nspikes_binned_normalized)
        self.ax2.set_ylim((0, self.nspikes_binned_normalized[:, 0].size))
        self.ax2.set_xlim((0, self.nspikes_binned_normalized[0, :].size))
        self.ax2.set_xlabel('Time [ms]')
        self.ax2.set_ylabel('GID')
        self.ax2.set_xticks(range(self.n_bins)[::2])
        self.ax2.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax2)


    def plot_vx_confidence_binned(self):
        self.ax3 = self.fig1.add_subplot(423)
        self.ax3.set_title('Vx confidence over time')
        self.cax3 = self.ax3.pcolor(self.vx_confidence_binned)
        self.ax3.set_ylim((0, self.vx_confidence_binned[:, 0].size))
        self.ax3.set_xlim((0, self.vx_confidence_binned[0, :].size))
        self.ax3.set_xlabel('Time [ms]')
        self.ax3.set_ylabel('$v_x$')
        self.ax3.set_xticks(range(self.n_bins)[::2])
        self.ax3.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        ny = self.vx_tuning.size
        n_ticks = 4
        yticks = [self.vx_tuning[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        ylabels = ['%.1e' % i for i in yticks]
        self.ax3.set_yticks([int(i * ny/n_ticks) for i in xrange(n_ticks)])
        self.ax3.set_yticklabels(ylabels)
        self.ax3.set_xticks(range(self.n_bins)[::2])
        self.ax3.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax3)



    def plot_vy_confidence_binned(self):
        self.ax4 = self.fig1.add_subplot(424)
        self.ax4.set_title('vy confidence over time')
        self.cax4 = self.ax4.pcolor(self.vy_confidence_binned)
        self.ax4.set_ylim((0, self.vy_confidence_binned[:, 0].size))
        self.ax4.set_xlim((0, self.vy_confidence_binned[0, :].size))
        self.ax4.set_xlabel('Time [ms]')
        self.ax4.set_ylabel('$v_y$')
        self.ax4.set_xticks(range(self.n_bins)[::2])
        self.ax4.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        ny = self.vy_tuning.size
        n_ticks = 4
        yticks = [self.vy_tuning[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        ylabels = ['%.1e' % i for i in yticks]
        self.ax4.set_yticks([int(i * ny/n_ticks) for i in xrange(n_ticks)])
        self.ax4.set_yticklabels(ylabels)
        self.ax4.set_xticks(range(self.n_bins)[::2])
        self.ax4.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax4)


    def plot_vx_estimates(self):
        self.ax5 = self.fig1.add_subplot(425)
        self.ax5.set_title('$v_{x}$-predictions: avg, moving_avg, nonlinear')
        self.ax5.plot(self.t_axis, self.vx_avg, ls='-')
        self.ax5.errorbar(self.t_axis, self.vx_moving_avg[:, 0], yerr=self.vx_moving_avg[:, 1], ls='--')
        self.ax5.plot(self.t_axis, self.vx_non_linear, ls=':')
        self.ax5.set_xlabel('Time [ms]')
        self.ax5.set_ylabel('$v_x$')
        ny = self.t_axis.size
        n_ticks = 5
        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in t_ticks]
        self.ax5.set_xticks(t_ticks)
        self.ax5.set_xticklabels(t_labels)

    def plot_vy_estimates(self):
        self.ax6 = self.fig1.add_subplot(426)
        self.ax6.set_title('$v_{y}$-predictions: avg, moving_avg, nonlinear')
        self.ax6.plot(self.t_axis, self.vy_avg, ls='-')
        self.ax6.errorbar(self.t_axis, self.vy_moving_avg[:, 0], yerr=self.vy_moving_avg[:, 1], ls='--')
        self.ax6.plot(self.t_axis, self.vy_non_linear, ls=':')
        self.ax6.set_xlabel('Time [ms]')
        self.ax6.set_ylabel('$v_y$')
        ny = self.t_axis.size
        n_ticks = 5
        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in t_ticks]
        self.ax6.set_xticks(t_ticks)
        self.ax6.set_xticklabels(t_labels)

    def plot_theta_estimates(self):
        self.ax7 = self.fig1.add_subplot(427)
        self.ax7.set_title('$\Theta$-predictions: avg, moving_avg, nonlinear')
        self.ax7.plot(self.t_axis, self.theta_avg, ls='-')
        self.ax7.errorbar(self.t_axis, self.theta_moving_avg[:, 0], yerr=self.theta_moving_avg[:, 1], ls='--')
        self.ax7.plot(self.t_axis, self.theta_non_linear, ls=':')
        self.ax7.set_xlabel('Time [ms]')
        self.ax7.set_ylabel('$\Theta$')
        ny = self.t_axis.size
        n_ticks = 5
        t_ticks = [self.t_axis[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
        t_labels= ['%d' % i for i in t_ticks]
        self.ax7.set_xticks(t_ticks)
        self.ax7.set_xticklabels(t_labels)

    def plot_fullrun_estimates(self):
        self.fig2 = pylab.figure()
        pylab.subplots_adjust(hspace=0.5)
        self.plot_fullrun_estimates_vx()
        self.plot_fullrun_estimates_vy()
        self.plot_fullrun_estimates_theta()



    def plot_fullrun_estimates_vx(self):
        self.ax8 = self.fig2.add_subplot(411)
        bin_width = np.mean(self.vx_tuning[:-1] - self.vx_tuning[1:])
        self.ax8.bar(self.vx_tuning, self.vx_avg_fullrun, width=bin_width)
#        self.ax8.bar(self.vx_tuning - bin_width, self.vx_nonlinear_fullrun, width=bin_width, facecolor='g')

#        n_bins = 50
#        count, vx_bins = np.histogram(self.vx_tuning, n_bins)
#        pred_avg, x = np.histogram(self.vx_avg_fullrun, n_bins)
#        pred_nonlinear, x = np.histogram(self.vx_nonlinear_fullrun, n_bins)
#        bin_width = vx_bins[1]-vx_bins[0]
#        self.ax8.bar(vx_bins[:-1], pred_avg, width=bin_width*.5)
#        self.ax8.bar(vx_bins[:-1]-.5*bin_width, pred_nonlinear, width=bin_width*.5, facecolor='g')
#        self.ax8.set_xlim((self.vx_tuning.min() - bin_width, self.vx_tuning.max()))
        self.ax8.set_title('Estimates based on full run activity')
        self.ax8.set_xlabel('$v_x$')
        self.ax8.set_ylabel('Confidence')

    def plot_fullrun_estimates_vy(self):
        self.ax9 = self.fig2.add_subplot(412)
        bin_width = np.mean(self.vy_tuning[:-1] - self.vy_tuning[1:])
        self.ax9.bar(self.vy_tuning, self.vy_avg_fullrun, width=bin_width)
#        self.ax9.bar(self.vy_tuning - bin_width, self.vy_nonlinear_fullrun, width=bin_width, facecolor='g')

#        n_bins = 50
#        count, vy_bins = np.histogram(self.vy_tuning, n_bins)
#        pred_avg, x = np.histogram(self.vy_avg_fullrun, n_bins)
#        pred_nonlinear, x = np.histogram(self.vy_nonlinear_fullrun, n_bins)
#        bin_width = vy_bins[1]-vy_bins[0]
#        self.ax9.bar(vy_bins[:-1], pred_avg, width=bin_width*.5)
#        self.ax9.bar(vy_bins[:-1]-.5*bin_width, pred_nonlinear, width=bin_width*.5, facecolor='g')
#        self.ax9.set_xlim((self.vy_tuning.min() - bin_width, self.vy_tuning.max()))
        self.ax9.set_xlabel('$v_y$')
        self.ax9.set_ylabel('Confidence')

    def plot_fullrun_estimates_theta(self):

        self.ax10 = self.fig2.add_subplot(413)
        bin_width = np.mean(self.theta[:-1] - self.theta[1:])
        self.ax10.bar(self.theta, self.theta_avg_fullrun, width=bin_width)
        self.ax10.bar(self.theta - bin_width, self.theta_nonlinear_fullrun, width=bin_width, facecolor='g')


#        n_bins = 50
#        count, theta_bins = np.histogram(self.theta_tuning, n_bins)
#        pred_avg, x = np.histogram(self.theta_avg_fullrun, n_bins)
#        pred_nonlinear, x = np.histogram(self.theta_nonlinear_fullrun, n_bins)
#        bin_width = theta_bins[1]-theta_bins[0]
#        self.ax10.bar(theta_bins[:-1], pred_avg, width=bin_width*.5)
#        self.ax10.bar(theta_bins[:-1]-.5*bin_width, pred_nonlinear, width=bin_width*.5, facecolor='g')
#        self.ax10.set_xlim((self.theta_tuning.min() - bin_width, self.theta_tuning.max()))
        self.ax10.set_xlabel('$\Theta$')
        self.ax10.set_ylabel('Confidence')

    def plot_nspike_histogram(self):
        self.ax10 = self.fig2.add_subplot(414)
        self.ax10.bar(range(self.params['n_exc']), self.nspikes)
        self.ax10.set_xlabel('Cell gids')
        self.ax10.set_ylabel('Number of spikes')

    def theta_uncertainty(self, vx, dvx, vy, dvy):
        """
        theta = arctan(vy / vx)
        Please check with http://en.wikipedia.org/wiki/Propagation_of_uncertainty
        """
        return vx / (vx**2 + vy**2) * dvy - vy / (vx**2 + vx**2) * dvx
