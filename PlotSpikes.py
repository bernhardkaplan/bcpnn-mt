import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import simulation_parameters
import utils
import os
from NeuroTools import parameters as ntp

class PlotSpikeActivity(object):
    """
    Class offering several plotting functions.
    The core data file is a file containing cell_gids and spike times.
    """

    def __init__(self, param_fn=None, spiketimes_fn=None):
        """
        params : dictionary or NeuroTools.parameters ParameterSet
        """

        print "debug", type(param_fn)
        if param_fn == None:
            print "Loading default parameters stored in simulation_parameters.py"
            self.network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
            self.params = self.network_params.load_params()                       # params stores cell numbers, etc as a dictionary
            self.params_fn = self.params['params_fn']
        else:
            if type(param_fn) != type(""): raise TypeError("File name expected for param_fn")
            self.params_fn = param_fn
            self.params = ntp.ParameterSet(param_fn)
            self.params
    
        self.spiketimes_fn = spiketimes_fn
        print os.path.abspath(self.params_fn)
        print os.path.abspath(self.spiketimes_fn)

        fig_width_pt = 800.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        params = {#'backend': 'png',
#                  'axes.labelsize': 10,
#                  'text.fontsize': 10,
#                  'legend.fontsize': 10,
#                  'xtick.labelsize': 8,
#                  'ytick.labelsize': 8,
#                  'text.usetex': True,
                  'figure.figsize': fig_size}
        pylab.rcParams.update(params)


    def plot_rasterplot(self, fn=None):
        if fn == None:
            fn = self.spiketimes_fn
        self.load_spiketimes(fn)


    def load_spiketimes(self, fn):
        """
        Fills the following arrays with data:
        self.nspikes = np.zeros(self.n_cells)                                   # summed activity
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        self.nspikes_binned_normalized = np.zeros((self.n_cells, self.n_bins))  # normalized so that for each bin, the sum of the population activity = 1
        self.nspikes_normalized = np.zeros(self.n_cells)                        # activity normalized, so that sum = 1
        self.nspikes_normalized_nonlinear
        """

        print(' Loading data .... ')
#        folder = self.params['spiketimes_folder']
#        fn = self.params['exc_spiketimes_fn_merged'].rsplit(folder)[1] + '%d.dat' % (sim_cnt)
        try:
            d = np.loadtxt(fn)
            self.spiketrains = [[] for i in xrange(self.n_cells)]
            for i in xrange(d[:, 0].size):
                self.spiketrains[int(d[i, 1])].append(d[i, 0])
        except:
            print 'WARNING: no spikes found in:', fn
            self.no_spikes = True
            return

        for gid in xrange(self.params['n_exc']):
#            spiketimes = spiketrains[gid+1.].spike_times
#            nspikes = spiketimes.size

            spiketimes = self.spiketrains[gid]
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

        # activity normalized, nonlinear
        nspikes_shifted = self.nspikes - self.nspikes.max()
        nspikes_exp = np.exp(nspikes_shifted)
        self.nspikes_normalized_nonlinear = nspikes_exp / nspikes_exp.sum()



    def load_parameters(self):
        """
        Loads parameters from the parameter file: n_cells, simulation time, etc
        """

        # define parameters
        self.n_cells = self.params['n_exc']

        self.time_binsize = 20 # [ms]
        self.n_bins = int((self.params['t_sim'] / self.time_binsize) )
        self.time_bins = [self.time_binsize * i for i in xrange(self.n_bins)]
        self.t_offset = 0
        self.t_axis = np.arange(self.t_offset, self.n_bins * self.time_binsize + self.t_offset, self.time_binsize)
        self.n_vx_bins, self.n_vy_bins = 20, 20                                 # needed for clustering activity colormaps according to a parameter

        # create data structures
        self.nspikes = np.zeros(self.n_cells)                                   # activity summed over whole run
        self.nspikes_normalized = np.zeros(self.n_cells)                        # activity normalized, so that sum = 1
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        self.nspikes_binned_normalized = np.zeros((self.n_cells, self.n_bins))  # normalized so that for each bin, the sum of the population activity = 1
    

    def load_cell_properties(self):
        # sort the cells by their tuning vx, vy properties
        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])

        # vx
        self.vx_tuning = self.tuning_prop[:, 2].copy()
        self.vx_tuning.sort()
        self.sorted_indices_vx = self.tuning_prop[:, 2].argsort()
        # vy
        self.vy_tuning = self.tuning_prop[:, 3].copy()
        self.vy_tuning.sort()
        self.sorted_indices_vy = self.tuning_prop[:, 3].argsort()

        if self.no_spikes:
            return


    def create_fig(self):
        print "plotting ...."
        self.fig = pylab.figure()
        pylab.subplots_adjust(hspace=0.95)
        pylab.subplots_adjust(wspace=0.3)

    def plot_colormap(self, d, title=''):

        ax1 = pylab.subplot(111)
        ax1.set_title('Spiking activity over time')
        cax1 = self.ax1.pcolor(self.nspikes_binned)
        ax1.set_ylim((0, self.nspikes_binned[:, 0].size))
        ax1.set_xlim((0, self.nspikes_binned[0, :].size))
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('GID')
        ax1.set_xticks(range(self.n_bins)[::2])
        ax1.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax1)
        return ax


    def plot_nspikes_binned(self):
        pass


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
        pylab.rcParams['legend.fontsize'] = 10
        pylab.subplots_adjust(hspace=0.5)
        self.plot_fullrun_estimates_vx()
        self.plot_fullrun_estimates_vy()
        self.plot_fullrun_estimates_theta()


    def plot_fullrun_estimates_vx(self):
        self.ax8 = self.fig2.add_subplot(411)
        bin_width = .5 * (self.vx_grid[1] - self.vx_grid[0])
        vx_linear = (np.sum(self.vx_grid * self.vx_marginalized_binned), self.get_uncertainty(self.vx_marginalized_binned, self.vx_grid))
        vx_nonlinear = (np.sum(self.vx_grid * self.vx_marginalized_binned_nonlinear), self.get_uncertainty(self.vx_marginalized_binned_nonlinear, self.vx_grid))
        self.ax8.bar(self.vx_grid, self.vx_marginalized_binned, width=bin_width, label='Linear votes: $v_x=%.2f \pm %.2f$' % (vx_linear[0], vx_linear[1]))
        self.ax8.bar(self.vx_grid+bin_width, self.vx_marginalized_binned_nonlinear, width=bin_width, facecolor='g', label='Non-linear votes: $v_x=%.2f \pm %.2f$' % (vx_nonlinear[0], vx_nonlinear[1]))
        self.ax8.set_title('Estimates based on full run activity with %s connectivity\nblue: linear marginalization over all positions, green: non-linear voting' % self.params['initial_connectivity'])
        self.ax8.set_xlabel('$v_x$')
        self.ax8.set_ylabel('Confidence')
        self.ax8.legend()


    def get_uncertainty(self, p, v):
        """
        p, v are vectors storing the confidence of the voters in p, and the values they vote for in v.
        The uncertainty is estimated as:
        sum_i p_i * (1. - p_i) * v_i
        Idea behind it:
        (1. - p_i) * v_i gives the uncertainty for each vote of v_i
        multiplying it with p_i takes into account how much weight this uncertainty should have in the overall vote
        """
        uncertainties = (np.ones(len(p)) - p) * v
        weighted_uncertainties = p * uncertainties
        return np.sum(weighted_uncertainties)

    def plot_fullrun_estimates_vy(self):
        self.ax9 = self.fig2.add_subplot(412)
        bin_width = .5 * (self.vy_grid[1] - self.vy_grid[0])
        vy_linear = (np.sum(self.vy_grid * self.vy_marginalized_binned), self.get_uncertainty(self.vy_marginalized_binned, self.vy_grid))
        vy_nonlinear = (np.sum(self.vy_grid * self.vy_marginalized_binned_nonlinear), self.get_uncertainty(self.vy_marginalized_binned_nonlinear, self.vy_grid))
        self.ax9.bar(self.vy_grid, self.vy_marginalized_binned, width=bin_width, label='Linear votes: $v_y=%.2f \pm %.2f$' % (vy_linear[0], vy_linear[1]))
        self.ax9.bar(self.vy_grid+bin_width, self.vy_marginalized_binned_nonlinear, width=bin_width, facecolor='g', label='Non-linear votes: $v_y=%.2f \pm %.2f$' % (vy_nonlinear[0], vy_nonlinear[1]))
        self.ax9.set_xlabel('$v_y$')
        self.ax9.set_ylabel('Confidence')
        self.ax9.legend()

    def plot_fullrun_estimates_theta(self):

        self.ax10 = self.fig2.add_subplot(413)
        bin_width = .5 * (self.theta_grid[-1] - self.theta_grid[-2])
        theta_linear = (np.sum(self.theta_grid * self.theta_marginalized_binned), self.get_uncertainty(self.theta_marginalized_binned, self.theta_grid))
        theta_nonlinear = (np.sum(self.theta_grid * self.theta_marginalized_binned_nonlinear), self.get_uncertainty(self.theta_marginalized_binned_nonlinear, self.theta_grid))
        self.ax10.bar(self.theta_grid, self.theta_marginalized_binned, width=bin_width, label='Linear votes: $\Theta=%.2f \pm %.2f$' % (theta_linear[0], theta_linear[1]))
        self.ax10.bar(self.theta_grid+bin_width, self.theta_marginalized_binned_nonlinear, width=bin_width, facecolor='g', label='Non-linear votes: $\Theta=%.2f \pm %.2f$' % (theta_nonlinear[0], theta_nonlinear[1]))
        self.ax10.bar(self.theta_grid, self.theta_marginalized_binned, width=bin_width)
        self.ax10.bar(self.theta_grid+bin_width, self.theta_marginalized_binned_nonlinear, width=bin_width, facecolor='g')
        self.ax10.set_xlim((-np.pi, np.pi))
        self.ax10.legend()


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
        mean_nspikes = self.nspikes.mean()* 1000./self.params['t_sim'] 
        std_nspikes = self.nspikes.std() * 1000./self.params['t_sim']
        self.ax10.bar(range(self.params['n_exc']), self.nspikes* 1000./self.params['t_sim'], label='$f_{mean} = (%.1f \pm %.1f)$ Hz' % (mean_nspikes, std_nspikes))
        self.ax10.set_xlabel('Cell gids')
        self.ax10.set_ylabel('Output rate $f_{out}$')
        self.ax10.legend()

    def theta_uncertainty(self, vx, dvx, vy, dvy):
        """
        theta = arctan(vy / vx)
        Please check with http://en.wikipedia.org/wiki/Propagation_of_uncertainty
        """
        return vx / (vx**2 + vy**2) * dvy - vy / (vx**2 + vx**2) * dvx
