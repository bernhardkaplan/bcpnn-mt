#import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np
import simulation_parameters
import utils

class PlotConductances(object):
    def __init__(self, params=None, data_fn=None, sim_cnt=0):

        if params == None:
            self.network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
            self.params = self.network_params.load_params()                       # params stores cell numbers, etc as a dictionary
        else:
            self.params = params
        self.no_spikes = False

        self.n_fig_x = 2
        self.n_fig_y = 2
        self.tuning_params = np.loadtxt(self.params['tuning_prop_means_fn'])

        # define parameters
        self.n_cells = self.params['n_exc']
        self.time_binsize = 20 # [ms]
        self.n_bins = int((self.params['t_sim'] / self.time_binsize) )
        self.time_bins = [self.time_binsize * i for i in xrange(self.n_bins)]
        self.t_axis = np.arange(0, self.n_bins * self.time_binsize, self.time_binsize)

        self.n_good = self.params['n_exc'] * .10 # fraction of 'interesting' cells
        print 'Number of cells with \'good\' tuning_properties = ', self.n_good

        # create data structures
        self.nspikes = np.zeros(self.n_cells)                                   # summed activity
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        self.spiketrains = [[] for i in xrange(self.n_cells)]

        self.tuning_prop = np.loadtxt(self.params['tuning_prop_means_fn'])
        # sort the cells by their proximity to the stimulus into 'good_gids' and the 'rest'
        # cell in 'good_gids' should have the highest response to the stimulus
        all_gids, all_distances = utils.sort_gids_by_distance_to_stimulus(self.tuning_prop, self.params['motion_params']) 
        self.good_gids, self.good_distances = all_gids[0:self.n_good], all_distances[0:self.n_good]
        print 'Saving gids to record to', self.params['gids_to_record_fn']
        np.savetxt(self.params['gids_to_record_fn'], np.array(self.good_gids), fmt='%d')
        self.rest_gids = range(self.n_cells)
        for gid in self.good_gids:
            self.rest_gids.remove(gid)

        self.load_spiketimes(data_fn)
        if self.no_spikes:
            return
        fig_width_pt = 800.0  # Get this from LaTeX using \showthe\columnwidth
        inches_per_pt = 1.0/72.27               # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]
        params = {#'backend': 'png',
#                  'axes.labelsize': 10,
#                  'text.fontsize': 10,
                  'legend.fontsize': 10,
#                  'xtick.labelsize': 8,
#                  'ytick.labelsize': 8,
#                  'text.usetex': True,
                  'figure.figsize': fig_size}
        pylab.rcParams.update(params)


    def load_spiketimes(self, fn=None):
        """
        Fills the following arrays with data:
        self.nspikes = np.zeros(self.n_cells)                                   # summed activity
        self.nspikes_binned = np.zeros((self.n_cells, self.n_bins))             # binned activity over time
        """

        print(' Loading data .... ')
        try:
            d = np.loadtxt(fn)
        except:
            self.no_spikes = True
            return

        for i in xrange(d[:, 0].size):
            self.spiketrains[int(d[i, 1])].append(d[i, 0])

        for gid in xrange(self.params['n_exc']):
            nspikes = len(self.spiketrains[gid])
            if (nspikes > 0):
                count, bins = np.histogram(self.spiketrains[gid], bins=self.n_bins, range=(0, self.params['t_sim']))
                self.nspikes_binned[gid, :] = count
            self.nspikes[gid] = nspikes

    def plot_good_cell_connections(self, fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        for gid in self.good_gids:
            x, y, u, v = self.tuning_prop[gid]
            thetas = np.arctan2(v, u)
            h = ((thetas + np.pi) / (2 * np.pi)) * 360. # theta determines h, h must be [0, 360)
            l = np.sqrt(u**2 + v**2) / np.sqrt(2 * self.params['v_max']**2) # lightness [0, 1]
            s = 1. # saturation
            assert (0 <= h and h < 360)
            assert (0 <= l and l <= 1)
            assert (0 <= s and s <= 1)
            (r, g, b) = utils.convert_hsl_to_rgb(h, s, l)
            ax.plot(x, y, 'o', c=(r,g,b), markersize=7, markeredgewidth=0)#, edgecolors=None)
        ax.set_xlabel('X-position')
        ax.set_ylabel('Y-position')



    def plot_group_spikes_vs_time(self, fig_cnt=1):
        good_nspikes = np.zeros(self.n_bins)
        rest_nspikes = np.zeros(self.n_bins)
        for gid in self.good_gids:
            good_nspikes += self.nspikes_binned[gid, :]
        for gid in self.rest_gids:
            rest_nspikes += self.nspikes_binned[gid, :]

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.plot(self.t_axis, good_nspikes, label='good')
        ax.plot(self.t_axis, rest_nspikes, label='rest')
            
        ax.set_title('Activity sorted by groups')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Number of spikes fired by group')
        ax.legend()


    def create_fig(self):
        print "plotting ...."
        self.fig = pylab.figure()
        pylab.subplots_adjust(hspace=0.6)
        pylab.subplots_adjust(wspace=0.3)


    def plot_rasterplot(self, cell_type, fig_cnt=1):
        if cell_type == 'inh':
            fn = self.params['inh_spiketimes_fn_base'] + '0.ras'
            n_cells = self.params['n_inh']
        elif cell_type == 'exc':
            fn = self.params['exc_spiketimes_fn_merged'] + '0.ras'
            n_cells = self.params['n_exc']
        try:
            nspikes, spiketimes = utils.get_nspikes(fn, n_cells, get_spiketrains=True)
        except:
            spiketimes = []

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        for cell in xrange(n_cells):
            ax.plot(spiketimes[cell], cell * np.ones(nspikes[cell]), 'o', color='k', markersize=1)
            
        ax.set_xlim(0, self.params['t_sim'])
        ax.set_title('Rasterplot of %s neurons' % cell_type)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Neuron GID')


    def plot_input_cond(self, fig_cnt=1):
        """
        Plots the conductance from the input stimulus to the excitatory cells
        """
        binned_spikes_in_good = np.zeros((len(self.good_gids), self.n_bins))
        for i, gid in enumerate(self.good_gids):
            fn = self.params['input_st_fn_base'] + '%d.npy' % gid
            spiketrain = np.load(fn)
            binned_spiketrain, bins = np.histogram(spiketrain, bins=self.n_bins, range=(0, self.params['t_sim']))
            binned_spikes_in_good[i, :] = binned_spiketrain
        spikes_in_good = np.zeros((self.n_bins, 3))
        for t in xrange(self.n_bins):
            spikes_in_good[t, 0] = binned_spikes_in_good[:, t].mean()
            spikes_in_good[t, 1] = binned_spikes_in_good[:, t].std()
            spikes_in_good[t, 2] = binned_spikes_in_good[:, t].sum()

        binned_spikes_in_rest = np.zeros((len(self.rest_gids), self.n_bins))
        for i, gid in enumerate(self.rest_gids):
            fn = self.params['input_st_fn_base'] + '%d.npy' % gid
            spiketrain = np.load(fn)
            binned_spiketrain, bins = np.histogram(spiketrain, bins = self.n_bins, range=(0, self.params['t_sim']))
            binned_spikes_in_rest[i, :] = binned_spiketrain
        spikes_in_rest = np.zeros((self.n_bins, 3))
        for t in xrange(self.n_bins):
            spikes_in_rest[t, 0] = binned_spikes_in_rest[:, t].mean()
            spikes_in_rest[t, 1] = binned_spikes_in_rest[:, t].std()
            spikes_in_rest[t, 2] = binned_spikes_in_rest[:, t].sum()

        w = self.params['w_input_exc'] * 1000. # *1000. for us --> nS
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.errorbar(self.t_axis, w * spikes_in_good[:, 0], yerr=w * spikes_in_good[:, 1], label='in \'good\' cells')
        ax.errorbar(self.t_axis, w * spikes_in_rest[:, 0], yerr=w * spikes_in_rest[:, 1], label='in rest')
        ax.set_title('Average input conductance per cell')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Conductance [nS]')
        ax.legend()

        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt+1)
        ax.plot(self.t_axis, w * spikes_in_good[:, 2], label='in \'good\' cells')
        ax.plot(self.t_axis, w * spikes_in_rest[:, 2], label='in \'rest\' cells')
        ax.set_title('Total input conductance into groups')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Conductance [nS]')
        ax.legend()


    def plot_cond_errorbar(self, d, fig_cnt=1, title=None, label=None):
        """
        Requires a call of get_intra_network_conductances before
        d[:, 0] : population mean  vs time
        d[:, 1] : population std   vs time
        d[:, 2] : population sum   vs time
        """

        if title == None:
            title = 'Average conductances '
        if label == None:
            label = '$g_{sum} = %.1e$ nS' % (d[:, 2].sum())

        params = { 'legend.fontsize': 10}
        pylab.rcParams.update(params)
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.errorbar(self.t_axis, 1000. * d[:, 0], yerr=1000. * d[:, 1], label=label)
        ax.set_title(title)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Conductances [nS]')
        ax.legend()


    def plot_conductances(self):

        self.create_fig()
        self.n_fig_x, self.n_fig_y = 1, 4
        self.get_intra_network_conductances()
        print 'Calculating conductance from good --> good'
        self.cond_good_to_good = self.get_input_cond(self.good_gids, self.good_gids)#, spike_fn)
        print 'Calculating conductance from rest -> good'
        self.cond_rest_to_good = self.get_input_cond(self.rest_gids, self.good_gids)
        print 'Calculating conductance from good --> rest'
        self.cond_good_to_rest = self.get_input_cond(self.good_gids, self.rest_gids)
        print 'Calculating conductance from rest --> rest'
        self.cond_rest_to_rest = self.get_input_cond(self.rest_gids, self.rest_gids)
        self.plot_cond_errorbar(self.cond_good_to_good, fig_cnt=1, title='Conductance good --> good')
        self.plot_cond_errorbar(self.cond_rest_to_good, fig_cnt=2, title='Conductance rest --> good')
        self.plot_cond_errorbar(self.cond_good_to_rest, fig_cnt=3, title='Conductance good --> rest')
        self.plot_cond_errorbar(self.cond_rest_to_rest, fig_cnt=4, title='Conductance rest --> rest')


    def get_intra_network_conductances(self, conn_list_fn=None):
        # calculate the (mean, std) conductances between different groups of neurons
        # good (g) -> good
        # g -> rest (r)
        # r -> g
        # r -> r
        # g -> inh
        # r -> inh
        # all exc -> inh
        # inh -> all exc
        # inh -> r
        # inh -> g

        if conn_list_fn == None:
            if self.params['initial_connectivity'] == 'precomputed':
                conn_list_fn = self.params['merged_conn_list_ee']
            else: 
                conn_list_fn = self.params['random_weight_list_fn'] + '0.dat'
        print 'utils.get_conn_dict from file:', conn_list_fn 
        self.conn_dict = utils.get_conn_dict(self.params, conn_list_fn)
        spike_fn = self.params['exc_spiketimes_fn_merged'] + '0.ras'


    def get_input_cond(self, tgts, srcs):#, source_spike_fn):
        """
        Calculates the time course for the input conductances into a population of cells given by 'gids'.
        Should be called after retrieving the connection dictionary from utils (-->util.get_conn_dict, e.g. from self.get_intra_network_conductances)
        This only works on binned spikes and hence gives valid information only if
        synaptic time constants are shorter than the time_binsize.
        tgts : list of target gids
        srcs : list of source gids
        """
        g_in_all = np.zeros((len(tgts), self.n_bins))

        for i_, tgt in enumerate(tgts):
            # get the sources 
            srcs_into_tgt = self.conn_dict[tgt]['sources']
            for src in srcs:
                if src in srcs_into_tgt:
                    j = srcs_into_tgt.index(src)
                    w = self.conn_dict[tgt]['w_in'][j]
                    g_in_all[i_, :] += self.nspikes_binned[src, :] * w

        g_in = np.zeros((self.n_bins, 3))
        for t in xrange(self.n_bins):
            g_in[t, 0] = g_in_all[:, t].mean()
            g_in[t, 1] = g_in_all[:, t].std()
            g_in[t, 2] = g_in_all[:, t].sum()
        return g_in


    def plot_grid_vs_time(self, data, title='', xlabel='', ylabel='', yticks=[], fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        ax.set_title(title)
        cax = ax.pcolor(data)
        ax.set_ylim((0, data[:, 0].size))
        ax.set_xlim((0, data[0, :].size))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        y_ticks = range(len(yticks))[::2]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['%.2f' %i for i in yticks[::2]])

        ax.set_xticks(range(self.n_bins)[::4])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::4]])
        pylab.colorbar(cax)


    

    def plot_nspikes_binned(self, gids, title=None, fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        if title == None:
            title = 'Spiking activity over time'
        ax.set_title(title)
        self.cax = ax.pcolor(self.nspikes_binned[gids, :])
        ax.set_ylim((0, self.nspikes_binned[gids, 0].size))
        ax.set_xlim((0, self.nspikes_binned[0, gids].size))
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('GID')
        ax.set_xticks(range(self.n_bins)[::2])
        ax.set_xticklabels(['%d' %i for i in self.time_bins[::2]])
        pylab.colorbar(self.cax)

    def plot_nspike_histogram(self, gids, fig_cnt=1):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        mean_nspikes = self.nspikes[gids].mean()* 1000./self.params['t_sim'] 
        std_nspikes = self.nspikes[gids].std() * 1000./self.params['t_sim']
        ax.bar(gids, self.nspikes[gids] * 1000./self.params['t_sim'], label='$f_{mean} = (%.1f \pm %.1f)$ Hz' % (mean_nspikes, std_nspikes))
        ax.set_xlabel('Cell gids')
        ax.set_ylabel('Output rate $f_{out}$')
        ax.legend()

    def make_infotextbox(self):
        pass

