import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import matplotlib
import pylab
import numpy as np
import sys
import os
import utils
import re

gid_axis = 0
time_axis = 1

class ActivityPlotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        if it_max == None:
            self.it_max = self.params['n_training_stim']
        else:
            self.it_max = it_max

        self.spike_times_loaded = False
        self.n_bins_x = 30
        self.n_x_ticks = 10
        self.x_ticks = np.linspace(0, self.n_bins_x, self.n_x_ticks)
        self.load_tuning_prop()


    def load_spike_data(self, cell_type):
        """
        cell_type -- 'exc', 'inh_spec', 'inh_unspec'
        """
        if cell_type == 'exc':
            fn = self.params['exc_spiketimes_fn_merged']
        elif cell_type == 'inh_spec':
            fn = self.params['inh_spec_spiketimes_fn_merged']
        elif cell_type == 'inh_unspec':
            fn = self.params['inh_unspec_spiketimes_fn_merged']
        else:
            print 'Invalid cell type provided to ActivityPlotter.load_spike_data: %s' % cell_type
        if not (os.path.exists(fn)):
            utils.merge_and_sort_files(params['%s_spiketimes_fn_base' % cell_type], params['%s_spiketimes_fn_merged' % cell_type])
        d = np.loadtxt(fn)
        print 'Debug cell_type %s size %d' % (cell_type, d.size)
        return d


    def load_tuning_prop(self):
        print 'ActivityPlotter.load_tuning_prop ...'
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
#        self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])

        self.x_grid = np.linspace(0, 1, self.n_bins_x, endpoint=False)
        self.gid_to_posgrid_mapping = utils.get_grid_index_mapping(self.tuning_prop_exc[:, 0], self.x_grid)




    def get_nspikes_interval(self, d, t0, t1):
        """
        d -- np.array containg the spike times (col 0 - gids, col 1 - times)
        """
        nspikes = np.zeros(self.params['n_exc'])
        for gid in xrange(1, self.params['n_exc'] + 1):
            cell_spikes = d[(d[:, 0] == gid).nonzero()[0], 1]
            idx = ((cell_spikes >= t0) == (cell_spikes <= t1)).nonzero()[0]
            nspikes[gid - 1] = idx.size
#            print 'cell %d spikes between %d - %d: %d times' % (gid, t0, t1, nspikes[gid - 1])
        return nspikes


    def plot_raster_sorted(self, title='', cell_type='exc', sort_idx=0):
        """
        sort_idx : the index in tuning properties after which the cell gids are to be sorted for  the rasterplot
        """
        if cell_type == 'exc':
            tp = self.tuning_prop_exc
        else:
            tp = self.tuning_prop_ing

        tp_idx_sorted = tp[:, sort_idx].argsort() # + 1 because nest indexing

        if not  self.spike_times_loaded:
            merged_spike_fn = self.params['exc_spiketimes_fn_merged']
            spikes_unsrtd = np.loadtxt(merged_spike_fn)
            self.spike_times_merged = spikes_unsrtd
            self.spike_times_loaded = True

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        for i_, gid in enumerate(tp_idx_sorted):
            spikes = utils.get_spiketimes(self.spike_times_merged, gid + 1)
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * tp[gid, sort_idx]
            ax.plot(spikes, y_, 'o', markersize=3, markeredgewidth=0., color='k')

        return fig, ax


    def plot_input_spikes_sorted(self, ax=None, title='', sort_idx=0):
        """
        Input spikes are stored in seperate files for each cell.
        --> filenames are Folder/InputSpikes/stim_spike_train_[GID].dat GID = cell gid
        """
        print 'plotting input spikes ...'

        if ax == None:
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            ax.set_title(title)

        tp = self.tuning_prop_exc
        tp_idx_sorted = tp[:, sort_idx].argsort() # + 1 because nest indexing

        cnt_file = 0
        for fn in os.listdir(self.params['input_folder']):
            m = re.match('stim_spike_train_(\d+).dat', fn)
            if m:
                gid = int(m.groups()[0])
                if (gid < self.params['n_exc']):
                    y_pos_of_cell = tp[gid, sort_idx]
                    fn_ = self.params['input_folder'] + fn
                    d = np.loadtxt(fn_)
                    ax.plot(d, y_pos_of_cell * np.ones(d.size), 'o', markersize=3, markeredgewidth=0., alpha=.1, color='b')
                    cnt_file += 1

        print 'Found %d files to plot:' % cnt_file


    def plot_nspike_histogram_in_MCs(self, d, cell_type='exc', time_range=False, f_max=100, output_fn=None):
        """
        Plot the number of spikes fired by the MCs within a certain time_range
        d          -- spike raw data
        cell_type  -- 'exc', 'inh_spec' or 'inh_unspec'
        time_range -- tuple with time boundaries
        """
        if d.size == 0:
            print '\n\tWARNING! plot_nspike_histogram_in_MCs got emtpy spike data for cell_type %s\n' % cell_type
            return

        if cell_type == 'exc':
            n_per_mc = self.params['n_exc_per_mc']
            n_per_hc = self.params['n_exc_per_hc']
            n_columns = self.params['n_mc']
            gid_offset = self.params['exc_offset']
        elif cell_type == 'inh_spec':
            n_per_mc = self.params['n_inh_per_mc']
            n_per_hc = self.params['n_inh_per_hc']
            n_columns = self.params['n_mc']
            gid_offset = self.params['inh_spec_offset']
        elif cell_type == 'inh_unspec':
            n_per_mc = 1
            n_per_hc = self.params['n_inh_unspec_per_hc']
            n_columns = self.params['n_inh_unspec']
            gid_offset = self.params['inh_unspec_offset']

        if time_range == False:
            time_range = (0, self.params['t_sim'])
            spikes = d[:, time_axis]
            gids = d[:, gid_axis]
        else:
            spikes = np.array(([], []))
            idx_0 = (d[:, time_axis] > time_range[0]).nonzero()[0]
            idx_1 = (d[:, time_axis] <= time_range[1]).nonzero()[0]
            idx = np.array(list(set(idx_0).intersection(set(idx_1))))
            spikes = d[idx, time_axis]
            gids = d[idx, gid_axis]

        nspikes_per_mc = np.zeros(n_columns)
        for gid in np.unique(gids):
            nspikes = (gid == gids).nonzero()[0].size
            mc_idx = (gid - 1 - gid_offset) / n_per_mc
#            print 'debug cell_type gid mc_idx', cell_type, gid, mc_idx
            nspikes_per_mc[mc_idx] += nspikes
        
        nspikes_per_mc  /= (time_range[1] - time_range[0]) / 1000.# transform into rate
        nspikes_per_mc  /= float(n_per_mc)
        print 'Plot histogram time_range:', time_range, 'nspikes max', gid, np.max(nspikes_per_mc)
        if f_max < np.max(nspikes_per_mc):
            print '\n\tWARNING: plot_nspike_histogram_in_MCs f_max < nspikes_per_mc\n'

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.bar(range(n_columns), nspikes_per_mc, width=1)
        ax.set_title('Activity during time %d - %d' % (time_range[0], time_range[1]))
        ax.set_ylabel('Mean output rate (averaged over %d %s cells) [Hz]' % (n_per_mc, cell_type))
        ax.set_xlabel('Column index')
        ax.set_ylim((0, f_max))
        ax.set_xlim((0, n_columns))
        if cell_type == 'inh_unspec':
            n_per_module = n_per_hc
        else:
            n_per_module = self.params['n_mc_per_hc']
        for hc in xrange(self.params['n_hc']):
            mc_0 = hc * n_per_module
            mc_1 = (hc + 1) * n_per_module
            ax.plot((mc_0, mc_0), (0, f_max), '--', c='k', lw=1)
            ax.plot((mc_1, mc_1), (0, f_max), '--', c='k', lw=1)

        if output_fn != None:
            print 'Saving to:', output_fn
            pylab.savefig(output_fn, dpi=300)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'
        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)

    else:
        import simulation_parameters
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params


    Plotter = ActivityPlotter(params)#, it_max=1)
    exc_spike_data = Plotter.load_spike_data('exc')
    inh_spec_spike_data = Plotter.load_spike_data('inh_spec')
    inh_unspec_spike_data = Plotter.load_spike_data('inh_unspec')
    fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by x-position', sort_idx=0)
    Plotter.plot_input_spikes_sorted(ax, sort_idx=0)

    time_steps = 1
    time_window = params['t_sim'] / time_steps
    f_max = 40 / time_steps
    for i_ in xrange(time_steps):
        time_range = (i_ * time_window, (i_ + 1) * time_window)
        output_fn = params['figures_folder'] + 'mc_exc_nspike_histogram_%02d.png' % i_
        Plotter.plot_nspike_histogram_in_MCs(exc_spike_data, cell_type='exc', time_range=(time_range[0], time_range[1]), f_max=f_max, output_fn=output_fn)

#        output_fn = params['figures_folder'] + 'mc_inh_nspike_histogram_%02d.png' % i_
#        Plotter.plot_nspike_histogram_in_MCs(inh_spec_spike_data, cell_type='inh_spec', time_range=(time_range[0], time_range[1]), f_max=f_max, output_fn=output_fn)

#        output_fn = params['figures_folder'] + 'hc_inh_nspike_histogram_%02d.png' % i_
#        Plotter.plot_nspike_histogram_in_MCs(inh_unspec_spike_data, cell_type='inh_unspec', time_range=(time_range[0], time_range[1]), f_max=f_max, output_fn=output_fn)

#    fig.savefig(params['figures_folder'] + 'rasterplot_in_and_out.png')
#    Plotter.plot_raster_sorted(title='Exc cells sorted by $v_x$', sort_idx=2)

    pylab.show()
