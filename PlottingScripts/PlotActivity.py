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
import json
from FigureCreator import plot_params
pylab.rcParams.update(plot_params)

gid_axis = 0
time_axis = 1

class ActivityPlotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        if it_max == None:
            self.n_stim_total = self.params['n_stim']
        else:
            self.n_stim_total = it_max

        self.spike_times_loaded = False
        self.n_bins_x = 30
        self.n_x_ticks = 10
        self.x_ticks = np.linspace(0, self.n_bins_x, self.n_x_ticks)
        self.load_tuning_prop()
        gids_f = self.params['gid_fn']
        f = file(gids_f, 'r')
        self.gids_dict = json.load(f)


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
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_exc_fn'])
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

    def plot_nspike_histogram_vs_gids(self, spike_data, cell_type='exc'):
        """
        spike_data is the raw rasterplot data
        """
        n_cells = self.params['n_%s' % cell_type]
        nspikes = utils.get_nspikes(spike_data, n_cells=n_cells)
        idx_0 = (nspikes == 0).nonzero()[0]
#        print 'Cells that did not fire any spikes:'
#        for gid in idx_0:
#            print 'tp[%d, :] = ' % gid, self.tuning_prop_exc[gid, :]
#        print 'Number of cells that fired zero spikes:', idx_0, idx_0.size
        x = range(n_cells)
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.bar(x, nspikes, width=1)
        ax.set_xlim((0, n_cells))
        ax.set_ylabel('Number of spikes')
        ax.set_title('Number of spikes fired by %s cells' % cell_type)
        ax.set_xlabel('Cell GIDs')


    def plot_raster_simple(self, title='', cell_type='exc', time_range=None):

        merged_spike_fn = self.params['%s_spiketimes_fn_merged' % cell_type]
        print 'Loading spikes from:', merged_spike_fn
        spikes_unsrtd = np.loadtxt(merged_spike_fn)
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        for gid in xrange(self.params['%s_offset' % cell_type], self.params['%s_offset' % cell_type] + self.params['n_%s' % cell_type]):
            gid_nest = gid + 1
            spikes = utils.get_spiketimes(spikes_unsrtd, gid_nest)
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * gid
            ax.plot(spikes, y_, 'o', markersize=3, markeredgewidth=0., color='k')

        ylim = ax.get_ylim()
        training_stim_duration = np.loadtxt(self.params['training_stim_durations_fn'])
        for i_stim in xrange(self.params['n_stim']):
            t0 = training_stim_duration[:i_stim].sum()
            t1 = training_stim_duration[:i_stim+1].sum()
            ax.plot((t0, t0), (0, ylim[1]), ls='--', c='k')
            ax.plot((t1, t1), (0, ylim[1]), ls='--', c='k')
            ax.text(t0 + .5 * (t1 - t0), 0.90 * ylim[1], '%d' % i_stim)

        if time_range != None:
            ax.set_xlim((time_range[0], time_range[1]))
        output_fn = self.params['figures_folder'] + '%s_raster_simple.png' % (cell_type)
        print 'Saving to:', output_fn
        fig.savefig(output_fn, dpi=300)
        return fig, ax


    def plot_raster_sorted(self, title='', cell_type='exc', sort_idx=0, time_range=None):
        """
        sort_idx : the index in tuning properties after which the cell gids are to be sorted for  the rasterplot
        """
        if cell_type == 'exc':
            tp = self.tuning_prop_exc
        else:
            tp = self.tuning_prop_inh

        n_cells = self.params['n_%s' % cell_type]
        if not self.spike_times_loaded:
            merged_spike_fn = self.params['%s_spiketimes_fn_merged' % cell_type]
            spikes_unsrtd = np.loadtxt(merged_spike_fn)
            self.spike_times_merged = spikes_unsrtd
            self.spike_times_loaded = True

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel('Time [ms]')
        if sort_idx == 0:
            ax.set_ylabel('Cell position')
        elif sort_idx == 2:
            ax.set_ylabel('Preferred speed')
        for gid in xrange(n_cells):
            gid_nest = gid + 1
            spikes = utils.get_spiketimes(self.spike_times_merged, gid_nest)
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * tp[gid, sort_idx]
            ax.plot(spikes, y_, 'o', markersize=3, markeredgewidth=0., color='k')

        ylim = ax.get_ylim()


        training_stim_duration = np.loadtxt(self.params['training_stim_durations_fn'])
        for i_stim in xrange(self.params['n_stim']):
            t0 = training_stim_duration[:i_stim].sum()
            t1 = training_stim_duration[:i_stim+1].sum()
            ax.plot((t0, t0), (0, ylim[1]), ls='--', c='k')
            ax.plot((t1, t1), (0, ylim[1]), ls='--', c='k')
            ax.text(t0 + .5 * (t1 - t0), 0.90 * ylim[1], '%d' % i_stim)

        if time_range != None:
            ax.set_xlim((time_range[0], time_range[1]))
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
            # alternative: self.gids_dict[cell_type]['gid_to_column'][gid]
            mc_idx = (gid - 1 - gid_offset) / n_per_mc
            nspikes_per_mc[mc_idx] += nspikes
        
        nspikes_per_mc  /= (time_range[1] - time_range[0]) / 1000.# transform into rate
        nspikes_per_mc  /= float(n_per_mc)
        print 'Plot histogram time_range:', time_range, 'nspikes max', gid, np.max(nspikes_per_mc)
        if f_max < np.max(nspikes_per_mc):
            print '\n\tWARNING: plot_nspike_histogram_in_MCs f_max < nspikes_per_mc\n'

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.bar(range(n_columns), nspikes_per_mc, width=1)
        ax.set_title('%s activity during time %d - %d [ms]' % (cell_type.capitalize(), time_range[0], time_range[1]))
        ax.set_ylabel('Mean output rate\n(avg over %d cells) [Hz]' % (n_per_mc))
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


    def plot_spike_rate_vs_time(self, spike_data, binsize=25., time_range=None, cell_type='exc'):

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

        # build a color scheme
        # define the colormap
        cmap = matplotlib.cm.jet

        # if you want to modify the colormap:
        # extract all colors from the cmap
#        cmaplist = [cmap(i) for i in xrange(cmap.N)]

        # force the first color entry to be grey 
        #cmaplist[0] = (.5,.5,.5,1.0)
        # create the new map
#        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)


        fig = pylab.figure()
        ax = fig.add_subplot(111)
        bounds = range(params['n_mc_per_hc'])
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(np.arange(bounds[0], bounds[-1], 1.))
        rgba_colors = m.to_rgba(bounds)
        cb = fig.colorbar(m)
        cb.set_label('Minicolumn index')#, fontsize=24)

        if time_range == None:
            time_range = (0, self.params['t_sim'])
            spikes = spike_data[:, time_axis]
            gids = spike_data[:, gid_axis]
        else:
            spikes, gids = utils.get_spikes_within_interval(spike_data, time_range[0], time_range[1], time_axis=time_axis, gid_axis=gid_axis)

        n_bins = np.int(np.round((time_range[1] - time_range[0]) / binsize))
#        for gid in np.unique(gids):
#            key = str(int(gid))
#            hc_idx, mc_idx = self.gids_dict['gid_to_column'][cell_type][key]
#            color = rgba_colors[mc_idx]
#            idx = (gids == gid).nonzero()[0]
#            cells_spikes = spikes[idx]
#            hist, edges = np.histogram(cells_spikes, bins=n_bins, range=time_range)
#            print 'debug hist edges', hist.size, edges.size

#            ax.plot(edges[:-1] + .5 * binsize,  hist * (1000. / binsize), lw=1, c=color)

            # another way of coloring (global mc index determines color)
#            mc = hc_idx * self.params['n_mc_per_hc'] + mc_idx
#            color = rgba_colors[mc]

        for i_hc in xrange(self.params['n_hc']):
            for i_mc in xrange(self.params['n_mc_per_hc']):
                cell_gids = self.gids_dict['column_to_gid'][cell_type][str(i_hc)][str(i_mc)]
                mc_activity = np.zeros(n_bins)
                for gid in cell_gids:
                    idx = (gids == gid).nonzero()[0]
                    cells_spikes = spikes[idx]
                    hist, edges = np.histogram(cells_spikes, bins=n_bins, range=time_range)
                    mc_activity += hist

                color = rgba_colors[i_mc]
                ax.plot(edges[:-1] + .5 * binsize,  mc_activity * (1000. / binsize) / n_per_mc, lw=1, c=color)





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

    trained_stim = params['trained_stimuli']
    training_stim_duration = np.loadtxt(params['training_stim_durations_fn'])
    stim_range = params['stim_range']
    binsize = 50
    for i_stim, stim in enumerate(range(stim_range[0], stim_range[1])):
        t0 = training_stim_duration[:stim].sum()
        t1 = training_stim_duration[:stim+1].sum()
        Plotter.plot_spike_rate_vs_time(exc_spike_data, binsize=binsize, time_range=(t0, t1))

    #inh_spec_spike_data = Plotter.load_spike_data('inh_spec')
    inh_unspec_spike_data = Plotter.load_spike_data('inh_unspec')

    Plotter.plot_nspike_histogram_vs_gids(exc_spike_data)

#    Plotter.plot_spike_rate_vs_time(exc_spike_data, binsize=10)

#    Plotter.plot_spike_rate_vs_time(inh_unspec_spike_data, binsize=15)

    time_range = None
#    stim = 1

#    Plotter.plot_raster_simple(title='Inh unspecific neurons', cell_type='inh_unspec')
#    Plotter.plot_raster_simple(title='Exc neurons', cell_type='exc')

    print 'Time range', time_range
    fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by x-position', sort_idx=0, time_range=time_range)
    Plotter.plot_input_spikes_sorted(ax, sort_idx=0)
    output_fn = params['figures_folder'] + 'exc_raster_x.png'
    print 'Saving to:', output_fn
    fig.savefig(output_fn, dpi=300)

#    fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by $v_x$', sort_idx=2, time_range=time_range)
#    Plotter.plot_input_spikes_sorted(ax, sort_idx=2)
#    output_fn = params['figures_folder'] + 'exc_raster_vx.png'
#    print 'Saving to:', output_fn
#    fig.savefig(output_fn, dpi=300)


#    trained_stim = params['trained_stimuli']
#    training_stim_duration = np.loadtxt(params['training_stim_durations_fn'])
#    stim_range = [0, 2]
#    f_max = 350.
#    for i_stim, stim in enumerate(range(stim_range[0], stim_range[1])):
#        t0 = training_stim_duration[:stim].sum()
#        t1 = training_stim_duration[:stim+1].sum()
#        print 'Stim:', stim, 'time_range:', t0, t1, 'mp:', trained_stim[stim]
#        output_fn = params['figures_folder'] + 'mc_exc_nspike_histogram_stim%02d.png' % (stim)
#        Plotter.plot_nspike_histogram_in_MCs(exc_spike_data, cell_type='exc', time_range=(t0, t1), f_max=f_max, output_fn=output_fn)
#        pylab.savefig(output_fn)

    pylab.show()
