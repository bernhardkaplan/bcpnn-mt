import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
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


class ActivityPlotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        if it_max == None:
            self.it_max = self.params['n_training_stim']
        else:
            self.it_max = it_max

        self.n_bins_x = 30
        self.n_x_ticks = 10
        self.x_ticks = np.linspace(0, self.n_bins_x, self.n_x_ticks)
        self.load_tuning_prop()


    def load_tuning_prop(self):
        print 'ActivityPlotter.load_tuning_prop ...'
        self.tuning_prop_exc = np.loadtxt(self.params['tuning_prop_means_fn'])
        self.tuning_prop_inh = np.loadtxt(self.params['tuning_prop_inh_fn'])

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

        merged_spike_fn = self.params['exc_spiketimes_fn_merged']
        spikes_unsrtd = np.loadtxt(merged_spike_fn)

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        for i_, gid in enumerate(tp_idx_sorted):
            spikes = utils.get_spiketimes(spikes_unsrtd, gid + 1)
            nspikes = spikes.size
            y_ = np.ones(spikes.size) * tp[gid, sort_idx]
            ax.plot(spikes, y_, 'o', markersize=5, markeredgewidth=0., color='k')

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
                y_pos_of_cell = tp[gid, sort_idx]
                fn_ = self.params['input_folder'] + fn
                d = np.loadtxt(fn_)
                ax.plot(d, y_pos_of_cell * np.ones(d.size), 'o', markersize=5, markeredgewidth=0., alpha=.3, color='b')
                cnt_file += 1

        print 'Found %d files to plot:' % cnt_file


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

    utils.merge_and_sort_files(params['exc_spiketimes_fn_base'], params['exc_spiketimes_fn_merged'])
    Plotter = ActivityPlotter(params)#, it_max=1)
    fig, ax = Plotter.plot_raster_sorted(title='Exc cells sorted by x-position', sort_idx=0)
    Plotter.plot_input_spikes_sorted(ax, sort_idx=0)
    fig.savefig(params['figures_folder'] + 'rasterplot_in_and_out.png')

#    Plotter.plot_raster_sorted(title='Exc cells sorted by $v_x$', sort_idx=2)
    pylab.show()
