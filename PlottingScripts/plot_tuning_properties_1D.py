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
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from FigureCreator import plot_params
pylab.rcParams.update(plot_params)
import set_tuning_properties

class Plotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        tp_fn = self.params['tuning_prop_exc_fn']
        rfs_fn = self.params['receptive_fields_exc_fn']
        if not (os.path.exists(tp_fn)) or not (os.path.exists(rfs_fn)):
            print 'RECOMPUTING tuning properties'
#            self.tp, self.rfs = set_tuning_properties.set_tuning_properties(self.params)
#            self.tp, self.rfs = set_tuning_properties.set_tuning_properties_and_rfs_const_fovea(self.params)
            self.tp, self.rfs = set_tuning_properties.set_tuning_prop_1D_with_const_fovea_and_const_velocity(self.params)
        else:
            print 'Loading', tp_fn
            self.tp = np.loadtxt(tp_fn)
            print 'Loading', rfs_fn
            self.rfs = np.loadtxt(self.params['receptive_fields_exc_fn'])

        # old
#            self.tp = utils.set_tuning_prop(self.params, mode='hexgrid', cell_type='exc')
#        if not os.path.exists(rfs_fn):
#            n_cells = self.params['n_exc']
#            self.rfs = np.zeros((n_cells, 4))
#            self.rfs[:, 0] = self.params['rf_size_x_gradient'] * np.abs(self.tp[:, 0] - .5) + self.params['rf_size_x_min']
#            self.rfs[:, 1] = self.params['rf_size_y_gradient'] * np.abs(self.tp[:, 1] - .5) + self.params['rf_size_y_min']
#            self.rfs[:, 2] = self.params['rf_size_vx_gradient'] * np.abs(self.tp[:, 2]) + self.params['rf_size_vx_min']
#            self.rfs[:, 3] = self.params['rf_size_vy_gradient'] * np.abs(self.tp[:, 3]) + self.params['rf_size_vy_min']
#        else:

    def plot_tuning_prop(self):

        tp = self.tp
        fig = pylab.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        n_bins = 40
        ax1.set_title('Distribution of spatial receptive fields')
        ax1.plot(tp[:, 0], tp[:, 1], marker='o', c='k', markersize=1, ls='')

        ax3.set_title('Histogram of spatial receptive fields')
        cnt, bins = np.histogram(tp[:, 0], bins=n_bins)
        ax3.bar(bins[:-1], cnt, width=bins[1]-bins[0])

        ax2.set_title('Distribution of speed tunings')
        ax2.plot(tp[:, 2], tp[:, 3], marker='o', c='k', markersize=1, ls='')

        ax4.set_title('Histogram of speed tunings')
        cnt, bins = np.histogram(tp[:, 2], bins=n_bins)
        ax4.bar(bins[:-1], cnt, width=bins[1]-bins[0])

        if os.path.exists(self.params['figures_folder']):
            output_fn = self.params['figures_folder'] + 'tuning_property_distribution.png'
            print 'Saving to:', output_fn
            pylab.savefig(output_fn)



    def plot_tuning_space(self):

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Receptive field center $x$', fontsize=18)
        ax.set_ylabel('Preferred speed', fontsize=18)
        patches = []
        for gid in xrange(self.tp[:, 0].size):
            ax.plot(self.tp[gid, 0], self.tp[gid, 2], 'o', c='k', markersize=3)
            ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2], linewidth=1.)
            patches.append(ellipse)

        collection = PatchCollection(patches, alpha=0.2, facecolor='blue', linewidth=1)
        ax.add_collection(collection)
        ylim = ax.get_ylim()
        ax.set_ylim((1.1 * ylim[0], 1.1 * ylim[1]))
        if os.path.exists(self.params['figures_folder']):
            output_fn = self.params['figures_folder'] + 'tuning_space.png'
            print 'Saving to:', output_fn
            pylab.savefig(output_fn)


    def plot_tuning_width_distribution(self):
        """
        Creates a two dimensional plot with rf_center on x-axis and rf-size on the y-axis
        """
        n_cells = self.params['n_exc']
        assert (n_cells  == self.rfs[:, 0].size), 'Mismatch in parameters given to plot_tuning_properties and simulation_parameters.py'

        fig = pylab.figure()
        pylab.subplots_adjust(hspace=0.5)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        x_axis = self.tp[:, 0]
        y_axis = self.rfs[:, 0]
        ax1.plot(x_axis, y_axis, marker='o', linestyle='None', markersize=2, c='k')
        ax1.set_xlabel('RF_x center')
        ax1.set_ylabel('RF_x size')
        ax1.set_title('Preferred position tuning widths')

        x_axis = self.tp[:, 2]
        y_axis = self.rfs[:, 2]
        ax2.plot(x_axis, y_axis, marker='o', linestyle='None', markersize=2, c='k')
        ax2.set_title('Preferred speed tuning widths')
        ax2.set_xlabel('RF_vx center')
        ax2.set_ylabel('RF_vx size')

        tp = self.tp
        fig = pylab.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        sort_idx_x = np.argsort(tp[:, 0])
        sort_idx_v = np.argsort(tp[:, 2])
        ax1.plot(range(tp[:, 0].size), tp[sort_idx_x, 0], 'o', ls='')
        ax2.plot(range(tp[:, 2].size), tp[sort_idx_v, 2], 'o', ls='')
        ax3.plot(range(tp[:, 0].size - 1), tp[sort_idx_x[1:], 0] - tp[sort_idx_x[:-1], 0], 'o', ls='', c='k', markersize=5)
        ax4.plot(range(tp[:, 2].size - 1), tp[sort_idx_v[1:], 2] - tp[sort_idx_v[:-1], 2], 'o', ls='', c='k', markersize=5)
        ax3.set_ylabel('Differences in tp')
        ax3.set_xlabel('GID')
        ax4.set_xlabel('GID')
        ax1.set_title('Spatial receptive field positions')
        ax2.set_title('Preferred speeds')


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

    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_tuning_prop()
    Plotter.plot_tuning_space()
    Plotter.plot_tuning_width_distribution()

    pylab.show()
