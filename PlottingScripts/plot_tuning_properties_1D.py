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
            self.tp, self.rfs = set_tuning_properties.set_tuning_properties_and_rfs_const_fovea(self.params)
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

#        fig = pylab.figure(figsize=utils.get_figsize(600, portrait=True))
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Feature space')
        ax.set_xlabel('Receptive field center $x$', fontsize=20)
        ax.set_ylabel('Preferred speed $v_x$', fontsize=20)
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
        else:
            output_fn = 'tuning_property_space.png'
        print 'Saving to:', output_fn
        pylab.savefig(output_fn, dpi=150)


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

#    pylab.show()
