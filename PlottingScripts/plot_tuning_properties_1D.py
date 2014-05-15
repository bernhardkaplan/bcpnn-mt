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

class Plotter(object):

    def __init__(self, params, it_max=None):
        self.params = params
        tp_fn = self.params['tuning_prop_means_fn']
        print 'Loading', tp_fn
        self.tp = np.loadtxt(tp_fn)
        print 'Loading', self.params['receptive_fields_exc_fn']
        self.rfs = np.loadtxt(self.params['receptive_fields_exc_fn'])

    def plot_tuning_prop(self):

        tp = self.tp
        fig = pylab.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title('Distribution of spatial receptive fields')
        ax1.scatter(tp[:, 0], tp[:, 1], marker='o', c='k')

        ax3.set_title('Histogram of spatial receptive fields')
        cnt, bins = np.histogram(tp[:, 0], bins=20)
        ax3.bar(bins[:-1], cnt, width=bins[1]-bins[0])

        ax2.set_title('Distribution of speed tunings')
        ax2.scatter(tp[:, 2], tp[:, 3], marker='o', c='k')

        ax4.set_title('Histogram of speed tunings')
        cnt, bins = np.histogram(tp[:, 2], bins=20)
        ax4.bar(bins[:-1], cnt, width=bins[1]-bins[0])

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
            ellipse = mpatches.Ellipse((self.tp[gid, 0], self.tp[gid, 2]), self.rfs[gid, 0], self.rfs[gid, 2])
            patches.append(ellipse)

        collection = PatchCollection(patches, alpha=0.1)
        ax.add_collection(collection)
        ylim = ax.get_ylim()
        ax.set_ylim((1.1 * ylim[0], 1.1 * ylim[1]))
        output_fn = self.params['figures_folder'] + 'tuning_space.png'
        print 'Saving to:', output_fn
        pylab.savefig(output_fn)



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

    pylab.show()
