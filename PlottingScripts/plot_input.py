import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np 
import utils
import json
import simulation_parameters
import re


def plot_rates(params):
    tp = np.loadtxt(params['tuning_prop_exc_fn'])

    feature_dimension = 4
    clim = (0., 180.)
    norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m.set_array(tp[:, feature_dimension])
    colorlist= m.to_rgba(tp[:, feature_dimension])

    clim2 = (0., 1.)
    norm2 = matplotlib.colors.Normalize(vmin=clim2[0], vmax=clim2[1])
    m2 = matplotlib.cm.ScalarMappable(norm=norm2, cmap=matplotlib.cm.jet) # large weights -- black, small weights -- white
    m2.set_array(tp[:, feature_dimension])
    colorlist2 = m2.to_rgba(tp[:, 0])


    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for fn in os.listdir(params['input_folder']):
        m_ = re.match('rate_(\d+)_(\d+)', fn)
        if m_:
            print m_.groups(), fn
            gid, stim_idx = int(m_.groups()[0]), int(m_.groups()[1])
            d = np.loadtxt(params['input_folder'] + fn)
            x_ = np.arange(d.size) * params['dt_rate']
            ax.plot(x_, d, c=m.to_rgba(tp[gid-1, feature_dimension]))


if __name__ == '__main__':

    folder = sys.argv[1]
    params = utils.load_params(folder)
    plot_rates(params)
    pylab.show()
