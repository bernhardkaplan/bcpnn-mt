import matplotlib
#matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy as np
import utils
import CreateConnections as CC
import sys

if sys.argv < 2:
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
    print "Taking the default parameters from", params['params_fn']

else:
    param_fn = sys.argv[1]
    print "Loading data from:", param_fn
    import NeuroTools.parameters as ntp
    ParamSet = ntp.ParameterSet(param_fn)
    params = ParamSet.as_dict()

#print params['test']
#print params['response_3d_fig']

class PlotCellResponsesPlusProperties(object):
    def __init__(self, params):
        """
        params can be either a standard dictionary or a NeuroTools.parameters ParameterSet
        """
        self.params = params
        self.tp = np.loadtxt(params['tuning_prop_means_fn'])
        self.mp = params['motion_params']
        print "Motion parameters", self.mp
        self.sim_cnt = 0

    def get_nspikes(self):
        spiketimes_fn_merged = self.params['exc_spiketimes_fn_merged'] + str(self.sim_cnt) + '.ras'
        print "Loading nspikes", spiketimes_fn_merged
        self.nspikes = utils.get_nspikes(spiketimes_fn_merged, self.params['n_exc'])

    def get_weights(self):
        conn_list_fn = self.params['conn_list_ee_fn_base'] + '0.dat'
        self.conn_mat, self.delays = utils.convert_connlist_to_matrix(conn_list_fn, self.params['n_exc'])
        self.w_in_sum = np.zeros(self.params['n_exc'])
        n_cells = self.params['n_exc']
        for i in xrange(n_cells):
            self.w_in_sum[i] = self.conn_mat[:, i].sum()

    def get_distances_between_cell_and_stimulus(self):
        n_cells = self.params['n_exc']
        self.dist_to_stim = np.zeros(n_cells)
        for i in xrange(n_cells):
            self.dist_to_stim[i] = utils.get_min_distance_to_stim(self.mp, self.tp[i, :])

    def plot(self):

        fig = pylab.figure()
        n_cells = self.params['n_exc']
        d = np.zeros((n_cells, 3))
        d[:, 0] = self.w_in_sum
        d[:, 1] = self.dist_to_stim
        d[:, 2] = self.nspikes

        ax = Axes3D(fig)
        #ax = fig.add_subplot(111, projection='3d')

        ax.scatter(d[:,0], d[:,1], d[:,2], marker='o')

        ax.set_xlabel('sum$(w_{in})$')
        ax.set_ylabel('$d(cell, stim)$')
        ax.set_zlabel('$n_{spikes}$')

        sigma_x, sigma_v = self.params['w_sigma_x'], self.params['w_sigma_v'] # small sigma values let p and w shrink
        ax.set_title('$\sigma_X=%.2e\t\sigma_V=%.2e' % (sigma_x, sigma_v))
        print "Saving to:", self.params['response_3d_fig']
        pylab.savefig(self.params['response_3d_fig'])
#        pylab.savefig('test_3d.png')
        pylab.show()

Plotter = PlotCellResponsesPlusProperties(params)
Plotter.get_nspikes()
Plotter.get_weights()
Plotter.get_distances_between_cell_and_stimulus()
Plotter.plot()
