import simulation_parameters
import numpy as np
import utils
import os
import sys
import time
import random
#import Bcpnn

"""
This script builds a one-dimensional model of cells with preference for position and direction (2-dimenstional tuning properties).
It sets the tuning properties and creates an abstract input sequence into each cell, which can be transformed into spike trains.
"""

class OneDimensionalAbstractModel(object):

    def __init__(self, params):
        self.params = params


    def set_tuning_properties(self):
        """
        Set the tuning properties for the 1-dimensional model.
        Keyword arguments:
        """
        self.n_units = self.params['N_RF_X'] * self.params['N_V']
        x_pos = np.linspace(0, 1, self.params['N_RF_X'] + 2)[1:-1]
        n_v = params['N_V']
        v_max = params['v_max_tp']
        v_min = params['v_min_tp']
        if params['log_scale']==1:
            v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
        else:
            v_rho = np.logspace(np.log(v_min)/np.log(params['log_scale']),
                            np.log(v_max)/np.log(params['log_scale']), num=n_v,
                            endpoint=True, base=params['log_scale'])

        self.tuning_prop = np.zeros((self.n_units, 2))
        unit_cnt = 0
        for i_, rf_x in enumerate(x_pos):
            for j_, rf_v in enumerate(v_rho):
                self.tuning_prop[unit_cnt, 0] = rf_x
                self.tuning_prop[unit_cnt, 1] = rf_v
                unit_cnt += 1

#        print 'x_pos:', x_pos.shape
#        print 'vx:', v_rho.shape
        print 'Tuning prop ', self.tuning_prop
        np.savetxt(self.params['tuning_prop_exc_fn'], self.tuning_prop)


    def create_stimuli(self, n_training_v=None, direction='forward'):
        """
        Keyword arguments:
        direction -- 'forward' means from left to right, 'backward' the opposite
        """

        self.n_training_speeds = n_training_v
        if n_training_v == None:
            n_training_v = self.params['n_training_v']

        v_max = self.params['v_max_tp']
        v_min = self.params['v_min_tp']

        if self.params['log_scale']==1:
            self.v_training = np.linspace(v_min, v_max, num=n_training_v, endpoint=True)
        else:
            self.v_training = np.logspace(np.log(v_min)/np.log(self.params['log_scale']),
                            np.log(v_max)/np.log(self.params['log_scale']), num=n_training_v,
                            endpoint=True, base=self.params['log_scale'])

        if direction == 'backward':
            self.v_training *= -1.
        print 'v_training:', self.v_training 

        stim_params = np.zeros((n_training_v, 2))
        # stim_params[:, 0] should store the start position
        stim_params[:, 1] = self.v_training
        output_fn = self.params['training_input_folder'] + 'stimulus_parameters.dat'
        np.savetxt(output_fn, stim_params)

        self.t_axis = np.arange(0, self.params['t_sim'], self.params['dt_rate'])
        self.trajectories = np.zeros((n_training_v, self.t_axis.size))
        x_offset = 0
        for i_, v_ in enumerate(self.v_training):
            self.trajectories[i_, :] = v_ * self.t_axis / 1000. + x_offset
        output_fn = self.params['training_input_folder'] + 'trajectories.dat'
        np.savetxt(output_fn, self.trajectories)



    def get_responses(self):


#        blur_X, blur_V = self.params['blur_X'], self.params['blur_V'] #0.5, 0.5
        blur_X, blur_V = .1, .1

        for i_ in xrange(self.n_training_speeds):
            trajectory = self.trajectories[i_, :]
            v_stim = self.v_training[i_]

            network_response = np.zeros((self.n_units, self.t_axis.size))

            for unit in xrange(self.n_units):
                network_response[unit, :] = np.exp(-.5 * ((trajectory - self.tuning_prop[unit, 0]) / blur_X)**2 \
                        - .5 * ((v_stim - self.tuning_prop[unit, 1]) / blur_V)**2)


            output_fn = self.params['activity_folder'] + 'network_response_no_plasticity_stim_%d.dat' % (i_)
            np.savetxt(output_fn, network_response)
#        L = np.exp(-.5 * ((torus_distance2D_vec(tuning_prop[:, 0], x*np.ones(n_cells), tuning_prop[:, 1], y*np.ones(n_cells)))**2 / blur_X**2)
#                -.5 * (tuning_prop[:, 2] - u0)**2 / blur_V**2
#                -.5 * (tuning_prop[:, 3] - v0)**2 / blur_V**2
#                )



if __name__ == '__main__':
    PS = simulation_parameters.parameter_storage()
    params = PS.params
    Model = OneDimensionalAbstractModel(params)
    try:
        from mpi4py import MPI
        USE_MPI = True
        comm = MPI.COMM_WORLD
        pc_id, n_proc = comm.rank, comm.size
        print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
    except:
        USE_MPI = False
        pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"

    if pc_id == 0:
        PS.create_folders()
        PS.write_parameters_to_file()
    Model.set_tuning_properties()
    Model.create_stimuli(n_training_v=params['N_V'], direction='forward')
    Model.get_responses()

