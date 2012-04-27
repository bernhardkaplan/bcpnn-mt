import os
import simulation_parameters
import numpy as np
import utils
import CreateConnections as CC
import Bcpnn
import NetworkSimModuleNoColumns as simulation
#import NetworkSimModule as simulation

# for parallel execution
from mpi4py import MPI
comm = MPI.COMM_WORLD
pc_id, n_proc = comm.rank, comm.size

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

do_prepare = True
# # # # # # # # # # # # 
#     P R E P A R E   #
# # # # # # # # # # # #
if (do_prepare):
    tuning_prop = utils.set_tuning_prop(params['n_exc'], mode='random')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
    np.savetxt(params['tuning_prop_means_fn'], tuning_prop)
    x0, y0, u0, v0 = params['motion_params']
    motion = (x0, y0, u0, v0)
    if (n_proc > 1): # distribute the number of minicolumns among processors
        my_units = utils.distribute_n(params['n_exc'], n_proc, pc_id)
    else:
        my_units = (0, params['n_exc'])
    # create the input
    input_spike_trains = utils.create_spike_trains_for_motion(tuning_prop, motion, params, my_units) # write to paths defined in the params dictionary

    # create initial connections 
    CC.create_conn_list(params['conn_list_ee_fn_base']+'0.dat', (0, params['n_exc']), (0, params['n_exc']), params['p_ee'], params['w_ee_mean'], params['w_ee_sigma'])

#    CC.create_conn_list(params['conn_list_ei_fn'],              (0, params['n_exc']), (params['n_exc'], params['n_inh']), params['p_ei'], params['w_ei_mean'], params['w_ei_sigma'])
#    CC.create_conn_list(params['conn_list_ie_fn'],              (params['n_exc'], params['n_inh']), (0, params['n_exc']), params['p_ie'], params['w_ie_mean'], params['w_ie_sigma'])
#    CC.create_conn_list(params['conn_list_ii_fn'],              (params['n_exc'], params['n_inh'])params['n_exc'], params['n_inh'], params['p_ii'], params['w_ii_mean'], params['w_ii_sigma'])
#    CC.create_conn_list(params['conn_list_input_fn'],           params['n_exc'], params['n_inh'], params['p_ii'], params['w_ii_mean'], params['w_ii_sigma'])

    # write inital bias values to file
    np.savetxt(params['bias_values_fn_base']+'0.npy', np.zeros(params['n_exc']))

n_sim = params['n_sim']
for sim_cnt in xrange(n_sim):
    # # # # # # # # # # # # # #
    #     S I M U L A T E     #
    # # # # # # # # # # # # # #
    print "Simulation run: %d / %d" % (sim_cnt+1, n_sim)
    simulation.run_sim(params, sim_cnt)

    print "Simulation ended on proc %d / %d" % (pc_id, n_proc)
    comm.barrier()

    # # # # # # # # # # #
    #     B C P N N     #
    # # # # # # # # # # #
    connection_matrix = np.loadtxt(params['conn_mat_ee_fn_base'] + str(sim_cnt) + '.npy')
#    utils.threshold_weights(connection_matrix, params['w_thresh_bcpnn'])
#    np.savetxt('debug_conn_mat_after_thresh_%d.txt' % (sim_cnt), connection_matrix)
    Bcpnn.bcpnn_offline(params, connection_matrix, sim_cnt, pc_id, n_proc, True)
    comm.barrier()


