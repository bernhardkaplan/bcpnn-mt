import os
import simulation_parameters
import numpy as np
import utils
import CreateConnections as CC
import Bcpnn
import NetworkSimModule as sim

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
    tuning_prop = utils.set_tuning_prop(params, mode='random')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
    np.savetxt(params['tuning_prop_means_fn'], tuning_prop)
    x0, y0, u0, v0 = params['motion_params']
    motion = (x0, y0, u0, v0)
    if (n_proc > 1): # distribute the number of minicolumns among processors
        my_units = utils.distribute_n(params['n_mc'], n_proc, pc_id)
    else:
        my_units = (0, params['n_mc'])
    # create the input
    input_spike_trains = utils.create_spike_trains_for_motion(tuning_prop, motion, params, my_units) # write to paths defined in the params dictionary
    # initial connectivity between minicolumns is written to a file
    conn_mat_init_fn = params['conn_mat_ee_fn_base'] + '0.npy'
    CC.create_initial_connection_matrix(params['n_mc'], output_fn=conn_mat_init_fn, \
            w_max=params['w_init_max'], sparseness=params['conn_mat_init_sparseness'])
    CC.create_connection_between_cells(params, conn_mat_init_fn)
    # write inital bias values to file
    np.savetxt(params['bias_values_fn_base']+'0.npy', np.zeros(params['n_exc_per_mc']))

    
n_sim = params['n_sim']
for sim_cnt in xrange(n_sim):
    # # # # # # # # # # # # # #
    #     S I M U L A T E     #
    # # # # # # # # # # # # # #
    print "Simulation run: %d / %d" % (sim_cnt+1, n_sim)
    sim.run_sim(params, sim_cnt)
#    if (pc_id == 0):
#        print "mpirun -np %d python network_sim.py nest %d" % (n_proc, sim_cnt)
#        os.system("mpirun -np %d python network_sim.py nest %d" % (n_proc, sim_cnt))

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


