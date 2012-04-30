
import os
import simulation_parameters
import numpy as np
import utils
import CreateConnections as CC
import Bcpnn
#import NetworkSimModuleNoColumns as simulation
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
    if (n_proc > 1):
        os.system("mpirun -np %d python prepare_sim.py")
    else:
        os.system("python prepare_sim.py")

n_sim = params['n_sim']
for sim_cnt in xrange(n_sim):
    # # # # # # # # # # # # # #
    #     S I M U L A T E     #
    # # # # # # # # # # # # # #
    print "Simulation run: %d / %d" % (sim_cnt+1, n_sim)
    if (n_proc > 1):
        os.system ("mpirun -np %d python NetworkSimModuleNoColumns.py %d" % sim_cnt)
    else:
        os.system ("python NetworkSimModuleNoColumns.py %d" % sim_cnt)

    print "Simulation ended on proc %d / %d" % (pc_id, n_proc)
    comm.barrier()

    # # # # # # # # # # #
    #     B C P N N     #
    # # # # # # # # # # #
    connection_matrix = np.loadtxt(params['conn_list_ee_fn_base'] + str(sim_cnt) + '.dat')
#    utils.threshold_weights(connection_matrix, params['w_thresh_bcpnn'])

#    np.savetxt('debug_conn_mat_after_thresh_%d.txt' % (sim_cnt), connection_matrix)
    print "Computing bcpnn traces %d" % (sim_cnt + 1)
    Bcpnn.bcpnn_offline_noColumns(params, connection_matrix, sim_cnt, pc_id, n_proc, True)
    comm.barrier()


