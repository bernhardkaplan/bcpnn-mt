"""
This is the main script to be started on a single core, i.e. without MPI.
To run the preparation, simulation and analysis scripts on mutliple cores
modify the variable n_proc according to your needs.

"""
import os
import simulation_parameters
import numpy as np
import utils
import CreateConnections as CC
import Bcpnn
#import NetworkSimModuleNoColumns as simulation
#import NetworkSimModule as simulation

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

do_prepare = False
# # # # # # # # # # # # 
#     P R E P A R E   #
# # # # # # # # # # # #
n_proc = 8
if (do_prepare):
    if (n_proc > 1):
        os.system("mpirun -np %d python prepare_sim.py" % n_proc)
    else:
        os.system("python prepare_sim.py")

n_sim = params['n_sim']
for sim_cnt in xrange(n_sim):
    # # # # # # # # # # # # # #
    #     S I M U L A T E     #
    # # # # # # # # # # # # # #
    print "Simulation run: %d / %d" % (sim_cnt+1, n_sim)
    if (n_proc > 1):
        os.system ("mpirun -np %d python NetworkSimModuleNoColumns.py %d" % (n_proc, sim_cnt))
    else:
        os.system ("python NetworkSimModuleNoColumns.py %d" % sim_cnt)

    # # # # # # # # # # #
    #     B C P N N     #
    # # # # # # # # # # #
    if (n_proc > 1):
        os.system ("mpirun -np %d python use_bcpnn_offline.py %d" % (n_proc, sim_cnt))
    else:
        os.system ("python use_bcpnn_offline.py %d" % sim_cnt)

#    utils.threshold_weights(connection_matrix, params['w_thresh_bcpnn'])

#    np.savetxt('debug_conn_mat_after_thresh_%d.txt' % (sim_cnt), connection_matrix)
#    print "Computing bcpnn traces %d" % (sim_cnt + 1)
#    Bcpnn.bcpnn_offline_noColumns(params, connection_matrix, sim_cnt, pc_id, n_proc, True)


