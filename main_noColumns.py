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
import time
import prepare_sim as Prep
try:
    from mpi4py import MPI
    USE_MPI = True
except:
    USE_MPI = False

if USE_MPI:
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
else:
    pc_id, n_proc, comm = 0, 1, None
    
import NetworkSimModuleNoColumns as simulation
#import NetworkSimModule as simulation

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

# # # # # # # # # # # # 
#     P R E P A R E   #
# # # # # # # # # # # #
do_BCPNN = False

do_prepare = not(os.path.isdir(params['folder_name']))
network_params.create_folders()
#n_proc = 2
if (do_prepare):
    Prep.prepare_sim(comm)
#    if (n_proc > 1):
#        os.system("mpirun -np %d python prepare_sim.py" % n_proc)
#    else:
#        os.system("python prepare_sim.py")

n_sim = params['n_sim']
for sim_cnt in xrange(n_sim):
    # # # # # # # # # # # # # #
    #     S I M U L A T E     #
    # # # # # # # # # # # # # #
    
    if (pc_id == 0):
        print "Simulation run: %d / %d" % (sim_cnt+1, n_sim)
        simulation.run_sim(params, sim_cnt)
    print "Pc %d waiting for proc 0 to finish simulation" % pc_id
    if USE_MPI: comm.barrier()

#    if (n_proc > 1):
#        os.system ("mpirun -np %d python NetworkSimModuleNoColumns.py %d" % (n_proc, sim_cnt))
#    else:
#        os.system ("python NetworkSimModuleNoColumns.py %d" % sim_cnt)

    if do_BCPNN:
        # # # # # # # # # # #
        #     B C P N N     #
        # # # # # # # # # # #
        t1 = time.time()
        print "Pc %d Bcpnn ... " % pc_id
        conn_list = np.loadtxt(params['conn_list_ee_fn_base'] + str(sim_cnt) + '.dat')
        Bcpnn.bcpnn_offline_noColumns(params, conn_list, sim_cnt, False, comm)
        t2 = time.time()
        print "Computation time for BCPNN: %d sec or %.1f min for %d cells" % (t2-t1, (t2-t1)/60., params['n_cells'])
    #    if (n_proc > 1):
    #        os.system ("mpirun -np %d python use_bcpnn_offline.py %d" % (n_proc, sim_cnt))
    #    else:
    #        os.system ("python use_bcpnn_offline.py %d" % sim_cnt)
        
    if USE_MPI: comm.barrier()

#    utils.threshold_weights(connection_matrix, params['w_thresh_bcpnn'])

#    np.savetxt('debug_conn_mat_after_thresh_%d.txt' % (sim_cnt), connection_matrix)
#    print "Computing bcpnn traces %d" % (sim_cnt + 1)
#    Bcpnn.bcpnn_offline_noColumns(params, connection_matrix, sim_cnt, pc_id, n_proc, True)


