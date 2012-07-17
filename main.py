import os
import simulation_parameters
import numpy as np
import utils
import SimulationManager
import calculate_conductances as cc
import plot_conductances
import plot_prediction

# for parallel execution
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

# load simulation parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

simStarter = SimulationManager.SimulationManager(PS, comm)

simStarter.create_folders()

do_prepare = False
# # # # # # # # # # # # 
#     P R E P A R E   #
# # # # # # # # # # # #
if (do_prepare):
    simStarter.prepare_tuning_properties()
    if comm != None:
        comm.Barrier()
    simStarter.prepare_spiketrains(simStarter.params['tuning_prop_means_fn'])
#    simStarter.prepare_connections()

n_sim = params['n_sim']
for sim_cnt in xrange(n_sim):

    # # # # # # # # # # # # # #
    #     S I M U L A T E     #
    # # # # # # # # # # # # # #
    print "Simulation run: %d / %d" % (sim_cnt+1, n_sim)
    simStarter.run_sim(connect_exc_exc=False)
    print "Simulation ended on proc %d / %d" % (pc_id, n_proc)
    if comm != None:
        comm.Barrier()

    if pc_id == 0:
        plot_prediction.plot_prediction(simStarter.params)
        cc.run_all(simStarter.params)
        plot_conductances.plot_conductances(simStarter.params)

    # # # # # # # # # # #
    #     B C P N N     #
    # # # # # # # # # # #
#    connection_matrix = np.loadtxt(params['conn_mat_ee_fn_base'] + str(sim_cnt) + '.npy')
#    utils.threshold_weights(connection_matrix, params['w_thresh_bcpnn'])
#    np.savetxt('debug_conn_mat_after_thresh_%d.txt' % (sim_cnt), connection_matrix)
#    Bcpnn.bcpnn_offline(params, connection_matrix, sim_cnt, pc_id, n_proc, True)
#    if comm != None:
#        comm.Barrier()


