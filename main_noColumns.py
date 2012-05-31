"""
This is the main script to be started on a single core, i.e. without MPI.
To run the preparation, simulation and analysis scripts on mutliple cores
modify the variable n_proc according to your needs.

"""
import os
import simulation_parameters
import numpy as np
import utils
import Bcpnn
import time
import prepare_sim as Prep
import CreateConnections as CC

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
    
t_start = time.time()
print "USE_MPI:", USE_MPI
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
if pc_id == 0:
    network_params.create_folders()
    network_params.write_parameters_to_file(params['params_fn'])# write parameters to a file

t1 = time.time()
if (do_prepare):
    # prepare stimulus
    Prep.prepare_sim(comm)

if pc_id == 0 and params['initial_connectivity'] == 'precomputed':
    print "Proc %d computes initial weights ... " % pc_id
    tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
    CC.compute_weights_from_tuning_prop(tuning_prop, params)

elif pc_id == 0 and params['initial_connectivity'] == 'random':
    print "Proc %d shuffles pre-computed weights ... " % pc_id
    input_fn = 'NoColumns_winit_precomputed_wsigmaX1.0e-01_motionblur2.5e-01_pthresh1.0e-02_ptow1.0e-02/Connections/conn_list_ee_0.dat'
    output_fn = params['random_weight_list_fn'] + '0.dat'
    CC.compute_random_weight_list(input_fn, output_fn)

if comm != None:
    comm.barrier() # 



if USE_MPI: comm.barrier()
t2 = time.time()
print "Preparation time: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t2-t1, (t2-t1)/60., params['n_cells'], params['n_exc'], params['n_inh'])

n_sim = params['n_sim']
for sim_cnt in xrange(n_sim):
    # # # # # # # # # # # # # #
    #     S I M U L A T E     #
    # # # # # # # # # # # # # #
    
    if (pc_id == 0):
        print "Simulation run: %d / %d. %d cells (%d exc, %d inh)" % (sim_cnt+1, n_sim, params['n_cells'], params['n_exc'], params['n_inh'])
        simulation.run_sim(params, sim_cnt, params['initial_connectivity'])
    else: 
        print "Pc %d waiting for proc 0 to finish simulation" % pc_id

    if USE_MPI: 
        comm.barrier()


    if do_BCPNN:
        # # # # # # # # # # #
        #     B C P N N     #
        # # # # # # # # # # #
        t1 = time.time()
        print "Pc %d Bcpnn ... " % pc_id
        conn_list = np.loadtxt(params['conn_list_ee_fn_base'] + str(sim_cnt) + '.dat')
        Bcpnn.bcpnn_offline_noColumns(params, conn_list, sim_cnt, False, comm)
        t2 = time.time()
        print "Computation time for BCPNN: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t2-t1, (t2-t1)/60., params['n_cells'], params['n_exc'], params['n_inh'])
        
        if USE_MPI: comm.barrier()

t_stop = time.time()
t_run = t_stop - t_start
print "Full run duration: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t_run, (t_run)/60., params['n_cells'], params['n_exc'], params['n_inh'])
#    utils.threshold_weights(connection_matrix, params['w_thresh_bcpnn'])

#    np.savetxt('debug_conn_mat_after_thresh_%d.txt' % (sim_cnt), connection_matrix)
#    print "Computing bcpnn traces %d" % (sim_cnt + 1)
#    Bcpnn.bcpnn_offline_noColumns(params, connection_matrix, sim_cnt, pc_id, n_proc, True)


