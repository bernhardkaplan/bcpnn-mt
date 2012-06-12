import numpy as np
import os
import SimulationManager
import time

# analysis modules
import plot_prediction


p_thresh_start = 1e-4
#p_thresh_step = 0.2
#p_thresh_stop = p_thresh_start + 4 * p_thresh_step
#p_thresh_range = np.arange(p_thresh_start, p_thresh_stop, p_thresh_step)
p_thresh_range = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

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

import simulation_parameters

PS = simulation_parameters.parameter_storage()

t_start = time.time()
simStarter = SimulationManager.SimulationManager(PS, comm)

i_ = 0
for p_thresh in p_thresh_range:
    # -----   pre-computed connectivity 
    new_params = {  'initial_connectivity' : 'precomputed', 'p_thresh_connection' : p_thresh}
    simStarter.update_values(new_params)

    simStarter.create_folders()
    simStarter.prepare_tuning_properties()
    if i_ == 0:
        input_folder = str(simStarter.params['input_folder']) # where spikes will be created
        simStarter.prepare_spiketrains(simStarter.params['tuning_prop_means_fn'])
#        pass
    else:
        simStarter.copy_folder(input_folder, simStarter.params['folder_name'])

    print "Running precomputed with p_thresh", p_thresh
    simStarter.prepare_connections()
    simStarter.run_sim()
    # analysis 1
    if pc_id == 0:
        plot_prediction.plot_prediction(simStarter.params)
    if comm != None:
        comm.barrier()

    # copy files from the previous folder needed for the next simulation
    src1 = simStarter.params['input_folder']
    src2 = simStarter.params['bias_folder']
    src3 = simStarter.params['parameters_folder']
    connections_fn = simStarter.params['conn_list_ee_fn_base'] + '0.dat'

    new_params = { 'initial_connectivity' : 'random'}
    simStarter.update_values(new_params)
    tgt1 = simStarter.params['folder_name']
    if pc_id == 0:
        simStarter.copy_folder(input_folder, tgt1)
        simStarter.copy_folder(src1, tgt1)
        simStarter.copy_folder(src2, tgt1)
        simStarter.copy_folder(src3, tgt1)
    if comm != None:
        comm.barrier()
    simStarter.create_folders()

    # random connectivity
    simStarter.prepare_connections(connections_fn)
    simStarter.run_sim()
    if pc_id == 0:
        plot_prediction.plot_prediction(simStarter.params)
    if comm != None:
        comm.barrier()
    i_ += 1

t_stop = time.time()
t_run = t_stop - t_start
print "Full sweep duration: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t_run, (t_run)/60., \
        simStarter.params['n_cells'], simStarter.params['n_exc'], simStarter.params['n_inh'])
