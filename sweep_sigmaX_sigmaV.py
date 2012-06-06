import numpy as np
import os
import SimulationManager
import time

# analysis modules
import plot_prediction


sigma_x_start = 0.20
sigma_x_step = 0.05
sigma_x_stop = sigma_x_start + 2 * sigma_x_step
sigma_x_range = np.arange(sigma_x_start, sigma_x_stop, sigma_x_step)

sigma_v_start = 0.05
sigma_v_step = 0.05
sigma_v_stop = sigma_v_start + 2 * sigma_v_step
sigma_v_range = np.arange(sigma_v_start, sigma_v_stop, sigma_v_step)

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
for sigma_v in sigma_v_range:
    for sigma_x in sigma_x_range:
        # -----   pre-computed connectivity 
        new_params = {  'initial_connectivity' : 'precomputed', 'w_sigma_x' : sigma_x, 'w_sigma_v' : sigma_v}
        simStarter.update_values(new_params)

        print "Prepare for w_sigma_v", simStarter.params['w_sigma_v']
        simStarter.create_folders()
        simStarter.prepare_tuning_properties()
        if i_ == 0:
            input_folder = str(simStarter.params['input_folder']) # where spikes will be created
            simStarter.prepare_spiketrains(simStarter.params['tuning_prop_means_fn'])
        else:
            simStarter.copy_folder(input_folder, simStarter.params['folder_name'])

        print "Running precomputed with sigma_v", i_, sigma_v, ' sigma_x ', sigma_x
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
        print "Running random with sigma_v", i_, sigma_v
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
