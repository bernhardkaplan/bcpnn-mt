import numpy as np
import os
import SimulationManager
import time
import plot_conductances
import calculate_conductances as cc
import plot_prediction

parameter_name = 'p_to_w_scaling'
parameter_start = 100
parameter_step = 100
parameter_stop = parameter_start + 1 * parameter_step
parameter_range = np.arange(parameter_start, parameter_stop, parameter_step)

#parameter_name = 'blur'
#parameter_start = 0.5
#parameter_step = 0.1
#parameter_stop = parameter_start + 6 * parameter_step
#parameter_range = np.arange(parameter_start, parameter_stop, parameter_step)

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

for i_, param_value in enumerate(parameter_range):
    # -----   pre-computed connectivity 
    new_params = {  'initial_connectivity' : 'precomputed', parameter_name : param_value}
#    new_params = {  'initial_connectivity' : 'precomputed', 'blur_X' : param_value, 'blur_V' : param_value}
    simStarter.update_values(new_params)

    simStarter.create_folders()
    simStarter.prepare_tuning_properties()
    if comm != None:
        comm.Barrier()

    if i_ == 0:
        input_folder = str(simStarter.params['input_folder']) # where spikes will be created
#        simStarter.prepare_spiketrains(simStarter.params['tuning_prop_means_fn'])
    else:
        simStarter.copy_folder(input_folder, simStarter.params['folder_name'])

    simStarter.prepare_connections()
    simStarter.run_sim()

    # analysis 1
    if pc_id == 0:
        plot_prediction.plot_prediction(simStarter.params)
        cc.run_all(simStarter.params)
        plot_conductances.plot_conductances(simStarter.params)

    if comm != None:
        comm.Barrier()
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

    simStarter.prepare_connections(connections_fn)
    simStarter.run_sim()
    if pc_id == 0:
        plot_prediction.plot_prediction(simStarter.params)
        cc.run_all(simStarter.params)
        plot_conductances.plot_conductances(simStarter.params)
    if comm != None:
        comm.barrier()

t_stop = time.time()
t_run = t_stop - t_start
print "Full sweep duration: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t_run, (t_run)/60., \
        simStarter.params['n_cells'], simStarter.params['n_exc'], simStarter.params['n_inh'])
