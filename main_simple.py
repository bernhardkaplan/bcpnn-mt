import os
import simulation_parameters
import numpy as np
import utils
import CreateConnections as CC
import plot_conductances
import plot_prediction
import NetworkSimModuleNoColumns as simulation

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
PS.create_folders()
PS.write_parameters_to_file()
print 'n_cells=%d\tn_exc=%d\tn_inh=%d' % (params['n_cells'], params['n_exc'], params['n_inh'])

analyse = True
# # # # # # # # # # # # 
#     P R E P A R E   #
# # # # # # # # # # # #
prepare_tuning_prop = True
prepare_spike_trains = True
prepare_connections = False
if (prepare_tuning_prop):
    tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
    if pc_id == 0:
        print "Saving tuning_prop to file:", pc_id, params['tuning_prop_means_fn']
        np.savetxt(params['tuning_prop_means_fn'], tuning_prop)
        np.savetxt(params['bias_values_fn_base']+'0.dat', np.zeros(params['n_exc'])) # write inital bias values to file
    if comm != None:
        comm.Barrier()
else:
    tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])

if (prepare_spike_trains):
    my_units = utils.distribute_n(params['n_exc'], n_proc, pc_id)
    input_spike_trains = utils.create_spike_trains_for_motion(tuning_prop, params, contrast=.9, my_units=my_units) # write to paths defined in the params dictionary
    if comm != None:
        comm.barrier()

if (prepare_connections and params['connect_exc_exc']):
    if params['initial_connectivity'] == 'precomputed':
        CC.compute_weights_from_tuning_prop(tuning_prop, params, comm)
        # optional
        if pc_id == 0:
            os.system('python balance_connlist.py')
        if comm != None:
            comm.Barrier()
    else:
#        input_fn = params['conn_list_ee_fn_base'] + '0.dat'
        input_fn = params['conn_list_ee_balanced_fn']
        output_fn = params['random_weight_list_fn'] + '0.dat'
        CC.compute_random_weight_list(input_fn, output_fn, params)


if analyse and pc_id == 0:
    tp = np.loadtxt(params['tuning_prop_means_fn'])
    mp = params['motion_params']
    indices, distances = utils.sort_gids_by_distance_to_stimulus(tp , mp) # cells in indices should have the highest response to the stimulus
    n = 10
#    gids_to_record = indices[:n]
    np.savetxt(params['gids_to_record_fn'], indices, fmt='%d')


# # # # # # # # # # # # # #
#     S I M U L A T E     #
# # # # # # # # # # # # # #
#sim_cnt = 0
#if (pc_id == 0):
#    print "Simulation run: %d cells (%d exc, %d inh)" % (params['n_cells'], params['n_exc'], params['n_inh'])
#    simulation.run_sim(params, sim_cnt, params['initial_connectivity'], params['connect_exc_exc'])

#if comm != None:
#    comm.Barrier()

# # # # # # # # # # # # # #
#     A N A L Y S I S     #
# # # # # # # # # # # # # #
#if analyse and pc_id == 0:
#    plot_prediction.plot_prediction(params)
#    os.system('python analyse_simple.py')
#    print 'blur:', params['blur_X'], params['blur_V']
#    print 'w_sigma:', params['w_sigma_x'], params['w_sigma_v']
#    os.system('python plot_voltage_noCompatibleOutput.py %s.v' % params['exc_volt_fn_base'])
#    import calculate_conductances as cc
#    cc.run_all(params)
#    plot_conductances.plot_conductances(params)

