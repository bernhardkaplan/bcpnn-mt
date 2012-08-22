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

# # # # # # # # # # # # 
#     P R E P A R E   #
# # # # # # # # # # # #
prepare_tuning_prop = False
prepare_spike_trains = False
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

if (prepare_connections):
    CC.compute_weights_from_tuning_prop(tuning_prop, params, comm)


# # # # # # # # # # # # # #
#     S I M U L A T E     #
# # # # # # # # # # # # # #
sim_cnt = 0
connect_exc_exc = True
if (pc_id == 0):
    print "Simulation run: %d cells (%d exc, %d inh)" % (params['n_cells'], params['n_exc'], params['n_inh'])
    simulation.run_sim(params, sim_cnt, params['initial_connectivity'], connect_exc_exc)

if comm != None:
    comm.Barrier()

# # # # # # # # # # # # # #
#     A N A L Y S I S     #
# # # # # # # # # # # # # #
analyse = True
if analyse and pc_id == 0:
    import calculate_conductances as cc
    plot_prediction.plot_prediction(params)
    cc.run_all(params)
    plot_conductances.plot_conductances(params)

