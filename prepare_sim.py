import utils
import simulation_parameters
import numpy as np
import CreateConnections as CC

def prepare_sim(comm):

    if (comm != None):
        pc_id, n_proc = comm.rank, comm.size
    else:
        pc_id, n_proc = 0, 1

    # load simulation parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
    tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
    if pc_id == 0:
        np.savetxt(params['tuning_prop_means_fn'], tuning_prop)
    if comm != None:
        comm.barrier() # 
    my_units = utils.distribute_n(params['n_exc'], n_proc, pc_id)
    if comm != None:
        print "DEBUG Proc %d processes " % pc_id, my_units

    # create the input
    input_spike_trains = utils.create_spike_trains_for_motion(tuning_prop, params, contrast=.9, my_units=my_units) # write to paths defined in the params dictionary

    # create initial connections 
    # with weights based on cell's tuning properties
    if comm != None:
        comm.barrier() # 
    print "Proc %d computes initial weights ... " % pc_id
#    CC.compute_weights_from_tuning_prop_distances(tuning_prop, params)
    if pc_id == 0 and params['initial_connectivity'] == 'precomputed':
        CC.compute_weights_from_tuning_prop(tuning_prop, params)

    # by random: # ugly function signature
#    CC.create_conn_list_by_random(params['conn_list_ee_fn_base']+'0.dat', (0, params['n_exc']), (0, params['n_exc']), params['p_ee'], params['w_ee_mean'], params['w_ee_sigma'])
    
    # write inital bias values to file
    np.savetxt(params['bias_values_fn_base']+'0.dat', np.zeros(params['n_exc']))




