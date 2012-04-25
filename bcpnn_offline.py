import numpy as np
import time
import os
import re
import utils
import simulation_parameters
import Bcpnn
import CreateConnections
from mpi4py import MPI
comm = MPI.COMM_WORLD

pc_id, n_proc = comm.rank, comm.size
print "Start process %d / %d " % (pc_id+1, n_proc)

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
folder = params['spiketimes_folder']
fn_base = params['exc_spiketimes_fn_base'].rsplit(folder)[1]

CC = CreateConnections.create_initial_connection_matrix(params['n_mc'], output_fn=params['conn_mat_init'], sparseness=params['conn_mat_init_sparseness'])
connection_matrix = np.load(params['conn_mat_init'])
non_zeros = connection_matrix.nonzero()
conns = zip(non_zeros[0], non_zeros[1])
my_conns = utils.distribute_list(conns, n_proc, pc_id)

#for i in xrange(len(my_conns)):
for i in xrange(1):
    print "Pc %d conn:" % pc_id, i
    pre_id = my_conns[i][0]
    post_id = my_conns[i][1]
    fn_pre = params['exc_spiketimes_fn_base'] + str(pre_id) + '.ras'
    fn_post = params['exc_spiketimes_fn_base'] + str(post_id) + '.ras'
    # load data
    spiketimes = np.loadtxt(fn_pre)[:, 0]
    spiketimes = np.loadtxt(fn_post)[:, 0]

    # convert
    pre_trace = Bcpnn.convert_spiketrain_to_trace(spiketimes, params['t_sim'])
    post_trace = Bcpnn.convert_spiketrain_to_trace(spiketimes, params['t_sim'])

    # compute
    wij, bias, pi, pj, pij = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace)

    # save
    output_fn = params['weights_fn_base'] + "%d_%d.npy" % (pre_id, post_id)
    np.save(output_fn, wij)

    output_fn = params['bias_fn_base'] + "%d_%d.npy" % (pre_id, post_id)
    np.save(output_fn, bias)


#def convert_spiketrain_to_trace(st, n):
#def get_spiking_weight_and_bias(pre_trace, post_trace, bin_size=1, tau_z=10, tau_e=100, tau_p=1000, dt=1, f_max=300., eps=1e-6):
