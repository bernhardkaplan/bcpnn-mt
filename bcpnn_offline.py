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

save_all = True # if True: z and e traces will be written to disk

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

CC = CreateConnections.create_initial_connection_matrix(params['n_mc'], output_fn=params['conn_mat_init'], sparseness=params['conn_mat_init_sparseness'])
connection_matrix = np.load(params['conn_mat_init'])
non_zeros = connection_matrix.nonzero()
conns = zip(non_zeros[0], non_zeros[1])
my_conns = utils.distribute_list(conns, n_proc, pc_id)

#for i in xrange(len(my_conns)):
for i in xrange(2):
    print "Pc %d conn:" % pc_id, i, my_conns[i]
    pre_id = my_conns[i][0]
    post_id = my_conns[i][1]
    fn_pre = params['exc_spiketimes_fn_base'] + str(pre_id) + '.ras'
    fn_post = params['exc_spiketimes_fn_base'] + str(post_id) + '.ras'
    # load data
    spiketimes_pre = np.loadtxt(fn_pre)[:, 0]
    spiketimes_post = np.loadtxt(fn_post)[:, 0]

    # convert
#    print "spike_times_pre", spiketimes_pre
#    print "spike_times_post", spiketimes_post
    pre_trace = utils.convert_spiketrain_to_trace(spiketimes_pre, params['t_sim'])
    post_trace = utils.convert_spiketrain_to_trace(spiketimes_post, params['t_sim'])

    # compute
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace)

    # save
    output_fn = params['weights_fn_base'] + "%d_%d.npy" % (pre_id, post_id)
    np.save(output_fn, wij)

    output_fn = params['bias_fn_base'] + "%d.npy" % (post_id)
    np.save(output_fn, bias)

    if (save_all):
        output_fn = params['ztrace_fn_base'] + "%d.npy" % pre_id
        np.save(output_fn, zi)
        output_fn = params['ztrace_fn_base'] + "%d.npy" % post_id
        np.save(output_fn, zj)

        output_fn = params['etrace_fn_base'] + "%d.npy" % pre_id
        np.save(output_fn, ei)
        output_fn = params['etrace_fn_base'] + "%d.npy" % post_id
        np.save(output_fn, ej)
        output_fn = params['etrace_fn_base'] + "%d_%d.npy" % (pre_id, post_id)
        np.save(output_fn, eij)

        output_fn = params['ptrace_fn_base'] + "%d.npy" % pre_id
        np.save(output_fn, pi)
        output_fn = params['ptrace_fn_base'] + "%d.npy" % post_id
        np.save(output_fn, pj)
        output_fn = params['ptrace_fn_base'] + "%d_%d.npy" % (pre_id, post_id)
        np.save(output_fn, pij)

#def get_spiking_weight_and_bias(pre_trace, post_trace, bin_size=1, tau_z=10, tau_e=100, tau_p=1000, dt=1, f_max=300., eps=1e-6):
