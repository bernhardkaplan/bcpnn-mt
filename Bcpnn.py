import numpy as np
import utils
import time
from NeuroTools import signals as nts


def bcpnn_offline_noColumns(params, conn_list, sim_cnt=0, save_all=False, comm=None):
    """
    This function computes the weight and bias values based on spiketimes during the simulation.

    Arguments:
        params: parameter dictionary
        conn_list:  two-dim numpy array storing cell-to-cell connections (only non-zero elements will be processed)
                            in the format (src, tgt, weight, delay)
                            or
                            file name in which the date is stored in this way
        sim_cnt: int for recording to file
        save_all: if True all traces will be saved
        comm = MPI communicator

    """
    if (type(conn_list) == type('')):
        d = np.load(conn_list)

    if (comm != None):
        pc_id, n_proc = comm.rank, comm.size
    else:
        pc_id, n_proc = 0, 1
    # extract the local list of elements 'my_conns' from the global conn_list
    n_total = len(conn_list)
    (min_id, max_id) = utils.distribute_n(n_total, n_proc, pc_id)
    my_conns = [(conn_list[i, 0], conn_list[i, 1], conn_list[i, 2], conn_list[i, 3]) for i in xrange(min_id, max_id)]

    fn = params['exc_spiketimes_fn_merged'] + str(sim_cnt) + '.ras'
    spklist = nts.load_spikelist(fn)#, range(params['n_exc_per_mc']), t_start=0, t_stop=params['t_sim'])
    spiketrains = spklist.spiketrains

    new_conn_list = np.zeros((len(my_conns), 4)) # (src, tgt, weight, delay)
    bias_dict = {}
    for i in xrange(params['n_exc']):
        bias_dict[i] = None
    
    for i in xrange(len(my_conns)):
#    for i in xrange(2):
        pre_id = my_conns[i][0]
        post_id = my_conns[i][1]

        # create traces from spiketimes
        # pre
        spiketimes_pre = spiketrains[pre_id+1.].spike_times
        pre_trace = utils.convert_spiketrain_to_trace(spiketimes_pre, params['t_sim'] + 1) # + 1 is to handle spikes in the last time step
        # post
        spiketimes_post = spiketrains[post_id+1.].spike_times
        post_trace = utils.convert_spiketrain_to_trace(spiketimes_post, params['t_sim'] + 1) # + 1 is to handle spikes in the last time step

        # compute
#        print "%d Computing traces for %d -> %d; %.2f percent " % (pc_id, pre_id, post_id, i / float(len(my_conns)) * 100.)
        get_traces = save_all
        if (get_traces):
            wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = get_spiking_weight_and_bias(pre_trace, post_trace, get_traces)
            dw = (wij.max() - wij.min()) * params['dw_scale']
            # bias update
            new_bias = bias.max()
        else:
            dw, new_bias = get_spiking_weight_and_bias(pre_trace, post_trace, get_traces)
            dw *= params['dw_scale']

        # bias update
        if bias_dict[post_id] == None:
            bias_dict[post_id] = new_bias


        # weight update
        new_conn_list[i, 0] = pre_id
        new_conn_list[i, 1] = post_id
        new_conn_list[i, 2] = dw + my_conns[i][2]
        new_conn_list[i, 3] = my_conns[i][3]

#        print "DEBUG Pc %d \t%d\t%d\t%.1e\t%.1e\tbias:%.4e\tconn:" % (pc_id, new_conn_list[i, 0], new_conn_list[i, 1],  new_conn_list[i, 2],  new_conn_list[i, 3], new_bias[i, 1]), my_conns[i]
        if (save_all):
            # save
            output_fn = params['weights_fn_base'] + "%d_%d.npy" % (pre_id, post_id)
            np.save(output_fn, wij)

            output_fn = params['bias_fn_base'] + "%d.npy" % (post_id)
            np.save(output_fn, bias)

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

    if (n_proc > 1):
        output_fn_conn_list = params['conn_list_ee_fn_base'] + str(sim_cnt+1) + '.dat'
        utils.gather_conn_list(comm, new_conn_list, n_total, output_fn_conn_list)

        output_fn_bias = params['bias_values_fn_base'] + str(sim_cnt+1) + '.dat'
        utils.gather_bias(comm, bias_dict, n_total, output_fn_bias)

    else:
        print "Debug saving to", params['conn_list_ee_fn_base'] + str(sim_cnt+1) + '.dat'
        np.savetxt(params['conn_list_ee_fn_base'] + str(sim_cnt+1) + '.dat', my_conns)#conn_list)
        print "Debug saving to", params['bias_values_fn_base'] + str(sim_cnt+1) + '.dat'
        np.savetxt(params['bias_values_fn_base'] + str(sim_cnt+1) + '.dat', bias)





def compute_traces(si, tau_z=10, tau_e=100, tau_p=1000, eps=1e-6, initial_value=None):
    dt = 1.
    n = si.size
    if initial_value == None:
        initial_value = (0.01, 0.01, 0.01)

    zi = np.ones(n) * initial_value[0]
    ei = np.ones(n) * initial_value[1]
    pi = np.ones(n) * initial_value[2]
    for i in xrange(1, n):
        dzi = dt * (si[i] - zi[i-1] + eps) / tau_z
        zi[i] = zi[i-1] + dzi

        # pre-synaptic trace zi follows zi
        dei = dt * (zi[i] - ei[i-1]) / tau_e
        ei[i] = ei[i-1] + dei

        # pre-synaptic probability pi follows zi
        dpi = dt * (ei[i] - pi[i-1]) / tau_p
        pi[i] = pi[i-1] + dpi

    return zi, ei, pi


def compute_pij(zi, zj, pi, pj, tau_eij, tau_pij, get_traces=False, dt=1., initial_values=(1e-4, 1e-4, 0, np.log(1e-2))):

    n = zi.size
    eij = np.ones(n) * initial_values[0]
    pij = np.ones(n) * initial_values[1]
    wij = np.ones(n) * initial_values[2]
    bias = np.ones(n) * initial_values[3]
    for i in xrange(1, n):
        # joint 
        deij = dt * (zi[i] * zj[i] - eij[i-1]) / tau_eij
        eij[i] = eij[i-1] + deij

        # joint probability pij follows zi * zj
        dpij = dt * (eij[i] - pij[i-1]) / tau_pij
        pij[i] = pij[i-1] + dpij

        # weights
        wij[i] = np.log(pij[i] / (pi[i] * pj[i]))

        # bias
        bias[i] = np.log(pj[i])

    if (get_traces):
        return wij, bias, pij, eij
    else:
        return pij[-1], wij[-1], bias[-1]


def get_spiking_weight_and_bias(pre_trace, post_trace, get_traces=False, bin_size=1, \
        tau_dict = None, dt=1., f_max=1000., initial_value=0.01):#, eps=1e-6):
    """
    Arguments:
        pre_trace, post_trace: pre-synaptic activity (0 means no spike, 1 means spike) (not spike trains!)
        
    """
    assert (len(pre_trace) == len(post_trace)), "Abstract pre and post activity have different lengths!"
    if tau_dict == None:
        tau_dict = {'tau_zi' : 10,    'tau_zj' : 10, 
                    'tau_ei' : 100,   'tau_ej' : 100, 'tau_eij' : 100,
                    'tau_pi' : 1000,  'tau_pj' : 1000, 'tau_pij' : 1000,
                    }
        print 'WARNING: No bcpnn parameters given, taking defaults. tau_dict=', tau_dict

#    if bin_size != 1:
#   TODO:
#        return get_spiking_weight_and_bias_binned(pre_spikes, post_spikes, bin_size=1, tau_z=10, tau_e=100, tau_p=1000, dt=1, eps=1e-2)

    eps = dt / tau_dict['tau_pi']
    n = len(pre_trace)
    si = pre_trace      # spiking activity (spikes have a width and a height)
    sj = post_trace
    zi = np.ones(n) * initial_value
    zj = np.ones(n) * initial_value
    ei = np.ones(n) * initial_value
    ej = np.ones(n) * initial_value
    eij = np.ones(n) * initial_value**2
    pi = np.ones(n) * initial_value
    pj = np.ones(n) * initial_value
    pij = np.ones(n) * initial_value**2
    wij = np.zeros(n)
    bias = np.ones(n) * np.log(initial_value)
    spike_height = 1000. / f_max

    for i in xrange(1, n):
        # pre-synaptic trace zi follows si
        dzi = dt * (si[i] - zi[i-1] + eps) / tau_dict['tau_zi']
        zi[i] = zi[i-1] + dzi

        # post-synaptic trace zj follows sj
        dzj = dt * (sj[i] * spike_height - zj[i-1] + eps) / tau_dict['tau_zj']
        zj[i] = zj[i-1] + dzj

        # pre-synaptic trace zi follows zi
        dei = dt * (zi[i] - ei[i-1]) / tau_dict['tau_ei']
        ei[i] = ei[i-1] + dei

        # post-synaptic trace ej follows zj
        dej = dt * (zj[i] - ej[i-1]) / tau_dict['tau_ej']
        ej[i] = ej[i-1] + dej

        # joint eij follows zi * zj
        deij = dt * (zi[i] * zj[i] - eij[i-1]) / tau_dict['tau_eij']
        eij[i] = eij[i-1] + deij

        # pre-synaptic probability pi follows zi
        dpi = dt * (ei[i] - pi[i-1]) / tau_dict['tau_pi']
        pi[i] = pi[i-1] + dpi

        # post-synaptic probability pj follows ej
        dpj = dt * (ej[i] - pj[i-1]) / tau_dict['tau_pj']
        pj[i] = pj[i-1] + dpj

        # joint probability pij follows e_ij
        dpij = dt * (eij[i] - pij[i-1]) / tau_dict['tau_pij']
        pij[i] = pij[i-1] + dpij

        # weights
        wij[i] = np.log(pij[i] / (pi[i] * pj[i]))

        # bias
        bias[i] = np.log(pj[i])

    return wij, bias, pi, pj, pij, ei, ej, eij, zi, zj



def bcpnn_offline(params, connection_matrix, sim_cnt=0, pc_id=0, n_proc=1, save_all=False):
    """
    Arguments:
        params: parameter dictionary
        connection_matrix: two-dim numpy array storing cell-to-cell connections (only non-zero elements will be processed)
                            or
                           file name
        sim_cnt: int for recording to file

    This function does basically the same thing as the script bcpnn_offline.py
    """
    if (type(connection_matrix) == type('')):
        connection_matrix = np.load(connection_matrix)
    non_zeros = connection_matrix.nonzero()
    conns = zip(non_zeros[0], non_zeros[1])
    my_conns = utils.distribute_list(conns, n_proc, pc_id)

    n, m = connection_matrix.shape
    for i in xrange(len(my_conns)):
#    for i in xrange(2):
        pre_id = my_conns[i][0]
        post_id = my_conns[i][1]

        # extract the spike times from the file where all cells belonging to one minicolumn are stored
        # pre
        mc_index_pre = pre_id / params['n_exc_per_mc']
        fn_pre = params['exc_spiketimes_fn_base'] + str(pre_id) + '.ras'
        spklist_pre = nts.load_spikelist(fn_pre, range(params['n_exc_per_mc']), t_start=0, t_stop=params['t_sim'])
        spiketimes_pre = spklist_pre[pre_id % params['n_exc_per_mc']].spike_times # TODO: check: + 1 for NeuroTools 
        pre_trace = utils.convert_spiketrain_to_trace(spiketimes_pre, params['t_sim'] + 1) # + 1 is to handle spikes in the last time step

        # post
        mc_index_post = post_id / params['n_exc_per_mc']
        fn_post = params['exc_spiketimes_fn_base'] + str(post_id) + '.ras'
        spklist_post = nts.load_spikelist(fn_post, range(params['n_exc_per_mc']), t_start=0, t_stop=params['t_sim'])
        spiketimes_post = spklist_post[post_id % params['n_exc_per_mc']].spike_times# TODO: check: + 1 for NeuroTools 
        post_trace = utils.convert_spiketrain_to_trace(spiketimes_post, params['t_sim'] + 1)

        # compute
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = get_spiking_weight_and_bias(pre_trace, post_trace)

        # update
        dw = (wij.max() - wij.min()) * params['dw_scale']
        print "DEBUG, updating weight[%d, %d] by %.1e to %.1e" % (pre_id, post_id, dw, connection_matrix[pre_id, post_id] + dw)
        connection_matrix[pre_id, post_id] += dw
        bias[post_id] = bias.max()
        
        ids_to_save = []
        if (save_all):
            ids_to_save = []

        if (save_all):
            # save
            output_fn = params['weights_fn_base'] + "%d_%d.npy" % (pre_id, post_id)
            np.save(output_fn, wij)

            output_fn = params['bias_fn_base'] + "%d.npy" % (post_id)
            np.save(output_fn, bias)

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

    print "debug", params['conn_mat_ee_fn_base'] + str(sim_cnt+1) + '.npy'
    np.savetxt(params['conn_mat_ee_fn_base'] + str(sim_cnt+1) + '.npy', connection_matrix)
    print "debug", params['bias_values_fn_base'] + str(sim_cnt+1) + '.npy'
    np.savetxt(params['bias_values_fn_base'] + str(sim_cnt+1) + '.npy', bias)

    return connection_matrix, bias

