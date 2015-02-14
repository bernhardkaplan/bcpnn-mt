import numpy as np

def convert_spiketrain_to_trace(st, t_max, t_min=0., dt=0.1, spike_width=1):
    """Converts a single spike train into a binary trace
    Keyword arguments: 
    st --  spike train in the format [time, id]
    spike_width -- number of time steps (in dt) for which the trace is set to 1
    Returns a np.array with st[i] = 1 if i in st[:, 0], st[i] = 0 else.

    TODO: get t_min
    """
    n = np.int((t_max - t_min)/ dt) + spike_width
    trace = np.zeros(n)
    if len(st) > 0:
        spike_idx = st / dt
        idx = (spike_idx - t_min / dt).astype(np.int)
        trace[idx] = 1
        for i in xrange(spike_width):
            trace[idx + i] = 1
    return trace


def get_spiking_weight_and_bias(pre_trace, post_trace, bcpnn_params, dt=.1, K_vec=None, w_init=0.):
    """
    Arguments:
        pre_trace, post_trace: pre-synaptic activity (0 means no spike, 1 means spike) (not spike trains!)
        bcpnn_params: dictionary containing all bcpnn parameters, initial value, fmax, time constants, etc
        dt -- should be the simulation time step because it influences spike_height
    """
    assert (len(pre_trace) == len(post_trace)), "Bcpnn.get_spiking_weight_and_bias: pre and post activity have different lengths!"
    if K_vec != None:
        assert (len(K_vec) == len(pre_trace)), "Bcpnn.get_spiking_weight_and_bias: pre-trace and Kappa-Vector have different lengths!\nlen pre_trace %d K_vec %d" % \
                (len(pre_trace), len(K_vec))

    initial_value = bcpnn_params['p_i']
    n = len(pre_trace)
    si = pre_trace      # spiking activity (spikes have a width and a height)
    sj = post_trace

#    zi = np.ones(n) * 0.01
#    zj = np.ones(n) * 0.01
#    eij = np.ones(n) * 0.0001
#    ei = np.ones(n) * 0.01
#    ej = np.ones(n) * 0.01
#    pi = np.ones(n) * 0.01
#    pj = np.ones(n) * 0.01

    zi = np.ones(n) * initial_value
    zj = np.ones(n) * initial_value
    eij = np.ones(n) * initial_value**2
    ei = np.ones(n) * initial_value
    ej = np.ones(n) * initial_value
    pi = np.ones(n) * initial_value
    pj = np.ones(n) * initial_value
    pij = np.ones(n) * initial_value**2 
    wij = np.ones(n)  * w_init #np.log(pij[0] / (pi[0] * pj[0]))
    bias = np.ones(n) * np.log(initial_value)
    spike_height = 1000. / (bcpnn_params['fmax'] * dt)
#    spike_height = 1000. / bcpnn_params['fmax']
#    eps = bcpnn_params['epsilon']
    eps = 0.001
    K = bcpnn_params['K']
    gain = bcpnn_params['gain']
    if K_vec == None:
        K_vec = np.ones(n) * K

    for i in xrange(1, n):
#        print 'debug', K_vec[i]
        # pre-synaptic trace zi follows si
        dzi = dt * (si[i] * spike_height - zi[i-1] + eps) / bcpnn_params['tau_i']
        zi[i] = zi[i-1] + dzi

        # post-synaptic trace zj follows sj
        dzj = dt * (sj[i] * spike_height - zj[i-1] + eps) / bcpnn_params['tau_j']
        zj[i] = zj[i-1] + dzj

        # pre-synaptic trace zi follows zi
        dei = dt * (zi[i] - ei[i-1]) / bcpnn_params['tau_e']
        ei[i] = ei[i-1] + dei

        # post-synaptic trace ej follows zj
        dej = dt * (zj[i] - ej[i-1]) / bcpnn_params['tau_e']
        ej[i] = ej[i-1] + dej

        # joint eij follows zi * zj
        deij = dt * (zi[i] * zj[i] - eij[i-1]) / bcpnn_params['tau_e']
        eij[i] = eij[i-1] + deij

        # pre-synaptic probability pi follows zi
        dpi = dt * K_vec[i] * (ei[i] - pi[i-1]) / bcpnn_params['tau_p']
        pi[i] = pi[i-1] + dpi

        # post-synaptic probability pj follows ej
        dpj = dt * K_vec[i] * (ej[i] - pj[i-1]) / bcpnn_params['tau_p']
        pj[i] = pj[i-1] + dpj

        # joint probability pij follows e_ij
        dpij = dt * K_vec[i] * (eij[i] - pij[i-1]) / bcpnn_params['tau_p']
        pij[i] = pij[i-1] + dpij
    
    print 'DEBUG BCPNN K_vec', K_vec, K_vec.mean(), '+-', K_vec.std()

    # weights
    wij = gain * np.log(pij / (pi * pj))

    # bias
    bias = gain * np.log(pj)

    return [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj]



def compute_traces(si, tau_z=10, tau_e=100, tau_p=1000, dt=1., eps=1e-6, initial_value=None, K_vec=None):
    n = si.size
    if initial_value == None:
        initial_value = (0.01, 0.01, 0.01)

    zi = np.ones(n) * initial_value[0]
    ei = np.ones(n) * initial_value[1]
    pi = np.ones(n) * initial_value[2]
    if K_vec == None:
        K_vec = np.ones(n)
    for i in xrange(1, n):
        dzi = dt * (si[i] - zi[i-1] + eps) / tau_z
        zi[i] = zi[i-1] + dzi

        # pre-synaptic trace zi follows zi
        dei = dt * (zi[i] - ei[i-1]) / tau_e
        ei[i] = ei[i-1] + dei

        # pre-synaptic probability pi follows zi
        dpi = dt * (ei[i] - pi[i-1]) / tau_p
        pi[i] = pi[i-1] * K_vec[i] + dpi

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


def compute_traces_new(si, z, e, p, tau_z=10, tau_e=100, tau_p=1000, dt=1., eps=1e-6):
    n = si.size
    for i in xrange(1, n):
        dz = dt * (si[i] - z[i-1] + eps) / tau_z
        z[i] = z[i-1] + dz

        # pre-synaptic trace z follows z
        de = dt * (z[i] - e[i-1]) / tau_e
        e[i] = e[i-1] + de

        # pre-synaptic probability p follows z
        dp = dt * (e[i] - p[i-1]) / tau_p
        p[i] = p[i-1] + dp


def compute_pij_new(zi, zj, pi, pj, eij, pij, wij, bias, tau_eij, tau_pij, get_traces=False, dt=1.):
    n = zi.size
    for i in xrange(1, n):
        # joint 
        deij = dt * (zi[i] * zj[i] - eij[i-1]) / tau_eij
        eij[i] = eij[i-1] + deij

        # joint probability pij follows zi * zj
        dpij = dt * (eij[i] - pij[i-1]) / tau_pij
        pij[i] = pij[i-1] + dpij

        # weights
        wij[i] = pij[i] / (pi[i] * pj[i])

        # bias
        bias[i] = np.log(pj[i])

    if (get_traces):
        return wij, bias, pij, eij
    else:
        return pij[-1], wij[-1], bias[-1]


def compute_bcpnn_in_place(st_pre, st_post, tau_dict, dt, fmax, tmax, save_interval):
    """
    Instead of operating on possibly long initially empty traces,
    filling them with values and returning them to store them to disk,
    this function computes the bcpnn traces iteratively and stores 
    values only periodically at a certain interval.
    Keyword arguments:
    st_pre -- array or list storing the presynaptic spikes
    st_post -- array or list storing the postsynaptic spikes
    tau_dict -- dictionary containing the time constants for the bcpnn learning rule
    dt -- time step for computing the traces
    fmax -- assumed maximimum rate for bcpnn spike trace, determines height of the z-traces
    save_interval -- time step interval in ms at which the values are stored
    """

    if type(st_pre) != type(np.array(1)):
        st_pre = np.array(st_pre)
    if type(st_post) != type(np.array(1)):
        st_post = np.array(st_post)
    n_intervals = tmax / save_interval
    n_steps_per_interval = int(save_interval / dt)

    spike_width = 1. / dt
    eps = dt / tau_dict['tau_pi']
    initial_value = eps
    # trace buffers for saving
    zi = np.ones(n_intervals) * initial_value
    zj = np.ones(n_intervals) * initial_value
    eij = np.ones(n_intervals) * initial_value**2
    ei = np.ones(n_intervals) * initial_value
    ej = np.ones(n_intervals) * initial_value
    pi = np.ones(n_intervals) * initial_value
    pj = np.ones(n_intervals) * initial_value
    pij = np.ones(n_intervals) * initial_value**2
    wij = np.ones(n_intervals)  *  np.log(pij[0] / (pi[0] * pj[0]))
    bias = np.ones(n_intervals) * np.log(initial_value)
    spike_height = 1000. / fmax
    
    # trace buffers for computation
    interval_si = np.zeros(n_steps_per_interval) # the spike traces created from the spike train, pre
    interval_sj = np.zeros(n_steps_per_interval) # post
    interval_zi = np.ones(n_steps_per_interval) * initial_value
    interval_zj = np.ones(n_steps_per_interval) * initial_value
    interval_eij = np.ones(n_steps_per_interval) * initial_value**2
    interval_ei = np.ones(n_steps_per_interval) * initial_value
    interval_ej = np.ones(n_steps_per_interval) * initial_value
    interval_pi = np.ones(n_steps_per_interval) * initial_value
    interval_pj = np.ones(n_steps_per_interval) * initial_value
    interval_pij = np.ones(n_steps_per_interval) * initial_value**2
    interval_wij = np.ones(n_steps_per_interval)  *  np.log(pij[0] / (pi[0] * pj[0]))
    interval_bias = np.ones(n_steps_per_interval) * np.log(initial_value)

    # Spike train:
    #  |         |   |
    #  t1       t2  t3
    # Spike traces (the z-traces converge to these):
    #  ||||      |||||||||
    # Intervals for storing the traces
    # +     +     +     + 
    for i_interval in xrange(0, n_intervals):
        # create the spike trace from the spike train
        # the spike trace is a binary trace (either 0 or spike_height)
        # the spike trace has for spike_width / dt steps the value 1000. / fmax

        t_interval_start = n_steps_per_interval * dt
        t_interval_stop = (n_steps_per_interval + 1) * dt
        pre_spikes_in_interval = st_pre[(st_pre > t_interval_start) == (st_pre < t_interval_stop)]
        post_spikes_in_interval = st_post[(st_post > t_interval_start) == (st_post < t_interval_stop)]
        if pre_spikes_in_intervals.size == 0:
            pass
            # Since nothinh happens on the pre-synaptic side, 
            # compute exact solution for the traces
#        for i_ in xrange(1, n_steps_per_interval):

    for i in xrange(1, n):
        # pre-synaptic trace zi follows si
        dzi = dt * (si[i] * spike_height - zi[i-1] + eps) / tau_dict['tau_zi']
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



