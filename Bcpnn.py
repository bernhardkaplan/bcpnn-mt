import numpy as np

def get_abstract_weight_and_bias(pre, post, alpha=0.01, dt=1, eps=1e-6):
    """
    Arguments:
        pre, post: abstract activity patterns, valued between 0 and 1
        alpha: learning rate
        dt: integration time step
    """

    assert (len(pre) == len(post)), "Abstract pre and post activity have different lengths!"
    n = len(pre)
    pi = np.zeros(n)
    pj = np.zeros(n)
    pij = np.zeros(n)
    bias = np.zeros(n)
    wij = np.zeros(n)
    pre_post = np.array(pre) * np.array(post)

    for i in xrange(1, n):
        # pre
        dpi = alpha * dt * (pre[i-1] - pi[i-1])
        pi[i] = pi[i-1] + dpi
        # post
        dpj = alpha * dt * (post[i-1] - pj[i-1])
        pj[i] = pj[i-1] + dpj
        # joint
        dpij = alpha * dt * (pre_post[i-1] - pij[i-1])
        pij[i] = pij[i-1] + dpij

        if ((pi[i] == 0) or (pj[i] == 0)):
            wij[i] = 0
        if (pij[i] == 0):
            wij[i] = eps**2
        else:
            wij[i] = np.log(pij[i] / (pi[i] * pj[i]))
#elif (pij[i] <= pi[i] * pj[i]): # this condition avoids weights going negative

        # bias
        if (pj[i] > 0):
            bias[i] = np.log(pj[i])
        else:
            bias[i] = np.log(eps)

    return wij, bias, pi, pj, pij


def get_spiking_weight_and_bias(pre_trace, post_trace, bin_size=1, tau_z=10, tau_e=100, tau_p=1000, dt=1, f_max=300., eps=1e-6):
    """
    Arguments:
        pre_trace, post_trace: pre-synaptic activity (0 mean no spike, 1 means spike) (not spike trains!)
        
    """
    assert (len(pre_trace) == len(post_trace)), "Abstract pre and post activity have different lengths!"

#    if bin_size != 1:
#   TODO:
#        return get_spiking_weight_and_bias_binned(pre_spikes, post_spikes, bin_size=1, tau_z=10, tau_e=100, tau_p=1000, dt=1, eps=1e-2)

    n = len(pre_trace)
    si = pre_trace      # spiking activity (spikes have a width and a height)
    sj = post_trace
    zi = np.ones(n) * eps
    zj = np.ones(n) * eps
    ei = np.ones(n) * eps
    ej = np.ones(n) * eps
    eij = np.ones(n) * eps**2
    pi = np.ones(n) * eps
    pj = np.ones(n) * eps
    pij = np.ones(n) * eps**2
    wij = np.zeros(n)
    bias = np.ones(n) * np.log(eps)
    spike_height = 1000. / f_max

    print "Integrating traces"
    for i in xrange(1, n):
        # pre-synaptic trace zi follows si
        dzi = dt * (si[i-1] * spike_height- zi[i-1] + eps) / tau_z
        zi[i] = zi[i-1] + dzi

        # post-synaptic trace zj follows sj
        dzj = dt * (sj[i-1] * spike_height- zj[i-1] + eps) / tau_z
        zj[i] = zj[i-1] + dzj

        # pre-synaptic trace zi follows zi
        dei = dt * (zi[i-1] - ei[i-1] + eps) / tau_e
        ei[i] = ei[i-1] + dei

        # post-synaptic trace ej follows zj
        dej = dt * (zj[i-1] - ej[i-1] + eps) / tau_e
        ej[i] = ej[i-1] + dej

        # joint 
        deij = dt * (zi[i-1] * zj[i-1] - eij[i-1] + eps**2) / tau_e
        eij[i] = eij[i-1] + deij

        # pre-synaptic probability pi follows zi
        dpi = dt * (ei[i-1] - pi[i-1] + eps) / tau_p
        pi[i] = pi[i-1] + dpi

        # post-synaptic probability pj follows ej
        dpj = dt * (ej[i-1] - pj[i-1] + eps) / tau_p
        pj[i] = pj[i-1] + dpj

        # joint probability pij follows zi * zj
        dpij = dt * (eij[i-1] - pij[i-1] + eps**2) / tau_p
        pij[i] = pij[i-1] + dpij

        # weights
        if ((pi[i] <= eps) or (pj[i] <= eps) or (pij[i] <= eps**2)):
            wij[i] = 0
#elif (pij[i] <= pi[i] * pj[i]): # this condition avoids weights going negative
        else:
            wij[i] = np.log(pij[i] / (pi[i] * pj[i]))

        # bias
        if (pj[i] > 0):
            bias[i] = np.log(pj[i])
        else:
            bias[i] = np.log(eps)
    return wij, bias, pi, pj, pij, ei, ej, eij, zi, zj

def convert_spiketrain_to_trace(st, n):
    """
    st: spike train in the format [time, id]
    n : size of the trace to be returned
    To be used with spike train inputs.
    Returns a np.array with st[i] = 1 if i in st[:, 0], st[i] = 0 else.
    """
    trace = np.zeros(n)
    for i in st:
        trace[int(round(i))] = 1
    return trace
