import numpy as np
import numpy.random as rnd

def create_initial_connection_matrix(n, output_fn, w_max=1.0, sparseness=0.0):
    """
    Returns an initial connection n x n matrix with a sparseness parameter.
    Sparseness is betwee 0 and 1.
    if sparseness == 0: return full matrix
    if sparseness != 0: set sparseness * n * m elements to 0
    if sparseness == 1: return empty matrix
    """
    d = rnd.rand(n, n) * w_max
    print "debug", d[0, 1], d[0, 3]
    # set the diagonal elements to zero
    for i in xrange(n):
        d[i, i] = 0

    if (sparseness != 0.0):
        z = (sparseness * n**2 - n) # ignore the diagonal
        cnt = 0
        while cnt <= z:
            d[rnd.randint(0, n), rnd.randint(0, n)] = 0.
            cnt += 1
    np.savetxt(output_fn, d)
    return d



def create_conn_list_by_random(output_fn, sources, targets, p, w_mean, w_sigma, d_mean=0., d_sigma=0., d_min=1):
    """
    This function writes a conn list with normal distributed weights (and delays) to output_fn.
    Arguments:
        output_fn: string
        sources: tuple with min_gid and max_gid
    """
    conns = []
    for src in xrange(sources[0], sources[1]):
        for tgt in xrange(targets[0], targets[1]):
            if rnd.rand() <= p:
                w = -1
                while (w < 0):
                    w = rnd.normal(w_mean, w_sigma)
                if d_sigma != 0.:
                    d = -1
                    while (d <= 0):
                        d = rnd.normal(d_mean, d_sigma)
                else:
                    d = d_min
                conns.append([src, tgt, w, d])
    output_array = np.array(conns)
    np.savetxt(output_fn, output_array)



def create_connections_between_cells(params, conn_mat_fn):
    """
    This function writes 
    Arguments:
        params : parameter dictionary (see: simulation_parameters.py)
        conn_mat_fn : filename for the minicolumn - minicolumn connectivity
    """
    for src_mc in xrange(params['n_mc']):
        for tgt_mc in xrange(params['n_mc']):
            pass



def compute_weights_from_tuning_prop(tuning_prop, motion_params):
    """
    Arguments:
        tuning_prop: 2 dimensional array with shape (n_cells, 4)
            tp[:, 0] : x-position
            tp[:, 1] : y-position
            tp[:, 2] : u-position (speed in x-direction)
            tp[:, 3] : v-position (speed in y-direction)
    """

    n_cells = tuning_prop[:, 0].size
    (x0_stim, y0_stim, u_stim, v_stim) = motion

    sigma_x, sigma_v = 1., 1. # tuning parameters , TODO: how to handle these

    weight_matrix = np.zeros((n_cells, n_cells))
    latency_matrix = np.zeros((n_cells, n_cells))
    p_to_w_scaling = 1.

    for cell0 in xrange(n_cells):
        for cell1 in xrange(n_cells):
            if cell0 != cell1:
                x0 = tuning_prop[cell0, 0]
                y0 = tuning_prop[cell0, 1]
                u0 = tuning_prop[cell0, 2]
                v0 = tuning_prop[cell0, 3]
                x1 = tuning_prop[cell1, 0]
                y1 = tuning_prop[cell1, 1]
                u1 = tuning_prop[cell1, 2]
                v1 = tuning_prop[cell1, 3]

                latency = np.sqrt((x0 - x1)**2 + (y0 - y1)**2) / np.sqrt(u_stim**2 + v_stim**2)
                p = .5 * np.exp(- ((x0 + u_stim * latency - x1)**2 + (y + v_stim * latency - y1)**2)/ (2 * sigma_x**2) \
                        / np.exp(-((u0-u1)**2 + (v0 - v1)**2) / (2 * sigma_v**2))

                # convert probability to weight
                weight_matrix[cell1, cell2] = p * p_to_w_scaling
                latency_matric[cell1, cell2] = latency

    return weight_matrix, latency_matrix

