import numpy as np
import numpy.random as rnd
from scipy.spatial import distance

def compute_weights_from_tuning_prop(tuning_prop, params):
    """
    Arguments:
        tuning_prop: 2 dimensional array with shape (n_cells, 4)
            tp[:, 0] : x-position
            tp[:, 1] : y-position
            tp[:, 2] : u-position (speed in x-direction)
            tp[:, 3] : v-position (speed in y-direction)
    """

    n_cells = tuning_prop[:, 0].size
    sigma_x, sigma_v = params['w_sigma_x'], params['w_sigma_v'] # small sigma values let p and w shrink
    p_to_w_scaling = params['p_to_w_scaling']
    p_thresh = params['p_connection_thresh']
    conn_list = []
#    output = np.zeros((n_cells ** 2 - n_cells, 4))
    output = ""
    p_output = np.zeros((n_cells**2 - n_cells, 4))
    p_above_threshold = np.zeros(n_cells**2 - n_cells)
#    p_output = np.zeros((n_cells**2 - n_cells, 5))


    i = 0
    for src in xrange(n_cells):
        for tgt in xrange(n_cells):
            if (src != tgt):
                x0 = tuning_prop[src, 0]
                y0 = tuning_prop[src, 1]
                u0 = tuning_prop[src, 2]
                v0 = tuning_prop[src, 3]
                x1 = tuning_prop[tgt, 0]
                y1 = tuning_prop[tgt, 1]
                u1 = tuning_prop[tgt, 2]
                v1 = tuning_prop[tgt, 3]

                latency = np.sqrt((x0 - x1)**2 + (y0 - y1)**2) / np.sqrt(u0**2 + v0**2)
                p = .5 * np.exp(-((x0 + u0 * latency - x1)**2 + (y0 + v0 * latency - y1)**2) / (2 * sigma_x**2)) \
                        * np.exp(-((u0-u1)**2 + (v0 - v1)**2) / (2 * sigma_v**2))

                p_output[i, 0], p_output[i, 1], p_output[i, 2], p_output[i, 3] = src, tgt, p, latency
                # decide which connection should be discarded
                if p >= p_thresh:
                    p_above_threshold[i] = p
                i += 1
        
#                if (p*params['p_to_w_scaling'] >= params['w_init_thresh']):
#                    w = p * p_to_w_scaling
#                    delay = min(max(latency * params['delay_scale'], params['delay_min']), params['delay_max'])
#                    output += "%d\t%d\t%.4e\t%.1e\n" % (src, tgt, w, delay)
#                    conn_list.append([src, tgt, w, delay])

    # for curiosity print also the probabilities
#    print "Calculating normalized probabilities"
#    buff = np.zeros((n_cells**2 - n_cells, 4))
#    buff[:,:3] = p_output[:,:3]
#    buff[:, 3] = p_output[:, 2] / p_output[:, 2].sum()
#    output_fn = params['conn_prob_fn']
#    np.savetxt(output_fn, buff, fmt='%d\t%d\t%.4e\t%.4e')

    # convert thresholded probabilities to weight
    valid_conns = p_above_threshold.nonzero()[0]
    w_output = np.zeros((valid_conns .size, 4))
    w_output[:, 0] = p_output[valid_conns, 0]
    w_output[:, 1] = p_output[valid_conns, 1]
    w_output[:, 2] = p_output[valid_conns, 2] * p_to_w_scaling
    w_output[:, 3] = p_output[valid_conns, 3]

    output_fn = params['conn_list_ee_fn_base'] + '0.dat'
    print "Saving to file ... ", output_fn
    np.savetxt(output_fn, w_output, fmt='%d\t%d\t%.4e\t%.1e')



#    np.savetxt(output_fn, np.array(conn_list))
#    return weight_matrix, latency_matrix

def compute_weights_from_tuning_prop_distances(tuning_prop, params):
    """
    Arguments:
        tuning_prop: 2 dimensional array with shape (n_cells, 4)
            tp[:, 0] : x-position
            tp[:, 1] : y-position
            tp[:, 2] : u-position (speed in x-direction)
            tp[:, 3] : v-position (speed in y-direction)
    """

    n_cells = tuning_prop[:, 0].size
    sigma_x, sigma_v = params['w_sigma_x'], params['w_sigma_v'] # small sigma values let p and w shrink
    p_to_w_scaling = params['p_to_w_scaling']
    n_nearest_neighbors = int(min(1000, params['n_exc'] * .25))
    distance_matrix = np.zeros((n_cells, n_cells))
    closest_neighbors = np.zeros((n_cells, n_nearest_neighbors)) # stores gids of n_nearest neighbors
    output = ""
    n_discarded_conn = 0

    print "computing distance matrix..."
    for i in xrange(n_cells):
        for j in xrange(i, n_cells):
            distance_matrix[i, j] = distance.euclidean(tuning_prop[i, :], tuning_prop[j, :])
            distance_matrix[j, i] = distance_matrix[i, j]
        closest_neighbors[i, :] = np.argsort(distance_matrix[i, :])[1:n_nearest_neighbors+1]
            
    for src in xrange(n_cells):
        for j in xrange(n_nearest_neighbors):
#            print "debug i %d j %d n_nn %d closest_n.shape:" % (src, j, n_nearest_neighbors), closest_neighbors.shape
            tgt = closest_neighbors[src, j]
            assert (src != tgt)
            x0 = tuning_prop[src, 0]
            y0 = tuning_prop[src, 1]
            u0 = tuning_prop[src, 2]
            v0 = tuning_prop[src, 3]
            x1 = tuning_prop[tgt, 0]
            y1 = tuning_prop[tgt, 1]
            u1 = tuning_prop[tgt, 2]
            v1 = tuning_prop[tgt, 3]

            latency = np.sqrt((x0 - x1)**2 + (y0 - y1)**2) / np.sqrt(u0**2 + v0**2)
            p = .5 * np.exp(-((x0 + u0 * latency - x1)**2 + (y0 + v0 * latency - y1)**2) / (2 * sigma_x**2)) \
                    * np.exp(-((u0-u1)**2 + (v0 - v1)**2) / (2 * sigma_v**2))

#            p_output[i, 0], p_output[i, 1], p_output[i, 2] = src, tgt, p
            i += 1
            # convert probability to weight
            if (p*params['p_to_w_scaling'] >= params['w_init_thresh']):
                w = p * p_to_w_scaling
                delay = min(max(latency * params['delay_scale'], params['delay_min']), params['delay_max'])
                output += "%d\t%d\t%.4e\t%.1e\n" % (src, tgt, w, delay)

                n_discarded_conn += 1


    print "n_discarded_conn: ", n_discarded_conn
    output_fn = params['conn_list_ee_fn_base'] + '0.dat'
    f = open(output_fn, 'w')
    f.write(output)
    f.close()

    # for curiosity print also the probabilities
    output_fn = params['conn_prob_fn']
#    np.savetxt(output_fn, p_output, fmt='%d\t%d\t%.4e')


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



