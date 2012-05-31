import numpy as np
import numpy.random as rnd
import pyNN
import pyNN.random
import utils
from scipy.spatial import distance

def get_p_conn(tuning_prop, src, tgt, sigma_x, sigma_v):

    x0 = tuning_prop[src, 0]
    y0 = tuning_prop[src, 1]
    u0 = tuning_prop[src, 2]
    v0 = tuning_prop[src, 3]
    x1 = tuning_prop[tgt, 0]
    y1 = tuning_prop[tgt, 1]
    u1 = tuning_prop[tgt, 2]
    v1 = tuning_prop[tgt, 3]
    dx = utils.torus_distance(x0, x1)
    dy = utils.torus_distance(x0, x1)
    latency = np.sqrt(dx**2 + dy**2) / np.sqrt(u0**2 + v0**2)
    x_predicted = x0 + u0 * latency  
    y_predicted = y0 + v0 * latency  
    p = .5 * np.exp(-((utils.torus_distance(x_predicted, x1))**2 + (utils.torus_distance(y_predicted, y1))**2 / (2 * sigma_x**2))) \
            * np.exp(-((u0-u1)**2 + (v0 - v1)**2) / (2 * sigma_v**2))
    return p, latency

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
    p_thresh = params['p_thresh_connection']
    conn_list = []
#    output = np.zeros((n_cells ** 2 - n_cells, 4))
    output = ""
    p_output = np.zeros((n_cells**2 - n_cells, 4))
    p_above_threshold = np.zeros(n_cells**2 - n_cells)
#    p_output = np.zeros((n_cells**2 - n_cells, 5))


    (delay_min, delay_max) = params['delay_range']
    i = 0
    for src in xrange(n_cells):
        for tgt in xrange(n_cells):
            if (src != tgt):
                p, latency = get_p_conn(tuning_prop, src, tgt, sigma_x, sigma_v)

                delay = min(max(latency * params['delay_scale'], delay_min), delay_max)
                p_output[i, 0], p_output[i, 1], p_output[i, 2], p_output[i, 3] = src, tgt, p, delay
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
    w_output = np.zeros((valid_conns.size, 4))
    w_output[:, 0] = p_output[valid_conns, 0]
    w_output[:, 1] = p_output[valid_conns, 1]
    w_output[:, 2] = p_output[valid_conns, 2] * p_to_w_scaling
    w_output[:, 3] = p_output[valid_conns, 3]

    output_fn = params['conn_list_ee_fn_base'] + '0.dat'
    print "Saving to file ... ", output_fn
    np.savetxt(output_fn, w_output, fmt='%d\t%d\t%.4e\t%.1e')
#    output_fn = params['conn_list_ee_fn_base'] + 'probabilities.dat'
#    print "Saving to file ... ", output_fn
#    np.savetxt(output_fn, p_output, fmt='%d\t%d\t%.4e\t%.1e')

    return 0

#    np.savetxt(output_fn, np.array(conn_list))
#    return weight_matrix, latency_matrix




def compute_random_weight_list(input_fn, output_fn, seed=98765):
    """
    Open the existing pre-computed (non-random) conn_list and shuffle sources and targets
    """
    rnd.seed(seed)
    d = np.loadtxt(input_fn)
    rnd.shuffle(d[:, 0])    # shuffle source ids
    rnd.shuffle(d[:, 1])    # shuffle target ids
#    rnd.shuffle(d[:, 3])    # shuffle delays ids

    np.savetxt(output_fn, d)


def create_initial_connection_matrix(n, output_fn, w_max=1.0, sparseness=0.0, seed=1234):
    """
    Returns an initial connection n x n matrix with a sparseness parameter.
    Sparseness is betwee 0 and 1.
    if sparseness == 0: return full matrix
    if sparseness != 0: set sparseness * n * m elements to 0
    if sparseness == 1: return empty matrix
    """
    rnd.seed(seed)
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



def create_conn_list_by_random_normal(output_fn, sources, targets, p, w_mean, w_sigma, d_mean=0., d_sigma=0., d_min=1):
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


#class MyDistr(pyNN.random.RandomDistribution):
#    self.rng = ....


class MyRNG(pyNN.random.WrappedRNG):
    def __init__(self):
        self.rng
    def _next(self, distribution, n, parameters):
        return self.draw(n, parameters)

    def draw(self, n, p):
        d = np.zeros(n)
        for i in xrange(n):
            d[i] = self.rnd_distr(p)

    def rnd_distr(self):
        return p[0] * np.random.rand * np.random.exponential( - np.random.rand / p[1])
                

            

