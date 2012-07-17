import numpy as np
import numpy.random as rnd
import pyNN
import pyNN.random
import utils
from scipy.spatial import distance
import os
import time

def get_p_conn(tuning_prop, src, tgt, w_sigma_x, w_sigma_v):

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
    p = np.exp(-((utils.torus_distance(x_predicted, x1))**2 + (utils.torus_distance(y_predicted, y1))**2 / (2 * w_sigma_x**2))) \
            * np.exp(-((u0-u1)**2 + (v0 - v1)**2) / (2 * w_sigma_v**2))
    return p, latency

def compute_weights_from_tuning_prop(tuning_prop, params, comm=None):
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
    (delay_min, delay_max) = params['delay_range']
    if comm != None:
        pc_id, n_proc = comm.rank, comm.size
        comm.barrier()
    else:
        pc_id, n_proc = 0, 1
    output_fn = params['conn_list_ee_fn_base'] + 'pid%d.dat' % (pc_id)
    print "Proc %d computes initial weights to file %s" % (pc_id, output_fn)
    gid_min, gid_max = utils.distribute_n(params['n_exc'], n_proc, pc_id)
    n_cells = gid_max - gid_min
    my_cells = range(gid_min, gid_max)
    output = ""

    i = 0
    for src in my_cells:
        for tgt in xrange(params['n_exc']):
            if (src != tgt):
                p, latency = get_p_conn(tuning_prop, src, tgt, sigma_x, sigma_v)
                delay = min(max(latency * params['delay_scale'], delay_min), delay_max)  # map the delay into the valid range
                output += '%d\t%d\t%.4e\t%.2e\n' % (src, tgt, p, delay)
                i += 1

    print 'Process %d writes connections to: %s' % (pc_id, output_fn)
    f = file(output_fn, 'w')
    f.write(output)
    f.flush()
    f.close()
    normalize_probabilities(params, comm, params['w_thresh_connection'])
    if (comm != None):
        comm.barrier()


def normalize_probabilities(params, comm, w_thresh=None):
    """
    Open all files named with params['conn_list_ee_fn_base'], sum the last 3rd column storing the probabilities,
    reopen all the files, divide by the sum, and save all the files again.
    
    """
    if comm != None:
        pc_id, n_proc = comm.rank, comm.size
    else:
        pc_id, n_proc = 0, 1

    fn = params['conn_list_ee_fn_base'] + 'pid%d.dat' % (pc_id)
    print 'Reading', fn
    d = np.loadtxt(fn)


    p_sum = d[:, 2].sum()
    if comm != None:
        p_sum_global = comm.allreduce(p_sum, None)
    else:
        p_sum_global = p_sum

    d[:, 2] /= p_sum_global
    d[:, 2] *= params['p_to_w_scaling']

    if w_thresh != None:
        indices = np.nonzero(d[:, 2] > w_thresh)[0]
        d = d[indices, :] # resize the output array

    fn = params['conn_list_ee_fn_base'] + 'pid%d_normalized.dat' % (pc_id)
    print 'Writing to ', fn
    np.savetxt(fn, d, fmt='%d\t%d\t%.4e\t%.2e')
    time.sleep(3)

    # make one file out of many
#    if pc_id == 0:
#        data = ''
#        for pid in xrange(n_proc):
#            fn = params['conn_list_ee_fn_base'] + 'pid%d_normalized.dat ' % (pid)
#            f = file(fn, 'r')
#            data += f.readlines()
#        output_fn = params['conn_list_ee_fn_base'] + '0.dat'
#        f = file(output_fn, 'w')
#        f.write(data)
#        f.flush()
#        f.close()

    if pc_id == 0:
        cat_command = 'cat '
        for pid in xrange(n_proc):
            fn = params['conn_list_ee_fn_base'] + 'pid%d_normalized.dat ' % (pid)
            cat_command += fn
        output_fn = params['conn_list_ee_fn_base'] + '0.dat'
        cat_command += ' > %s' % output_fn
        print 'Merging to:', output_fn
        os.system(cat_command)
        time.sleep(2)



def compute_random_weight_list(input_fn, output_fn, params, seed=98765):
    """
    Open the existing pre-computed (non-random) conn_list and shuffle sources and targets
    """
    rnd.seed(seed)
    d = np.loadtxt(input_fn)
    d[:, 0] = rnd.randint(0, params['n_exc'], d[:, 0].size)
    d[:, 1] = rnd.randint(0, params['n_exc'], d[:, 0].size)
#    rnd.shuffle(d[:, 0])    # shuffle source ids
#    rnd.shuffle(d[:, 1])    # shuffle target ids
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
                

            

