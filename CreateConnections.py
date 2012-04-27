import numpy as np
import numpy.random as rnd

def create_initial_connection_matrix(n, output_fn, sparseness=0.1):
    """
    Returns an initial connection n x n matrix with a sparseness parameter.
    Sparseness is betwee 0 and 1.
    if sparseness = 0: return full matrix
    if sparseness != 0: set sparseness * n * m elements to 0
    """
    d = rnd.rand(n, n)
    # set the diagonal elements to zero
    for i in xrange(n):
        d[i, i] = 0

    if (sparseness != 0.0):
        z = (sparseness * n**2 - n) # ignore the diagonal
        cnt = 0
        while cnt <= z:
            d[rnd.randint(0, n), rnd.randint(0, n)] = 0.
            cnt += 1
    np.save(output_fn, d)
    return d

#def connect_by_random(pre_list, post_list, p_conn, weight=0.0):


