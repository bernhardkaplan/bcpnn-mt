import sys
import os
import numpy as np
import utils
import simulation_parameters

class CheckConnections(object):

    def __init__(self, params):
        self.params = params

    def count_neurons_without_outgoing_connections(self, conn_type='ee'):

        fn = self.params['merged_conn_list_%s' % conn_type]
        if not os.path.exists(fn):
            print 'Merging connections for ', conn_type
            os.system('python merge_connlists.py %s' % self.params['params_fn_json'])
        print 'Loading connections from:', fn
        d = np.loadtxt(fn)

        (n_src, n_tgt, syn_type) = utils.resolve_src_tgt(conn_type, self.params)

        # No offse is needed, becase cells are stored NOT by GID in conn_list, but by the index within the population
        n_without_tgt = 0
        for i_src in xrange(n_src):
            tgts_ = utils.get_targets(d, i_src)
            n_tgt = tgts_.shape[0]
            if n_tgt == 0:
                n_without_tgt += 1
        print 'Number of cells without outgoing %s connections: %d' % (conn_type, n_without_tgt)







if __name__ == '__main__':
    if len(sys.argv) > 1:
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn):
            param_fn += '/Parameters/simulation_parameters.json'

        import json
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
    else:
        print '\nUsing the default parameters give in simulation_parameters.py\n'
        network_params = simulation_parameters.parameter_storage()
        params = network_params.params

    CC = CheckConnections(params)
    CC.count_neurons_without_outgoing_connections(conn_type='ee')
