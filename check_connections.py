import sys
import os
import numpy as np
import utils
import simulation_parameters

class CheckConnections(object):

    def __init__(self, params):
        self.params = params
        self.conn_data = {}
        self.conn_data_loaded = {'ee': False, 'ei': False, 'ie': False, 'ii':False}

    def get_conn_data(self, conn_type):
        if self.conn_data_loaded[conn_type] == False:
            fn = self.params['merged_conn_list_%s' % conn_type]
            if not os.path.exists(fn):
                print 'Merging connections for ', conn_type
                os.system('python merge_connlists.py %s' % self.params['params_fn_json'])
            print 'Loading connections from:', fn
            d = np.loadtxt(fn)
            self.conn_data[conn_type] = d
            self.conn_data_loaded[conn_type] = True
        else:
            d = self.conn_data[conn_type]
        return d

    def count_neurons_without_outgoing_connections(self, conn_type='ee'):

        (n_src, n_tgt, syn_type) = utils.resolve_src_tgt(conn_type, self.params)
        d = self.get_conn_data(conn_type)

        # No offse is needed, becase cells are stored NOT by GID in conn_list, but by the index within the population
        n_without_tgt = 0
        for i_src in xrange(n_src):
            tgts_ = utils.get_targets(d, i_src)
            n_tgt = tgts_.shape[0]
            if n_tgt == 0:
                n_without_tgt += 1
        print 'Number of cells without outgoing %s connections: %d' % (conn_type, n_without_tgt)



    def check_anisotropy(self, conn_type='ee'):
        """
        Calculates the mean difference between outward connections and source cell position.
        """

        d = self.get_conn_data(conn_type)
        (n_src, n_tgt, tp_src, tp_tgt) = utils.resolve_src_tgt_with_tp(conn_type, self.params)
        mean_diff = np.zeros(n_src)
        np.random.seed(0)
        for i_src in xrange(n_src):
            src_gid = np.random.randint(0, n_src)
            connections = utils.get_targets(d, src_gid)
            connection_gids = connections[:, 1].astype(int)
            weights = connections[:, 2]
            cms = utils.get_connection_center_of_mass(connection_gids, weights, tp_tgt)
#            diff_x = utils.torus_distance(tp_src[src_gid, 0] - cms[0])
            diff_x = tp_src[src_gid, 0] - cms[0]
            diff_x *= np.sign(tp_src[src_gid, 2])
            # if negative --> wrong direction of connectivity, diff_x will be negative
            mean_diff[src_gid] = diff_x
            print 'src_gid %d\tx_src = %.3f\tcms_x = %.3f\tvx = %.3f\tx - cms[0] = %.3f' % (src_gid, tp_src[src_gid, 0], cms[0], tp_src[src_gid, 2], diff_x)
        print 'mean difference between source cell position and cms_x = %.2e +- %.2e' % (mean_diff.mean(), mean_diff.std())



    def get_realized_connections(self, conn_type='ee'):
        d = self.get_conn_data(conn_type)
        (n_src, n_tgt, tp_src, tp_tgt) = utils.resolve_src_tgt_with_tp(conn_type, self.params)
        n_out = np.zeros(n_src)
        for i_src in xrange(n_src):
            connections = utils.get_targets(d, i_src)
            n_conn = connections[:, 0].size
            n_out[i_src] = n_conn

        n_out_mean = n_out.mean()
        n_out_std = n_out.std()
        print 'n_out_mean:', n_out_mean, 'n_out_std:', n_out_std



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
    CC.get_realized_connections()
#    CC.count_neurons_without_outgoing_connections(conn_type='ee')
#    CC.check_anisotropy(conn_type='ee')
