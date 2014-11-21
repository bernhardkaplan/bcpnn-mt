import numpy as np
import numpy.random as nprnd
import sys
#import NeuroTools.parameters as ntp
import os
import utils
import nest
import json
import simulation_parameters

class Converter(object):
    def __init__(self, params):
        self.params = params
        self.load_tuning_prop()

    def load_tuning_prop(self):
        fn = self.params['tuning_prop_exc_fn']
        print 'Loading tuning properties from:', fn 
        self.tp_exc = np.loadtxt(fn)

    def get_delay_1D(self, src_gid, tgt_gid):
        xi, yi, ui, vi, theta_i = self.tp_exc[src_gid - 1, :]
        xj, yj, uj, vj, theta_j = self.tp_exc[tgt_gid - 1, :]
        d_ij = (xi - xj)
        tau_ij = np.abs(d_ij) / np.abs(ui)
        return tau_ij





    def convert_adj_list_to_fns(self, fn_list, add_delays=False):
        n = len(fn_list)
        output_txt = ''
        for i_, fn in enumerate(fn_list):
            print 'File %d / %d' % (i_, n)
            f = file(fn, 'r')
            adj_list = json.load(f)
            tgt_gids = adj_list.keys()
            for tgt in tgt_gids:
                for (src, w) in adj_list[tgt]:
                    if add_delays:
                        delay = self.get_delay_1D(int(src), int(tgt))
                        output_txt += '%d\t%d\t%.6e\t%.6e\n' % (int(src) - 1, int(tgt) - 1, w, delay)
                    else:
                        output_txt += '%d\t%d\t%.6e\n' % (int(src) - 1, int(tgt) - 1, w)
        output_fn = params['merged_conn_list_ee']
        print 'Writing to file:', output_fn
        output_file = file(output_fn, 'w')
        output_file.write(output_txt)
        output_file.close()
        print 'Done'


def main(params, add_delays):
    C = Converter(params)
    conn_folder = params['connections_folder']
    to_match = 'adj_list_tgt_index_'
    fns = utils.get_filenames(conn_folder, to_match, to_match_contains_folder=True, return_abspath=True)
    C.convert_adj_list_to_fns(fns, add_delays=add_delays)


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
        print '\nTaking the default parameters give in simulation_parameters.py\n'
        ps = simulation_parameters.parameter_storage()
        params = ps.params

    add_delays = True
    main(params, add_delays)

