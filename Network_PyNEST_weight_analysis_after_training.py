import sys
import os
import numpy as np
import re
import json



class WeightAnalyser(object):

    def __init__(self, params):
        self.params = params


    def get_filenames(self, iteration=0):
        fns = []
        folder = self.params['connections_folder']

        # find files written by different processes
        conn_mat_fn = self.params['conn_mat_fn_base'] + 'AS_(\d)_(\d).json'
        to_match = conn_mat_fn.rsplit('/')[-1]
        for fn in os.listdir(os.path.abspath(folder)):
            m = re.match(to_match, fn)
            if m:
                it_ = int(m.groups()[0])
                if it_ == iteration:
                    fn_abs_path = self.params['connections_folder'] +  fn
                    fns.append(fn_abs_path)
        print 'Found files:', fns
        return fns
                
        
    def load_weights(self):

        fns = self.get_filenames()

        self.adj_list = {}
        for fn in fns:
            f = file(fn, 'r')
            d = json.load(f)
            self.adj_list.update(d)


    def get_weights_to_cell(self, tgt_gid):

        print 'Cells projecting to tgt:', tgt_gid
        d = np.array(self.adj_list[str(tgt_gid)])
        src_gids = d[:, 0] # source gids projecting to the gid


    def get_weight(self, src_gid, tgt_gid):

        if self.adj_list.has_key(str(tgt_gid)):
            d = np.array(self.adj_list[str(tgt_gid)])
        else:
            print 'Could not find tgt_gid %d in the connection files' % (tgt_gid)
            return False
        src_gids = d[:, 0] # source gids projecting to the gid
        idx = (src_gid == src_gids).nonzero()[0]
        if len(idx) > 0:
            w = d[(src_gid == src_gids).nonzero()[0], 1]
            print 'w (%d - %d) = %f' % (src_gid, tgt_gid, w)
            return w
        else:
            print 'Could not find a projection from %d to %d in the connection files' % (src_gid, tgt_gid)
            return False
#            for tgt in tgt_gids:


    



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
        import simulation_parameters
        param_tool = simulation_parameters.parameter_storage()
        params = param_tool.params

    WA = WeightAnalyser(params)
    WA.load_weights()
#    WA.get_weights_to_cell(201)
    WA.get_weight(201, 203)
    WA.get_weight(201, 13201928)
    WA.get_weight(13201928, 201)

