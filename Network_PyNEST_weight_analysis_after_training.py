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
                
        
    def plot_weights(self):

        fns = self.get_filenames()

        for fn in fns:
            f = file(fn, 'r')
            d = json.load(f)
            tgt_gids = d.keys()
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
    WA.plot_weights()
