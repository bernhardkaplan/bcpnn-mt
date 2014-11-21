import os
import json
import sys
import utils
import numpy as np

def get_cells_near_stim(mp, tp_cells, n=1):
    """
    Get the cell GIDS (0 - aligned) from those n cells,
    that are closest to mp
    mp : target parameters (x, y, u, v)
    tp_cells : same format as mp
    """
    dx = (utils.torus_distance_array(tp_cells[:, 0], mp[0]))**2
    dy = (utils.torus_distance_array(tp_cells[:, 1], mp[1]))**2

    velocity_dist = np.sqrt((tp_cells[:, 2] - mp[2])**2 + (tp_cells[:, 3] - mp[3])**2)
    summed_dist = dx + dy + velocity_dist
    gids_sorted = np.argsort(summed_dist)[:n]
    return gids_sorted, summed_dist[gids_sorted]
    


if __name__ == '__main__':

#    if len(sys.argv) > 1:
#        param_fn = sys.argv[1]
#        if os.path.isdir(param_fn):
#            param_fn += '/Parameters/simulation_parameters.json'
#        import json
#        f = file(param_fn, 'r')
#        print 'Loading parameters from', param_fn
#        params = json.load(f)
#    else:

    x_pos = float(sys.argv[1])
    vx = float(sys.argv[2])
    import simulation_parameters
    param_tool = simulation_parameters.parameter_storage()
    params = param_tool.params
    tuning_prop_exc = np.loadtxt(params['tuning_prop_exc_fn'])

    n = 10
    gids_min, distances = get_cells_near_stim((x_pos, 0, vx, 0), tuning_prop_exc, n=n)
    print 'gid_min\n', gids_min
    print 'distances\n', distances
    print 'tp:\n', 
    for gid in gids_min:
        print gid, tuning_prop_exc[gid, :]

    gid_nest = gids_min + 1

