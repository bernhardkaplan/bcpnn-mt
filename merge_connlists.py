import os
import utils
import numpy as np
import sys


if len(sys.argv) > 1:
    param_fn = sys.argv[1]
    if os.path.isdir(param_fn):
        param_fn += '/Parameters/simulation_parameters.json'
    import json
    f = file(param_fn, 'r')
    print 'Loading parameters from', param_fn
    params = json.load(f)

else:
    print '\n NOT successfull\nLoading the parameters currently in simulation_parameters.py\n'
    import simulation_parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

# E -> E 
tmp_fn = 'delme_tmp_%d' % (np.random.randint(0, 1e8))
cat_cmd = 'cat %s* > %s' % (params['conn_list_ee_fn_base'], tmp_fn)
sort_cmd = 'sort -gk 1 -gk 2 %s > %s' % (tmp_fn, params['merged_conn_list_ee'])
rm_cmd = 'rm %s' % (tmp_fn)

print cat_cmd
os.system(cat_cmd)
print sort_cmd
os.system(sort_cmd)
print rm_cmd
os.system(rm_cmd)


# E -> I
tmp_fn = 'delme_tmp_%d' % (np.random.randint(0, 1e8))
cat_cmd = 'cat %s* > %s' % (params['conn_list_ei_fn_base'], tmp_fn)
sort_cmd = 'sort -gk 1 -gk 2 %s > %s' % (tmp_fn, params['merged_conn_list_ei'])
rm_cmd = 'rm %s' % (tmp_fn)

print cat_cmd
os.system(cat_cmd)
print sort_cmd
os.system(sort_cmd)
print rm_cmd
os.system(rm_cmd)

# I -> E
tmp_fn = 'delme_tmp_%d' % (np.random.randint(0, 1e8))
cat_cmd = 'cat %s* > %s' % (params['conn_list_ie_fn_base'], tmp_fn)
sort_cmd = 'sort -gk 1 -gk 2 %s > %s' % (tmp_fn, params['merged_conn_list_ie'])
rm_cmd = 'rm %s' % (tmp_fn)

print cat_cmd
os.system(cat_cmd)
print sort_cmd
os.system(sort_cmd)
print rm_cmd
os.system(rm_cmd)


# I -> E
tmp_fn = 'delme_tmp_%d' % (np.random.randint(0, 1e8))
cat_cmd = 'cat %s* > %s' % (params['conn_list_ii_fn_base'], tmp_fn)
sort_cmd = 'sort -gk 1 -gk 2 %s > %s' % (tmp_fn, params['merged_conn_list_ii'])
rm_cmd = 'rm %s' % (tmp_fn)

print cat_cmd
os.system(cat_cmd)
print sort_cmd
os.system(sort_cmd)
print rm_cmd
os.system(rm_cmd)

