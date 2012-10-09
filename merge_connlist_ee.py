import simulation_parameters
import os
import numpy as np
PS = simulation_parameters.parameter_storage()
params = PS.load_params()

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

