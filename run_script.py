import sys
import time
import numpy as np
import os
t1 = time.time()

#script_name = 'PlottingScripts/PlotCurrents.py'
script_name = 'main_test.py'
#conn_fn_1 = 'connection_matrix_20x2_taui5_v0.5.dat'
#conn_fn_2 = 'connection_matrix_20x2_taui5_v0.8.dat'
#conn_fn_2 = 'connection_matrix_20x2_taui200_v0.5.dat'
#conn_fn_2 = 'connection_matrix_20x2_taui150_v0.8.dat'

conn_fn_1 = 'connection_matrix_20x4_taui5_v0.4_0.8.dat'
conn_fn_2 = 'connection_matrix_20x4_taui200_v0.4_0.8.dat'
#bcpnn_gain_range = [1.0, 1.5, 2.0, 2.5]
bcpnn_gain_range = [1., 2., 3.]
#w_ie_factor_range = [-5.] # FACTOR
w_ie_factor_range = [-5.]
#w_ei_range = [6., 8., 10., 12., 15.]
w_ei_range = [5.]
#ampa_nmda_ratio_range = [0.1]#, 0.2, 0.5, 1.0, 2.]
ampa_nmda_ratio_range = [0., 1.]
w_ii_range = [-2.]
#w_ii_range = [-1., -2., -3., -0.5, -0.1]
n_runs = len(bcpnn_gain_range) * len(w_ie_factor_range) * len(w_ei_range) * len(ampa_nmda_ratio_range) * len(w_ii_range)
it_cnt = 0

silent = True
for bcpnn_gain in bcpnn_gain_range:
    for w_ie_factor in w_ie_factor_range:
        for w_ei in w_ei_range:
            w_ie = w_ie_factor * w_ei
            for w_ii in w_ii_range:
                for ampa_nmda_ratio in ampa_nmda_ratio_range:
                    if silent:
                        command = 'mpirun -np 8 python %s %s %s %f %f %f %f %f > delme_%d' % (script_name, conn_fn_1, conn_fn_2, bcpnn_gain, w_ie, w_ei, ampa_nmda_ratio, w_ii, it_cnt)
                    else:
                        command = 'mpirun -np 8 python %s %s %s %f %f %f %f %f ' % (script_name, conn_fn_1, conn_fn_2, bcpnn_gain, w_ie, w_ei, ampa_nmda_ratio, w_ii)
                    print '\n\n-------------------\n\n\tIteration: %d / %d \n\n----------------------\n\n' % (it_cnt + 1, n_runs)
                    print command
                    os.system(command)
                    it_cnt += 1

t2 = time.time() - t1
print "Sweep with %d runs took %.2f seconds or %.2f minutes" % (n_runs, t2, t2/60.)
