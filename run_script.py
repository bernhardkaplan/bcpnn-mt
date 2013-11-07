import sys
import time
import numpy as np
import os
t1 = time.time()

script_name = 'toy_experiment.py'

#tau_zis = [10, 100, 1000, 5000]
#tau_zis = [10, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]
tau_zis = [10, 100, 500, 1000, 2500, 5000]
#tau_zis = [1000]
#tau_zis = [10, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]
#tau_zjs = [10, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]
tau_zjs = [10]
#tau_es = [10, 100, 500, 1000, 2500, 5000]
tau_es = [10]
tau_ps = [100000]
#dxs = [.5]
dxs = np.around(np.arange(2.0, -.2, -.10), decimals=2)
#dvs = [.0]
dvs = np.around(np.arange(.0, .20, .05), decimals=2)
#v_stims = np.around(np.arange(0.1, 2.00, 0.05), decimals=2)
v_stims = [.1, 0.5, 1.0, 1.5]
#v_stims = [0.1]
n_runs = len(tau_ps) * len(tau_es) * len(dxs) * len(v_stims) * len(tau_zis) * len(tau_zjs) * len(dvs)
it_cnt = 0 
#x0, u0 = .5, .5
for tau_p in tau_ps:
    for tau_e in tau_es:
        for tau_zj in tau_zjs:
            for v_stim in v_stims:
                x0, u0 = .0 + .5 * v_stim, v_stim # stimulus starts at 0
                for tau_zi in tau_zis:
                    for dv in dvs:
                        for dx in dxs:
#                            output_folder = 'TwoCellTauZiZjESweep_taup%d_vstim%.2f_prex%.2f_u%.2f_dx%.2f_%.2f/' % (tau_p, v_stim, x0, u0, dx, dv)
#                            output_folder = 'TwoCell_dxdvSweep_tauzi%d_tauzj%d_taup%d_taue%d_vstim%.2f_prex%.2f_u%.2f/' % (tau_zi, tau_zj, tau_p, tau_p, v_stim, x0, u0)
                            output_folder = 'TwoCell_combined_dxdv_tauzizj_Sweep_taue%d_taup%d_vstim%.2f_prex%.2f_u%.2f/' % (tau_e, tau_p, v_stim, x0, u0)
                            command = 'python %s %d %f %f %f %d %d %d %f %f %s' % (script_name, tau_zi, v_stim, dx, dv, tau_zj, tau_e, tau_p, x0, u0, output_folder)
                            print '\n-------------------\n\tIteration: %d / %d\tdv = %.1f\n----------------------\n' % (it_cnt, n_runs, dv)
                            print command
                            os.system(command)
                            it_cnt += 1

t2 = time.time() - t1
print "Sweep with %d runs took %.2f seconds or %.2f minutes" % (n_runs, t2, t2/60.)
