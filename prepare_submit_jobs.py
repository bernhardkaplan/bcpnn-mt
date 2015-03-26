import numpy as np
import os

stim_idx0 = 0
stim_idx1 = 1

cnt = 0
#for bcpnn_gain in [1.]:
    #for ampa_nmda_ratio in [5.]:
        #for w_ei in [0.5]:
            #for w_ie_factor in [2.]:
                #for w_ii in [-1.]:

for w_ie_factor in [5.]:
    for w_ii in [-1.]:
        for w_ei in [.5]:
            for bcpnn_gain in [2.0]:
                for ampa_nmda_ratio in [5.]:
                    for b_adaptation in [0., 20., 40., 60., 80.4]:
                        w_ie = -w_ie_factor * w_ei
                        cmd = 'sbatch jobfile_testing_milner_with_params.sh %f %f %f %f %f %f %d %d' % (bcpnn_gain, w_ie, w_ei, ampa_nmda_ratio, w_ii, b_adaptation, stim_idx0, stim_idx1)
                        print cmd
                        os.system(cmd)
                        sleep = 'sleep 0.1'
                        print sleep
                        os.system(sleep)
                        cnt += 1

        #conn_fn_ampa = sys.argv[1]
        #conn_fn_nmda = sys.argv[2]
        #bcpnn_gain = float(sys.argv[3])
        #w_ie = float(sys.argv[4])
        #w_ei = float(sys.argv[5])
        #ampa_nmda_ratio = float(sys.argv[6])
        #w_input_exc = float(sys.argv[7])#15.
        #w_ii = float(sys.argv[8])

print 'cnt:', cnt
