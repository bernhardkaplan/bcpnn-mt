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

for tau_i in [5., 10., 20., 50., 100., 150., 200., 250.]:
    #for speed in [.2, .4, .6, .8]:#, 1.]:
    #for speed in [1.]:
    for speed in [1., .8, .6, .4, .2]:
        cmd = 'sbatch jobfile_training_milner_multiple_speeds.job %f %f' % (tau_i, speed)
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
