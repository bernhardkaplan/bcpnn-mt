import os
import sys
import numpy as np

script_name = 'PlottingScripts/PlotPrediction.py'

#folder = 'TestSim_4-10_taui2000_taup50000_nHC10_nMC10_nExcPerMc8/'
#n_stim = 10
#import PlottingScripts.PlotPrediction as PP
#for stim in xrange(3, n_stim - 1):
#    cmd = 'python %s %s %d %d' % (script_name, folder, stim, stim + 1)
#    argv = [folder, stim, stim + 1]
#    MAC = PP.MetaAnalysisClass(argv)
#    print cmd

folders = sys.argv[1:]
for f in folders:
    cmd = 'python %s %s ' % (script_name, f)
    os.system(cmd)
    print 'cmd:', cmd

display_cmd = 'ristretto $(find %s -name prediction_stim0.png)' % f[:30]
