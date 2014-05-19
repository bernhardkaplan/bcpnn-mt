import os
import sys
import numpy as np
import PlottingScripts.PlotPrediction as PP

script_name = 'PlottingScripts/PlotPrediction.py'

folder = 'TestSim_4-10_taui2000_taup50000_nHC10_nMC10_nExcPerMc8/'
n_stim = 10

for stim in xrange(3, n_stim - 1):

#    cmd = 'python %s %s %d %d' % (script_name, folder, stim, stim + 1)
    argv = [folder, stim, stim + 1]
    MAC = PP.MetaAnalysisClass(argv)

#    print cmd


