import numpy
import sys
import NeuroTools.signals as nts
import pylab

for fn in sys.argv[1:]:
    print "Plotting", fn
    vmlist= nts.load_vmlist(fn)
    n = len(vmlist.analog_signals)
    t_axis = vmlist.time_axis()
    print "debug", vmlist.analog_signals
    print "debug", vmlist.analog_signals[1]
    vmlist.analog_signals[1].plot()
#    print len(vmlist)
#    for i in xrange(n):
#        vmlist.analog_signals[i].plot()

pylab.show()

