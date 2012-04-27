import pylab
import numpy as np
import sys

import sys
import numpy as np

fn = sys.argv[1]

f = open(fn, 'r')
header = []
l = f.readline()
size = int(l.rsplit('=')[-1]) # the number of cells recorded in this file
l = f.readline()
first_index = int(l.rsplit('=')[-1])
l = f.readline()
first_id = int(l.rsplit('=')[-1])
l = f.readline()
n = int(l.rsplit('=')[-1])
l = f.readline()
variable = l.rsplit('=')[-1]
l = f.readline()
last_id = int(l.rsplit('=')[-1])
l = f.readline()
last_index = int(l.rsplit('=')[-1])
l = f.readline()
dt = float(l.rsplit('=')[-1])
l = f.readline()
label = l.rsplit('=')[-1]

d = np.loadtxt(fn)
vms = [[] for i in xrange(size)]

#print d.shape
if d.shape[-1] == 3: # nest output
    # ids     time    volt
    t_axis = d[::size, 1]
    vm_index = 2

else:   # compatible_output
    # ids volt
    t_stop = n / size
    t_axis = numpy.arange(0, t_stop) * dt
    vm_index = 1

for i in xrange(size):
    vms[i] = d[i::size, vm_index]


pylab.rcParams.update({'path.simplify' : False})
pylab.figure()

for i in xrange(size):
    pylab.plot(t_axis, vms[i])

#pylab.legend()
pylab.show()
