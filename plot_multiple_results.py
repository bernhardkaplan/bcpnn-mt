import pylab
import numpy
import sys

pylab.rcParams.update({'path.simplify' : False})
pylab.figure()

for fn in sys.argv[1:]:
    data = pylab.loadtxt(fn, skiprows=1)
    if (data.ndim == 1):
        x_axis = numpy.arange(data.size)
        pylab.plot(x_axis, data, lw=2)
#        pylab.scatter(x_axis, data)
    else:
        pylab.plot(data[:,0], data[:,1], lw=2, label=fn)

pylab.legend()
pylab.show()
