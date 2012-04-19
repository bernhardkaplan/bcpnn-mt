import pylab
import numpy
import sys

pylab.rcParams.update({'path.simplify' : False})
pylab.figure()

for fn in sys.argv[1:]:
    data = pylab.loadtxt(fn, skiprows=1)
    if (data.ndim == 1):
        x_axis = numpy.arange(data.size)
        pylab.plot(x_axis, data, lw=2, label=fn)
#        pylab.scatter(x_axis, data)
    else:
        x_axis = numpy.arange(data[:, 0].size)
        pylab.plot(x_axis, data[:,0], lw=1, label=fn)
#        pylab.plot(data[:,0], data[:,1], lw=2, label=fn)
#        pylab.scatter(numpy.arange(0, data[:, 0].size), data[:, 0], label=fn)

pylab.legend()
pylab.show()
