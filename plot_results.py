import pylab
import numpy
import sys

if (len(sys.argv) < 2):
    fn = raw_input("Please enter data file to be plotted\n")
else:
    fn = sys.argv[1]

data = pylab.loadtxt(fn)
if (data.ndim == 1):
    x_axis = numpy.arange(data.size)
    pylab.plot(x_axis, data)
else:
    pylab.plot(data[:,0], data[:,1])
pylab.show()
