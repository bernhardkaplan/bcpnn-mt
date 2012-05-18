import numpy as np
import pylab
import sys

if (len(sys.argv) < 2):
    fn = raw_input("Please enter data file to be plotted\n")
else:
    fn = sys.argv[1]

data = np.loadtxt(fn)#, skiprows=1)

d = data[:, 2]
print "sum :", d.sum()
print "max :", d.max(), d.argmax()
print "min: ", d.min()
print "mean:", d.mean()
print "std: ", d.std()
print "median: ", np.median(d)

fig = pylab.figure()
ax = fig.add_subplot(111)

n_bins = 20
counts, bins = np.histogram(d, bins=100)
print bins
print counts
bin_width = bins[1] - bins[0]
ax.bar(bins[:-1], counts, width=bin_width, color='b')
#n, bins, hist = ax.hist(d, n_bins, facecolor='blue')#, normed=1)

pylab.xlabel("Value")
pylab.ylabel("Count")
#pylab.xlabel("x")
#pylab.ylabel("y")

#pylab.xlim((0, 0.01))


#n, bins, hist = ax.hist(d, 20)
pylab.show()
#counts, bins = numpy.histogram(d, bins=100)
#ax.bar(bins[:-1], counts, width=bin_width/2., color='b')
#ax.bar(bins, counts, width=bin_width., color='b')
