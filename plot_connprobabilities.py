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

label = "sum : %.2e\n \
max : %.2e\n \
min : %.2e\n \
mean: %.2e\n \
std : %.2e\n \
median: %.2e" % (d.sum() , d.max() , d.min() , d.mean() , d.std() , np.median(d))

fig = pylab.figure()
ax = fig.add_subplot(111)

n_bins = 100
counts, bins = np.histogram(d, bins=n_bins)
print bins
print counts
bin_width = bins[1] - bins[0]
ax.bar(bins[:-1], counts, width=bin_width, color='b', label=label)
#n, bins, hist = ax.hist(d, n_bins, facecolor='blue')#, normed=1)

pylab.xlabel("Connection probability")
#pylab.xlabel("Value")
pylab.ylabel("Count")
#pylab.xlabel("x")
#pylab.ylabel("y")

#pylab.xlim((0, 0.01))
pylab.ylim((0, 10000))

pylab.legend()
#pylab.title('Connection probability histogram')

pylab.title('$\sigma_X=\sigma_V=0.1$')
output_fn = fn.rsplit('.dat')[0] + '.png'
print "Saving to ", output_fn
pylab.savefig(output_fn)

#pylab.title(fn)

#n, bins, hist = ax.hist(d, 20)
#pylab.show()
#counts, bins = numpy.histogram(d, bins=100)
#ax.bar(bins[:-1], counts, width=bin_width/2., color='b')
#ax.bar(bins, counts, width=bin_width., color='b')
