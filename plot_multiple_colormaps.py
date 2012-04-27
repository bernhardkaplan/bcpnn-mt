import pylab
import numpy
import sys
import os

if (len(sys.argv) < 2):
    print("Please enter data folder to be analysed after the script name")
    print("e.g.\npython analyse_data_folder.py data_today/ ")
    exit(1)
else:
    fn = sys.argv[1]

print sys.argv[1:]
for fn in sys.argv[1:]:
    path = os.path.abspath(fn)

    print "loading data ....", fn

    # if it's only one line:
    #d = numpy.loadtxt(path)
    #data = numpy.zeros((1, d.size))
    #for i in xrange(d.size):
    #    data[0, i] = d[i]

    try:
        data = numpy.loadtxt(path, delimiter=",")#.transpose()
    except:
        data = numpy.loadtxt(path)

    # if you want to take the log of the data to plot
    #n_row = data[:, 0].size
    #n_col = data[0, :].size
    #log_data = numpy.zeros((n_row, n_col))
    #for i in xrange(n_row):
    #    for j in xrange(n_col):
    #        if data[i, j] > 0:
    #            log_data[i, j] = numpy.log(data[i, j])
    #data = log_data.copy()

    #data_rev = numpy.zeros(data.shape)
    #n_row = data[:, 0].size - 1
    #for row in xrange(data[:, 0].size):
    #    data_rev[n_row - row, :] = data[row, :]

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    print "plotting ...."
    #cax = ax.imshow(data[:,:12])
    #cax = ax.pcolor(data, edgecolor='k', linewidths='1')


    ax.set_title(fn)
    cax = ax.pcolor(data)#, edgecolor='k', linewidths='1')
    #cax = ax.pcolor(data, cmap='binary')
    #cax = ax.pcolor(data, cmap='RdBu')

    ax.set_ylim(0, data.shape[0])
    ax.set_xlim(0, data.shape[1])

    #cax = ax.pcolor(log_data)#, edgecolor='k', linewidths='1')


    pylab.colorbar(cax)

    #plot_fn = "testfig.png"
    #print "saving ....", plot_fn
    #pylab.savefig(plot_fn)

pylab.show()
