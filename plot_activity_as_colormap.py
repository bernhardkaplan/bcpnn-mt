import pylab
import numpy
import sys
import os
import re
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

folder = params['spiketimes_folder']
fn_base = params['exc_spiketimes_fn_base'].rsplit(folder)[1]

n_mc = params['n_mc']
time_binsize = 50 # [ms]
n_bins = (params['t_sim'] / time_binsize) + 1
activity = numpy.zeros((n_mc, n_bins))
normed_activity = numpy.zeros((n_mc, n_bins))

# arrange minicolumns in a grid
x_max = int(round(numpy.sqrt(n_mc)))
y_max = int(round(numpy.sqrt(n_mc)))
if (n_mc > x_max * y_max):
    x_max += 1
spike_count = numpy.zeros((x_max, y_max))

print "Loading data ...."
for fn in os.listdir(folder):
    m = re.match("%s(\d+)\." % fn_base, fn)
#    print fn_base, fn
    if m:
#        try:
        print "Found file:", folder + fn,
        data = numpy.loadtxt(folder + fn)
        mc_index = int(m.groups()[0])
        nspikes = data[:, 0].size
        print "nspikes: ", nspikes
        spike_count[mc_index % x_max, mc_index / x_max] = nspikes
        for i in xrange(data[:, 0].size):
            bin_index = int(round(data[i, 0] / time_binsize))
            activity[mc_index, bin_index] += 1
#        except:
#             no spikes found
#            pass

for i in xrange(int(n_bins)):
    if (activity[:, i].sum() > 0):
        normed_activity[:, i] = activity[:, i] / activity[:,i].sum()



print "plotting ...."
fig = pylab.figure()

ax1 = fig.add_subplot(221)
ax1.set_title('Spiking activity over time')
cax1 = ax1.pcolor(activity)#, edgecolor='k', linewidths='1')
#cax1 = ax1.imshow(activity, interpolation='nearest')
ax1.set_ylim((0, activity[:, 0].size))
ax1.set_xlim((0, activity[0, :].size))
pylab.colorbar(cax1)

ax2 = fig.add_subplot(222)
ax2.set_title('Normalized spiking activity over time')
cax2 = ax2.pcolor(normed_activity)#, edgecolor='k', linewidths='1')
#cax2 = ax2.imshow(normed_activity, interpolation='nearest')
ax2.set_ylim((0, normed_activity[:, 0].size))
ax2.set_xlim((0, normed_activity[0, :].size))
pylab.colorbar(cax2)


ax3 = fig.add_subplot(223)
ax3.set_title('Spike count of cells in grid')
cax3 = ax3.pcolor(spike_count)#, edgecolor='k', linewidths='1')
#cax3 = ax3.imshow(spike_count, interpolation='nearest')
ax3.set_ylim((0, spike_count[:, 0].size))
ax3.set_xlim((0, spike_count[0, :].size))
pylab.colorbar(cax3)

pylab.show()


#figure_params = {
#    'figure.subplot.bottom': 0.10,
#    'figure.subplot.hspace': 0.9,
#    'figure.subplot.left': 0.125,
#    'figure.subplot.right': 0.90,
#    'figure.subplot.top': 0.90,
#    'figure.subplot.wspace': 0.50,
#    'legend.fontsize' : 6
#    }

#cax = ax.pcolor(data[:,:128])
#bbax=ax.get_position()
#posax = bbax.get_points()
#print "ax pos:", posax
#x0 = posax[0][0] + 0.1
#x1 = posax[1][0] + 0.1
#y0 = posax[0][1] - 0.1
#y1 = posax[1][1] - 0.1
#x0 = range(0, xmax, 8)
#y0 = range(0, ymax, 8)
#print x0, y0, x1, y1
#for i in xrange(len(x0)):
    # plot vertical lines
#    ax.plot((x0[i], x0[i]), (ymin, ymax), color='w', linewidth=1)
    # plot horizontal lines
#    ax.plot((xmin, xmax), (y0[i], y0[i]), color='w', linewidth=1)
