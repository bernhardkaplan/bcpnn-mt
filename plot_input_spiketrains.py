import os
import simulation_parameters
import pylab
import re
import numpy

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

folder = params['input_folder']
fn_base = params['input_st_fn_base'].rsplit(folder)[1]

fn_to_plot = []
gids = []
for fn in os.listdir(folder):
    m = re.match("%s(\d+)\." % fn_base, fn)
    if m:
        print "Found file:", fn
        try:
            data = numpy.load(folder + fn)
            fn_to_plot.append(fn)
            gids.append(int(m.groups()[0]))
        except:
            pass

fn_to_plot = sorted(fn_to_plot)
gids = sorted(gids)
n_plots = len(fn_to_plot)

# plot the input activity as map; arrange minicolumns in a grid
#n_units = params['n_mc']
n_units = params['n_exc']
x_max = int(round(numpy.sqrt(n_units)))
y_max = int(round(numpy.sqrt(n_units)))
if (n_units > x_max * y_max):
    x_max += 1
spike_count = numpy.zeros((x_max, y_max))

fig = pylab.figure()
ax = fig.add_subplot(111)
pylab.title('Input spike trains')

spikes = []
gid = []
# plot rasterplots
n_spikes = 0
for i in xrange(n_plots):
    fn = fn_to_plot[i]
    m = re.match("%s(\d+)\." % fn_base, fn)
    mc_index = int(m.groups()[0])
    data = numpy.load(folder + fn)
    spike_count[mc_index % x_max, mc_index / x_max] = data.size
    ax.plot(data, mc_index * numpy.ones(data.size), 'o', markersize=1, color='k')

ax.set_ylim(-1, max(gids)+1)

# spike count grid
fig = pylab.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Spike count over time')
cax1 = ax1.pcolor(spike_count)#, edgecolor='k', linewidths='1')
#cax1 = ax1.imshow(activity, interpolation='nearest')
#ax1.set_ylim((0, spike_count[:, 0].size))

if False:
    # plot histograms
    n_bins = 50
    fig = pylab.figure()
    pylab.subplots_adjust(bottom=0.05, top=0.9, hspace=-0.1)
    
    x_max = 1000
    xticks = numpy.arange(0, x_max, x_max/10.)
    for i in xrange(n_plots):
        fn = fn_to_plot[i]
        ax = fig.add_subplot(n_plots,1,n_plots-i)
        data = numpy.load(folder + fn)
        print fn, data.size
        count, bins = numpy.histogram(data, n_bins)
        gid = gids[i]
        ax.bar(bins[:-1], count, width=bins[1]-bins[0], facecolor='blue')#, normed=1)
        ax.set_yticks((0, max(count)))
        ax.set_yticklabels(('0', str(max(count))))
        ax.set_xlim((0, x_max))
        ax.set_xticks(xticks)
        ax.set_xticklabels(['' for i in xrange(len(xticks))])
    
    ax = fig.get_axes()[0]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%d' % i for i in xticks])
    ax.set_title('Input spike counts')
    
    pylab.show()
