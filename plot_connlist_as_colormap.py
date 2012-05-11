import pylab
import numpy as np
import sys
import os
import utils
import simulation_parameters

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
sim_cnt = 0
if (len(sys.argv) < 2):
    fns = [params['conn_list_ee_fn_base'] + '%d.dat' % sim_cnt]
    print "Plotting default file:", fn
else:
    fns = sys.argv[1:]

n_cells = params['n_exc']
n_dw = len(fns) - 1

#for fn in fns:
#    data = utils.convert_connlist_to_matrix(fn, n_cells)
#    fig = pylab.figure()
#    ax = fig.add_subplot(111)
#    print "plotting ...."
#    ax.set_title(fn)
#    cax = ax.pcolor(data)#, edgecolor='k', linewidths='1')
#    ax.set_ylim(0, data.shape[0])
#    ax.set_xlim(0, data.shape[1])
#    pylab.colorbar(cax)

    #cax = ax.pcolor(data, cmap='binary')
    #cax = ax.pcolor(data, cmap='RdBu')
    #cax = ax.imshow(data[:,:12])

if (n_dw > 0):
#    dws = [np.zeros((n_cells, n_cells)) for i in xrange(n_dw)]
    # plot the difference weight matrix
    for i in xrange(len(fns)-1):
        fn1 = fns[i]
        fn2 = fns[i+1]
        d1 = utils.convert_connlist_to_matrix(fn1, n_cells)
        d2 = utils.convert_connlist_to_matrix(fn2, n_cells)
        dw = d2 - d1
        data = dw

        print "plotting dw"
        fig = pylab.figure()
        ax = fig.add_subplot(121)
        ax.set_title("Difference %s \n- %s" % (fn1, fn2))
        cax = ax.pcolor(data)#, edgecolor='k', linewidths='1')
        #cax = ax.pcolor(data, cmap='binary')
        #cax = ax.pcolor(data, cmap='RdBu')
        #cax = ax.imshow(data[:,:12])
        ax.set_ylim(0, data.shape[0])
        ax.set_xlim(0, data.shape[1])
        pylab.colorbar(cax)

        ax = fig.add_subplot(122)
        ax.set_title("dw histogram")
        count, bins = np.histogram(dw, bins=20)
        ax.bar(bins[:-1], count) 
pylab.show()
