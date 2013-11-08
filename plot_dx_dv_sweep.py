import numpy as np
import pylab
import sys

rcParams = { 'axes.labelsize' : 18,
            'label.fontsize': 20,
            'xtick.labelsize' : 16, 
            'ytick.labelsize' : 16, 
            'axes.titlesize'  : 16,
            'legend.fontsize': 9, 
            'lines.markeredgewidth' : 0}
color_list = ['k', 'b', 'g', 'r', 'y', 'c', 'm', '#00f80f', '#deff00', '#ff00e4', '#00ffe6']

def plot_data_for_one_dv():
    """
    Here, there is only one dv and it's really simple
    """
    fn = 'TwoCell_dxdvSweep_tauzi1000_tauzj1000_taup50000_taue50000_vstim0.50_prex0.25_u0.50/dx_sweep_dv0.00.dat'
    d = np.loadtxt(fn)
    x_axis = d[:, 0]
    y_axis = d[:, 10]

    pylab.rcParams.update(rcParams)
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_axis, y_axis, 'o-')
    ax.set_xlabel('x_post - x_pre')
    ax.set_ylabel('BCPNN weight')
    ax.set_title('Weight after training')
    pylab.show()


def plot_set_of_curves(fn, x_axis_idx=0, y_axis_idx=10, idx_for_the_set=1, xlabel=None, ylabel=None, title=None, setlabel='', output_fn=None):
    """
    This function processes the raw data writte by a sweep over dx and dv
    Keyword arguments:
    x_axis_idx -- column index which contains data to be displayed on x-axis
    y_axis_idx -- column index which contains data to be displayed on y-axis
    idx_for_the_set -- column index for data that contains to one set
    xlabel, ylabel, title -- strings for the corresponding 
    """
    d = np.loadtxt(fn)

    pylab.rcParams.update(rcParams)
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    dvs, indices = np.unique(d[:, idx_for_the_set], return_inverse=True)
    for i in xrange(dvs.size): # iterate over the different sets
        idx = (indices == i).nonzero()[0] # indices belonging to this set
        x_data = d[idx, x_axis_idx]
        y_data = d[idx, y_axis_idx]
        idx_max = idx[y_data.argmax()]
        print 'debug setlabel', setlabel, dvs[i], 'max at %s' % xlabel, d[idx_max, x_axis_idx], d[idx_max, y_axis_idx]

        ax.plot(x_data, y_data, 'o-', c=color_list[i % len(color_list)], label='%s: %.2e' % (setlabel, dvs[i]))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    pylab.legend(loc='upper right')#'lower right')
    xmin, xmax = np.min(d[:, x_axis_idx]), np.max(d[:, x_axis_idx])
    ax.set_xlim((xmin - .1 * (xmax - xmin), xmax + .1 * (xmax - xmin)))
    if output_fn != None:
        print 'Saving figure to:', output_fn
        pylab.savefig(output_fn)
    else:
        pylab.show()

#    print dvs, indices

if __name__ == '__main__':
    plot_set_of_curves(fn = 'TwoCell_dxdvSweep_tauzi1000_tauzj1000_taup50000_taue50000_vstim0.50_prex0.25_u0.50/tauzizje_sweep_taup50000_x00.25_u00.50_vstim0.50.dat', \
            x_axis_idx=0, y_axis_idx=10, idx_for_the_set=1, xlabel='x_post - x_pre', ylabel='BCPNN weight', setlabel='dv', title='Weight after training, $\\tau_p = 50 s$',\
            output_fn='TwoCell_dxdvSweep_tauzi1000_tauzj1000_taup50000_taue50000_vstim0.50_prex0.25_u0.50/dx_dv_sweep_taup_50s.png')
    
    plot_set_of_curves(fn = 'TwoCell_dxdvSweep_tauzi1000_tauzj1000_taup10000_taue10000_vstim0.50_prex0.25_u0.50/tauzizje_sweep_taup10000_x00.25_u00.50_vstim0.50.dat', \
            x_axis_idx=0, y_axis_idx=10, idx_for_the_set=1, xlabel='x_post - x_pre', ylabel='BCPNN weight', setlabel='dv', title='Weight after training, $\\tau_p = 10 s$',\
            output_fn='TwoCell_dxdvSweep_tauzi1000_tauzj1000_taup10000_taue10000_vstim0.50_prex0.25_u0.50/dx_dv_sweep_taup_10s.png')

    plot_set_of_curves(fn = 'TwoCell_dxdvSweep_tauzi1000_tauzj1000_taup100000_taue100000_vstim0.50_prex0.25_u0.50/tauzizje_sweep_taup100000_x00.25_u00.50_vstim0.50.dat', 
            x_axis_idx=0, y_axis_idx=10, idx_for_the_set=1, xlabel='x_post - x_pre', ylabel='BCPNN weight', setlabel='dv', title='Weight after training, $\\tau_p = 100 s$',\
            output_fn='TwoCell_dxdvSweep_tauzi1000_tauzj1000_taup100000_taue100000_vstim0.50_prex0.25_u0.50/dx_dv_sweep_taup_100s.png')
