import numpy as np
from plot_dx_dv_sweep import plot_set_of_curves

fn1 = 'TwoCellTauZiZjESweep_taup10000_vstim0.50_prex0.00_u0.50_dx0.50_0.00/tauzizje_sweep_taup10000_x00.00_u00.50_vstim0.50_sorted.dat'
fn2 = 'TwoCellTauZiZjESweep_taup50000_vstim0.50_prex0.00_u0.50_dx0.50_0.00/tauzizje_sweep_taup50000_x00.00_u00.50_vstim0.50.dat'
#d_taup10 = np.loadtxt(fn1)
#d_taup50 = np.loadtxt(fn2)


plot_set_of_curves(fn1, x_axis_idx=3, y_axis_idx=10, idx_for_the_set=4, \
        xlabel='$\\tau_{z, i}$', ylabel='$w_{avg, end}$', setlabel='$\\tau_{z, j}$',
        output_fn='TwoCellTauZiZjESweep_taup10000_vstim0.50_prex0.00_u0.50_dx0.50_0.00/tauzizj_set_of_curves.png')


plot_set_of_curves(fn1, x_axis_idx=3, y_axis_idx=10, idx_for_the_set=5, \
        xlabel='$\\tau_{z, i}$', ylabel='$w_{avg, end}$', setlabel='$\\tau_{e}$',
        output_fn='TwoCellTauZiZjESweep_taup10000_vstim0.50_prex0.00_u0.50_dx0.50_0.00/tauzi_taue_set_of_curves.png')


