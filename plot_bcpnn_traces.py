import pylab
import numpy as np
import sys
# --------------------------------------------------------------------------
def get_figsize(fig_width_pt):
    inches_per_pt = 1.0/72.0                # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

def get_figsize_A4():
    fig_width = 8.27
    fig_height = 11.69
    fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

params2 = {'backend': 'eps',
          'axes.labelsize': 12,
          'text.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.pad': 0.2,     # empty space around the legend box
          'legend.fontsize': 12,
           'lines.markersize': 3,
          'font.size': 12,
          'path.simplify': False,
          'figure.figsize': get_figsize_A4()}
#          'figure.figsize': get_figsize(800)}

pylab.rcParams.update(params2)
# --------------------------------------------------------------------------

import plot_stimulus_and_cell_tp as psac

if (len(sys.argv) < 3):
    print "Please give 2 gids to be plotted:\n"
    pre_id = int(raw_input("GID 1:\n"))
    post_id = int(raw_input("GID 2:\n"))
else:
    pre_id = int(sys.argv[1])
    post_id = int(sys.argv[2])

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.params

input_fn = params['input_rate_fn_base'] + "%d.dat" % pre_id
L_i = np.loadtxt(input_fn)
input_fn = params['input_rate_fn_base'] + "%d.dat" % post_id
L_j = np.loadtxt(input_fn)

input_fn = params['weights_fn_base'] + '%d_%d.dat' % (pre_id, post_id)
d_wij = np.loadtxt(input_fn)
input_fn = params['bias_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
d_bias = np.loadtxt(input_fn)
input_fn = params['ztrace_fn_base'] + "%d.dat" % pre_id
d_zi = np.loadtxt(input_fn)
input_fn = params['ztrace_fn_base'] + "%d.dat" % post_id
d_zj = np.loadtxt(input_fn)
input_fn = params['etrace_fn_base'] + "%d.dat" % pre_id
d_ei = np.loadtxt(input_fn)
input_fn = params['etrace_fn_base'] + "%d.dat" % post_id
d_ej = np.loadtxt(input_fn)
input_fn = params['etrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
d_eij = np.loadtxt(input_fn)
input_fn = params['ptrace_fn_base'] + "%d.dat" % pre_id
d_pi = np.loadtxt(input_fn)
input_fn = params['ptrace_fn_base'] + "%d.dat" % post_id
d_pj = np.loadtxt(input_fn)
input_fn = params['ptrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
d_pij = np.loadtxt(input_fn)

t_axis = np.arange(0, d_zi.size * params['dt_rate'], params['dt_rate'])

tp_fn = params['tuning_prop_means_fn']
tp = np.loadtxt(tp_fn)
text = 'tp_pre: ' + str(tp[pre_id, :])
text += '\ntp_post: ' + str(tp[post_id, :])
text += '\nstim: ' + str(params['motion_params'])

fig = pylab.figure()
pylab.subplots_adjust(hspace=.6)
pylab.subplots_adjust(wspace=.4)
fig.text(0.2, 0.95, text, fontsize=12)

n_rows, n_cols = 5, 2
ax = fig.add_subplot(n_rows, n_cols, 1)

ax.plot(np.arange(0, L_i.size * params['dt_rate'], params['dt_rate']), L_i)
ax.plot(np.arange(0, L_i.size * params['dt_rate'], params['dt_rate']), L_j)
ax.set_xlabel('Time [ms]')
#ax.plot(d_volt_2[:, 0], d_volt_2[:, 1])
ax.set_title("Input signal")

ax = psac.return_plot([pre_id, post_id], '%d%d%d' % (n_rows, n_cols, 2), fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Stimulus, and predicted directions")

ax = fig.add_subplot(n_rows, n_cols, 3)
ax.plot(t_axis, d_zi)
ax.plot(t_axis, d_zj)
ax.set_title("z_i, z_j")
ax.set_xlabel('Time [ms]')

ax = fig.add_subplot(n_rows, n_cols, 4)
ax.plot(t_axis, d_bias)
ax.set_title("Bias")
ax.set_xlabel('Time [ms]')

ax = fig.add_subplot(n_rows, n_cols, 5)
ax.plot(t_axis, d_ei)
ax.plot(t_axis, d_ej)
ax.set_title("e_i, e_j")
ax.set_xlabel('Time [ms]')

ax = fig.add_subplot(n_rows, n_cols, 6)
ax.plot(t_axis, d_eij)
ax.set_title("e_ij")
ax.set_xlabel('Time [ms]')

ax = fig.add_subplot(n_rows, n_cols, 7)
ax.plot(t_axis, d_pi)
ax.plot(t_axis, d_pj)
ax.set_title("p_i, p_j")
ax.set_xlabel('Time [ms]')

ax = fig.add_subplot(n_rows, n_cols, 8)
ax.plot(t_axis, d_pij)
ax.set_title("p_ij")
ax.set_xlabel('Time [ms]')

ax = fig.add_subplot(n_rows, n_cols, 10)
ax.plot(t_axis, d_wij)
ax.set_title("w_ij")
ax.set_xlabel('Time [ms]')


output_fig_fn = params['figures_folder'] + 'bcpnn_traces.pdf'
print 'Saving figure to:', output_fig_fn
pylab.savefig(output_fig_fn)
#pylab.show()
