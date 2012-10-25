import pylab
import numpy as np
import sys

def plot_all(params, pre_id, post_id, iteration, fig=None, text=None, show=True, output_fn=None, **kwargs):

    # --------------------------------------------------------------------------
    def get_figsize(fig_width_pt):
        inches_per_pt = 1.0/72.0                # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width*golden_mean      # height in inches
        fig_size =  [fig_width,fig_height]      # exact figsize
        return fig_size

    def get_figsize_landscape(fig_width_pt):
        inches_per_pt = 1.0/72.0                # Convert pt to inch
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height = fig_width/golden_mean      # height in inches
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
#              'figure.figsize': get_figsize_A4()}
              'figure.figsize': get_figsize_landscape(600)}
#              'figure.figsize': get_figsize(1200)}

    pylab.rcParams.update(params2)
    # --------------------------------------------------------------------------
    
    motion_params_fn = "%sTrainingInput_%d/input_params.txt" % (params['folder_name'], iteration)
    mp = np.loadtxt(motion_params_fn)

    # get filenames from keywords or set the default names
    input_fn_base = kwargs.get('input_fn_base', params['input_rate_fn_base'])
    L_i_fn = input_fn_base + '%d.dat' % (pre_id)
    L_j_fn = input_fn_base + '%d.dat' % (post_id)
#    L_i_fn = kwargs.get('L_i_fn', params['input_rate_fn_base'] + "%d.dat" % pre_id)
#    L_j_fn = kwargs.get('L_j_fn', params['input_rate_fn_base'] + "%d.dat" % post_id)
    wij_fn = kwargs.get('wij_fn', params['weights_fn_base'] + '%d_%d.dat' % (pre_id, post_id))
    bias_fn = kwargs.get('bias_fn', params['bias_fn_base'] + "%d.dat" % (post_id))
    zi_fn = kwargs.get('zi_fn', params['ztrace_fn_base'] + "%d.dat" % pre_id)
    zj_fn = kwargs.get('zj_fn', params['ztrace_fn_base'] + "%d.dat" % post_id)
    ei_fn = kwargs.get('ei_fn', params['etrace_fn_base'] + "%d.dat" % pre_id)
    ej_fn = kwargs.get('ej_fn', params['etrace_fn_base'] + "%d.dat" % post_id)
    eij_fn = kwargs.get('eij_fn', params['etrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id))
    pi_fn = kwargs.get('pi_fn', params['ptrace_fn_base'] + "%d.dat" % pre_id)
    pj_fn = kwargs.get('pj_fn', params['ptrace_fn_base'] + "%d.dat" % post_id)
    pij_fn = kwargs.get('pij_fn', params['ptrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id))

    L_i = np.loadtxt(L_i_fn)
    L_j = np.loadtxt(L_j_fn)
    d_wij = np.loadtxt(wij_fn)
    d_bias = np.loadtxt(bias_fn)
    d_zi = np.loadtxt(zi_fn)
    d_zj = np.loadtxt(zj_fn)
    d_ei = np.loadtxt(ei_fn)
    d_ej = np.loadtxt(ej_fn)
    d_eij = np.loadtxt(eij_fn)
    d_pi = np.loadtxt(pi_fn)
    d_pj = np.loadtxt(pj_fn)
    d_pij = np.loadtxt(pij_fn)

    t_axis = np.arange(0, d_zi.size * params['dt_rate'], params['dt_rate'])

    tp_fn = params['tuning_prop_means_fn']
    tp = np.loadtxt(tp_fn)
    if fig == None:
        fig = pylab.figure()
    pylab.subplots_adjust(hspace=.6)
    pylab.subplots_adjust(wspace=.4)
#    fig.text(0.2, 0.95, text, fontsize=12)

    c1, c2, c3 = 'b', 'g', 'k' # line colors
    n_rows, n_cols = 5, 2
    ax = fig.add_subplot(n_rows, n_cols, 1)

    ax.plot(np.arange(0, L_i.size * params['dt_rate'], params['dt_rate']), L_i, c=c1)
    ax.plot(np.arange(0, L_i.size * params['dt_rate'], params['dt_rate']), L_j, c=c2)
    ax.set_xlabel('Time [ms]')
    #ax.plot(d_volt_2[:, 0], d_volt_2[:, 1])
    ax.set_title("Input signal")

    ax = psac.return_plot([pre_id, post_id], '%d%d%d' % (n_rows, n_cols, 2), fig, input_fn_base=input_fn_base, motion_params=mp)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Stimulus, and predicted directions")

    ax = fig.add_subplot(n_rows, n_cols, 3)
    ax.plot(t_axis, d_zi, c=c1)
    ax.plot(t_axis, d_zj, c=c2)
    ax.set_title("z_i, z_j")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 4)
    ax.plot(t_axis, d_bias, c=c3)
    ax.set_title("Bias")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 5)
    ax.plot(t_axis, d_ei, c=c1)
    ax.plot(t_axis, d_ej, c=c2)
    ax.set_title("e_i, e_j")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 6)
    ax.plot(t_axis, d_eij, c=c3)
    ax.set_title("e_ij")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 7)
    ax.plot(t_axis, d_pi, c=c1)
    ax.plot(t_axis, d_pj, c=c2)
    ax.set_title("p_i, p_j")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 8)
    ax.plot(t_axis, d_pij, c=c3)
    ax.set_title("p_ij")
    ax.set_xlabel('Time [ms]')

    ax = fig.add_subplot(n_rows, n_cols, 9)
    if text == None:
        text = 'iteration: %d\n' % (iteration)
        text += 'pre_id=%d  post_id=%d\n' % (pre_id, post_id)
        text += 'tp_pre: ' + str(tp[pre_id, :])
        text += '\ntp_post: ' + str(tp[post_id, :])
        text += '\nstim: ' + str(params['motion_params'])
    ax.annotate(text, (.1, .1), fontsize=12)

    ax = fig.add_subplot(n_rows, n_cols, 10)
    ax.plot(t_axis, d_wij, c=c3)
    ax.set_title("w_ij")
    ax.set_xlabel('Time [ms]')

    if output_fn == None:
        output_fig_fn = params['figures_folder'] + 'bcpnn_traces.png'
    else:
        output_fig_fn = output_fn

    if show == True:
        pylab.show()
    else:
        print 'Saving figure to:', output_fig_fn
        pylab.savefig(output_fig_fn)
        return fig

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print "Please give 2 gids to be plotted:\n"
        pre_id = int(raw_input("GID 1:\n"))
        post_id = int(raw_input("GID 2:\n"))
        iteration = int(raw_input("Iteration:\n"))
    else:
        pre_id = int(sys.argv[1])
        post_id = int(sys.argv[2])
        iteration = int(sys.argv[3])

    import plot_stimulus_and_cell_tp as psac
    import simulation_parameters
    PS = simulation_parameters.parameter_storage()
    params = PS.params

#    plot_all(params, pre_id, post_id)

#    L_i_fn = "%sTrainingInput_%d/%s%d.dat" % (params['folder_name'], iteration, params['abstract_input_fn_base'], pre_id)
#    L_j_fn = "%sTrainingInput_%d/%s%d.dat" % (params['folder_name'], iteration, params['abstract_input_fn_base'], post_id)
    input_fn_base = '%sTrainingInput_%d/%s' % (params['folder_name'], iteration, params['abstract_input_fn_base'])
    wij_fn = "%swij_%d_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id, post_id)
    bias_fn = "%sbias_%d_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id, post_id)
    zi_fn = "%szi_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id)
    zj_fn = "%szj_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, post_id)
    ei_fn = "%sei_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id)
    ej_fn = "%sej_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, post_id)
    pi_fn = "%spi_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, pre_id)
    pj_fn = "%spj_%d_%d.dat" % (params['bcpnntrace_folder'], iteration, post_id)
    eij_fn = '%seij_%d_%d_%d.dat' % (params['bcpnntrace_folder'], iteration, pre_id, post_id)
    pij_fn = '%spij_%d_%d_%d.dat' % (params['bcpnntrace_folder'], iteration, pre_id, post_id)


    plot_all(params, pre_id, post_id, iteration, \
#            L_i_fn=L_i_fn, \
#            L_j_fn=L_j_fn, \
            input_fn_base=input_fn_base, \
            wij_fn=wij_fn, \
            bias_fn=bias_fn, \
            zi_fn=zi_fn, \
            zj_fn=zj_fn, \
            ei_fn=ei_fn, \
            ej_fn=ej_fn, \
            eij_fn=eij_fn, \
            pi_fn=pi_fn, \
            pj_fn=pj_fn, \
            pij_fn=pij_fn)

