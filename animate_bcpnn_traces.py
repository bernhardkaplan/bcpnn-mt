import numpy as np
import Bcpnn
import pylab
import os

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.params

import plot_bcpnn_traces as plotter

#pylab.ion()

pre_id = 145
post_id = 177
trace_len = params['t_stimulus'] / params['dt_rate']
bcpnn_trace_len = params['t_stimulus'] * 2 / params['dt_rate']
pre_trace = np.zeros(bcpnn_trace_len)
d = np.loadtxt(params['input_rate_fn_base'] + str(pre_id) + '.dat')
print 'debug', trace_len, d.size, bcpnn_trace_len
pre_trace[:trace_len] = d
post_trace = np.zeros(bcpnn_trace_len)
d = np.loadtxt(params['input_rate_fn_base'] + str(post_id) + '.dat')
post_trace[:trace_len] = d
post_trace[trace_len:] = 0.
post_trace.resize(bcpnn_trace_len)

for i in xrange(5):
    tau_z = 10
    tau_e = 100
    tau_p = 500 + 500 * i
    # compute
    wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = Bcpnn.get_spiking_weight_and_bias(pre_trace, post_trace, get_traces=True, tau_z=tau_z, tau_e=tau_e, tau_p=tau_p, f_max=1000.)
    weight_fn = params['weights_fn_base'] + '%d_%d.dat' % (pre_id, post_id)
    print 'Saving to ', weight_fn
    np.savetxt(weight_fn, wij)
    output_fn = params['bias_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
    np.savetxt(output_fn, bias)

    output_fn = params['ztrace_fn_base'] + "%d.dat" % pre_id
    np.savetxt(output_fn, zi)
    output_fn = params['ztrace_fn_base'] + "%d.dat" % post_id
    np.savetxt(output_fn, zj)

    output_fn = params['etrace_fn_base'] + "%d.dat" % pre_id
    np.savetxt(output_fn, ei)
    output_fn = params['etrace_fn_base'] + "%d.dat" % post_id
    np.savetxt(output_fn, ej)
    output_fn = params['etrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
    np.savetxt(output_fn, eij)

    output_fn = params['ptrace_fn_base'] + "%d.dat" % pre_id
    np.savetxt(output_fn, pi)
    output_fn = params['ptrace_fn_base'] + "%d.dat" % post_id
    np.savetxt(output_fn, pj)
    output_fn = params['ptrace_fn_base'] + "%d_%d.dat" % (pre_id, post_id)
    np.savetxt(output_fn, pij)


    text = ''
    text += 'tau_z = %d\n' % tau_z
    text += 'tau_e = %d\n' % tau_e
    text += 'tau_p = %d\n' % tau_p
    text += 'x_stim(t) = (%.1f, %.1f) + (%.1f, %.1f) * t\n'% (mp[0], mp[1], mp[2], mp[3])
    output_fn = params['figures_folder'] + 'bcpnn_%d.png' % (i)
    fig = None
    fig = plotter.plot_all(pre_id, post_id, fig, text, show=False, output_fn=output_fn)
    #fig.canvas.draw() 

input_fn = params['figures_folder'] + 'bcpnn_%d.png'
output_fn = params['movie_folder'] + 'bcpnn_traces.avi'
fps = 0.5
import make_movie as mm
mm.avconv(input_fn, output_fn, fps)
