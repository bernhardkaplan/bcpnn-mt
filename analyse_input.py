import numpy as np
import utils
import pylab
import sys
import re
import os


import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

#params['blur_X'], params['blur_V'] = float(sys.argv[1]), float(sys.argv[2])
#file_count = int(sys.argv[3])
print 'Blur', params['blur_X'], params['blur_V']


#def sort_cells_by_distance_to_stimulus(n_cells):
#    tp = np.loadtxt(params['tuning_prop_means_fn'])
#    mp = params['motion_params']
#    print 'sort_gids_by_distance_to_stimulus'
#    indices, distances = utils.sort_gids_by_distance_to_stimulus(tp , mp) # cells in indices should have the highest response to the stimulus
#    print 'Motion parameters', mp
#    print 'GID\tdist_to_stim\tx\ty\tu\tv\t\t'
#    for i in xrange(n_cells):
#        gid = indices[i]
#        print gid, '\t', distances[i], tp[gid, :]
#    return indices, distances

#fig = pylab.figure()
#ax1 = fig.add_subplot(211)
#n_cells = params['n_gids_to_record']
#idx, dist = sort_cells_by_distance_to_stimulus(n_cells)
#idx = np.random.randint(0, params['n_exc'], n)
#idx = range(params['n_exc'])
#w_stim = params['w_input_exc']
#n = params['n_exc']
#print 'gid\tn_spikes_stim\tt_stim\tg_in [nS]\tg_in_t [nS/s]'
#for i in xrange(n):
#    stim_spikes = np.load(params['input_st_fn_base'] + str(idx[i]) + '.npy')
#    n_spikes_stim = stim_spikes.size
#    if n_spikes_stim > 1:
#        t_stim = np.max(stim_spikes) - np.min(stim_spikes)
#        g_in = n_spikes_stim * w_stim * params['tau_syn_exc'] * 1000. / t_stim
#        g_in_rate = n_spikes_stim * w_stim * params['tau_syn_exc'] * (1000. / t_stim)**2
#        print '%d\t%d\t\t%d\t%.3f\t\t%.3f' % (idx[i], n_spikes_stim, t_stim, g_in,g_in_rate)
#        ax1.plot(n_spikes_stim, g_in, 'k.')

#pylab.show()
#exit(1)

def nspikes_to_g(n, w):
    """
    returns the average conductance received by the n input spikes within the t_stimulus in [uS]
    """
    return n / (params['t_stimulus'] / 1000.) * w * params['tau_syn_exc']

def update_ax2(ax1):
   y1, y2 = ax1.get_ylim()
   ax2.set_ylim(nspikes_to_g(y1, params['w_input_exc']), nspikes_to_g(y2, params['w_input_exc']))
   ax2.figure.canvas.draw()

w_exc = params['w_input_exc']
folder = params['input_folder']
fn_base = params['input_st_fn_base'].rsplit(folder)[1]
all_spikes = np.zeros(params['n_exc'], dtype='int')
input_spike_files = {}

for fn in os.listdir(folder):
    to_match = "%s(\d+)\.npy" % fn_base
    m = re.match(to_match, fn)
    if m:
        gid = int(m.groups()[0])
        path = '%s%s' % (folder, fn)
        all_spikes[gid] = np.load(path).size

receiving_nrns = all_spikes.nonzero()[0]
n_receiving_nrns = len(receiving_nrns)
input_spikes = all_spikes[receiving_nrns]
#print 'debug', type(input_spikes)
if len(input_spikes) == 0:
    input_spikes_mean = 0.
    input_spikes_std = 0.
    input_spikes_max = 0.
else:
    input_spikes_mean = input_spikes.mean()
    input_spikes_std = input_spikes.std()
    input_spikes_max= input_spikes.max()

#print 'Neurons receiving inputs:', receiving_nrns
#print 'n spikes :', input_spikes[receiving_nrns]
print 'Num spike receiving neurons: %d ~ %.2f percent of all' % (n_receiving_nrns, n_receiving_nrns / float(params['n_exc']) * 100.)
print ' average number of spikes: %.2f +- (%.2f)   g_exc = %.2e +- (%2.e) [uS]' \
        % (input_spikes_mean, input_spikes_std, nspikes_to_g(input_spikes_mean, w_exc), nspikes_to_g(input_spikes_std, w_exc))
idx = all_spikes.argsort().tolist()
idx.reverse()
#print input_spikes[idx]
print 'gid\tnspikes\tg_in [uS]'
#for i_, gid in enumerate(idx):

for i in xrange(20):#n_receiving_nrns):
    gid = idx[i]
    nspk = all_spikes[gid]
    print '%d\t%d\t%.3f' % (gid, nspk, nspikes_to_g(nspk, params['w_input_exc']))


g_in_all = nspikes_to_g(all_spikes, params['w_input_exc'])
g_in = nspikes_to_g(input_spikes, params['w_input_exc'])
#g_sum, g_min, g_max, g_mean, g_std, g_median = g_in.sum(), g_in.min(), g_in.max(), g_in.mean(), g_in.std(), np.median(g_in)

g_in_sorted = np.copy(g_in)
g_in_sorted.sort()
pmax = 0.10
n_ = round(pmax * params['n_exc'])

fig = pylab.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.callbacks.connect("ylim_changed", update_ax2)
sorted_input_spikes = np.copy(all_spikes)
sorted_input_spikes.sort()
x = np.arange(params['n_exc'])
t_stim = params['t_stimulus'] / 1000.
label_text = ''
#label_text += 'nspikes_in\n'
label_text += 'nspikes_in_sum = %d\n' % (all_spikes.sum())
label_text += 'nspikes_in_mean = %.2f +- %.2f\n' % (all_spikes.mean(), all_spikes.std())
label_text += 'nspikes_in_non-zeros mean = %.2f +- %.2f\n' % (input_spikes_mean, input_spikes_std)
label_text += 'nspikes_in_non-zero mean rate = %.2f +- %.2f [Hz]\n' % (input_spikes_mean / t_stim, input_spikes_std / t_stim)
label_text += 'g_in_sum = %d [uS]\n' % (g_in_all.sum())
label_text += 'g_in_mean = %.2f +- %.2f [uS]\n' % (g_in_all.mean(), g_in_all.std())
label_text += 'g_in_nonzero_mean = %.2f +- %.2f [uS]\n' % (g_in.mean(), g_in.std())
label_text += 'g_in_nonzero_mean / t_stim = %.2f +- %.2f [nS / ms]\n' % (g_in.mean() / t_stim, g_in.std() / t_stim)
label_text += 'g_in_top_%d_percent / t_stim = %.2f +- %.2f [nS / ms]\n' % (pmax*100, g_in_sorted[-n_:].mean() / t_stim, g_in_sorted[-n_:].std() / t_stim)
bar = ax1.bar(x, sorted_input_spikes)

ax1.set_xlabel('Cells sorted by num input spikes')
ax1.set_ylabel('Number of input spikes')
ax1.set_xlim((0, params['n_exc']+1))
#ax1.set_ylim((0, 150))
#ax2.set_ylim((0, 3))
ax2.set_ylabel('Input conductance [uS]')

title = 'Input parameters:\n blur_x(v)=%.1e (%.1e)\n f=%d Hz w=%.1e uS' % (params['blur_X'], params['blur_V'], params['f_max_stim'], params['w_input_exc'])
ax1.set_title(title)
pylab.subplots_adjust(top=0.85)

(text_pos_x, text_pos_y) = ax2.get_xlim()[1] * 0.03, ax2.get_ylim()[1] * 0.55
print 'Info:',label_text
pylab.text(text_pos_x, text_pos_y, label_text, bbox=dict(pad=5.0, ec="k", fc="none"))


#output_fig = params['figures_folder'] + 'input_analysis.png'
#print 'Output fig:', output_fig
#output_fig = 'Figures_BlurSweep/' + 'fin%d_w%.1e_blurXV%.1e_%.1e.png' % (params['f_max_stim'], params['w_input_exc'], params['blur_X'], params['blur_V'])

print 'Saving to:', output_fig
pylab.savefig(output_fig)
output_fig = params['figures_folder']

#output_fig = 'Figures_BlurSweep/' + '%d.png' % (file_count)

# only needed when a sweep is done
output_fn = 'Figures_BlurSweep/nspikes_blur_sweep_new_unscaled.dat'
output_file = open(output_fn, 'a')
output_string = '%.2e\t%.2e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%d\n' % (params['blur_X'], params['blur_V'], all_spikes.sum(), all_spikes.mean(), all_spikes.std(), input_spikes_mean, input_spikes_std, input_spikes_max)
#   0                       1               2               3                   4               5                   5                   6
#(params['blur_X'], params['blur_V'], all_spikes.sum(), all_spikes.mean(), all_spikes.std(), input_spikes_mean, input_spikes_std, input_spikes_max)
output_file.write(output_string)
output_file.close()
#print 'Saving to:', output_fig
pylab.savefig(output_fig)
#pylab.show()
