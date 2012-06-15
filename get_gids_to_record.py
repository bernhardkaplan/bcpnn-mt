import numpy as np
import utils
import simulation_parameters
import CreateConnections as CC

network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tp = np.loadtxt(params['tuning_prop_means_fn'])

mp = params['motion_params']
print "Motion parameters", mp
sigma_x, sigma_v = params['w_sigma_x'], params['w_sigma_v'] # small sigma values let p and w shrink

print 'utils.sort_gids_by_distance_to_stimulus...'
indices, distances = utils.sort_gids_by_distance_to_stimulus(tp , mp) # cells in indices should have the highest response to the stimulus
print 'utils.convert_connlist_to_matrix...'
conn_mat, delays = utils.convert_connlist_to_matrix(params['conn_list_ee_fn_base'] + '0.dat', params['n_exc'])
#n = 50
n = int(params['n_exc'] * .05) # fraction of 'interesting' cells

print "Loading nspikes", params['exc_spiketimes_fn_merged'] + '0.ras'
nspikes = utils.get_nspikes(params['exc_spiketimes_fn_merged'] + '0.ras', n_cells=params['n_exc'])
mans = (nspikes.argsort()).tolist() # mans = most active neurons
mans.reverse()
mans = mans[0:n]

print 'w_out_good = weights to other well tuned neurons'
print 'w_in_good = weights from other well tuned neurons'
print 'w_out_total = sum of all outgoing weights'
print 'w_in_total = sum of all incoming weights'
print 'distance_to_stim = minimal distance to the moving stimulus (linear sum of  (time-varying) spatial distance and (constant) distance in velocity)'
print "\nGID\tnspikes\tdistance_to_stim\tw_out_good\tw_in_good\tw_out_total\tw_in_total\ttuning_prop"
for i in xrange(n):
    gid = indices[i]
    other_gids = list(indices)
    other_gids.remove(gid)
    w_in_good = conn_mat[other_gids, gid].sum()
    w_out_good = conn_mat[gid, other_gids].sum()
    w_in_sum = conn_mat[:, gid].sum()
    w_out_sum = conn_mat[gid, :].sum()
    distance_to_stim = utils.get_min_distance_to_stim(mp, tp[gid, :])
    print '%d\t%d\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e' % (gid, nspikes[gid], distance_to_stim, w_out_good, w_in_good, w_out_sum, w_in_sum), tp[gid, :]
#    print '%d\t%d\t%.3e\t%.3e\t%.3e' % (gid, nspikes[gid], distance_to_stim, w_in_good, w_in_sum), tp[gid, :]

print '\n Look at the most active neurons in the network'
print 'w_out = weights to other mans (MostActiveNeuronS)'
print 'w_in = weights from other mans (MostActiveNeuronS)'
print 'w_out_total = sum of all outgoing weights'
print 'w_in_total = sum of all incoming weights'
print 'distance_to_stim = minimal distance to the moving stimulus (linear sum of  (time-varying) spatial distance and (constant) distance in velocity)'
print "\nGID\tnspikes\tdistance_to_stim\tw_out\tw_in\tw_out_total\tw_in_total\ttuning_prop"
for i in xrange(n):
    gid = mans[i]
    other_gids = list(mans)
    other_gids.remove(gid)
    w_in_good = conn_mat[other_gids, gid].sum()
    w_out_good = conn_mat[gid, other_gids].sum()
    w_in_sum = conn_mat[:, gid].sum()
    w_out_sum = conn_mat[gid, :].sum()
    distance_to_stim = utils.get_min_distance_to_stim(mp, tp[gid, :])
    print '%d\t%d\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e' % (gid, nspikes[gid], distance_to_stim, w_out_good, w_in_good, w_out_sum, w_in_sum), tp[gid, :]


print 'Overlap between most active neurons and well-tuned neurons:'
mans_set = set(mans)
good_gids = set(indices)
print mans_set.intersection(good_gids)

# sort them according to their x-pos
x_sorted_indices = np.argsort(tp[indices, 0])
sorted_gids = indices[x_sorted_indices]


print '\nConnection probabilities between the cells with \'good\' tuning properies:'
conn_probs = []
latencies = []
for i in xrange(n):
    src = sorted_gids[i]
    for tgt in sorted_gids:
        p, latency = CC.get_p_conn(tp, src, tgt, sigma_x, sigma_v)
        conn_probs.append(p)
        latencies.append(latency)
#        print "p(%d, %d):\t %.3e\tlatency: %.3e\t" % (src, tgt, p, latency)
#    print '\n'

conn_probs = np.array(conn_probs)
latencies = np.array(latencies)
print "Average connection probability between neurons with \'good\' tuning_prop", np.mean(conn_probs), '\t std:', np.std(conn_probs)
print "Min max and median connection probability between neurons with \'good\' tuning_prop", np.min(conn_probs), np.max(conn_probs), np.median(conn_probs)
print "Average latencies between neurons with \'good\' tuning_prop", np.mean(latencies), '\t std:', np.std(latencies)
print "Min and max latencies between neurons with \'good\' tuning_prop", np.min(latencies), np.max(latencies)

print "\nCompare to cells connected via strong weights"
all_weights = conn_mat.flatten()
sorted_weight_indices = list(all_weights.argsort())
#print "sorted_weight_indices", sorted_weight_indices[0:n]
sorted_weight_indices.reverse(), sorted_weight_indices[0:n]
n_strongest_weights = sorted_weight_indices[0:n]
#n_strongest_weights = list(conn_mat.argsort()).reverse()[0:n]
print "src\t tgt \t weight\t\t tp[src] \t\t\t\t\t tp[tgt] \tindex_in_distance list\t\tdistance_to_stimulus[src]"
for i in xrange(n):
    i_, j_ = n_strongest_weights[i] / params['n_exc'], n_strongest_weights[i] % params['n_exc']
    print i_, '\t', j_, '\t', conn_mat[i_, j_], tp[i_, :], '\t', tp[j_, :], '\t', indices.tolist().index(i_), '\t', distances[indices.tolist().index(i_)]


