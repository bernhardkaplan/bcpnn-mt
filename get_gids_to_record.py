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

indices, distances = utils.sort_gids_by_distance_to_stimulus(tp , mp)
# cells in indices should have the highest response to the stimulus

conn_mat, delays = utils.convert_connlist_to_matrix(params['conn_list_ee_fn_base'] + '0.dat', params['n_exc'])
n = 20


#try:

print "Loading nspikes", params['exc_spiketimes_fn_merged'] + '0.ras'
nspikes = utils.get_nspikes(params['exc_spiketimes_fn_merged'] + '0.ras', n_cells=params['n_exc'])
got_spikes = True
mans = (nspikes.argsort()).tolist()
mans.reverse()
mans = mans[0:n]
print "Max nspikes fired:", nspikes[mans], mans

#    print "Cell %d has tuning_prop:" % nspikes.argmax(), tp[nspikes.argmax(), :]
#except:
#    print "No nspikes"
#    got_spikes = False
#    nspikes = np.zeros(params['n_exc'])
#    pass


# get the tuning properties of the n cells being closest to the stimulus
print "\n'Good' tuning properties (they should lead to a strong response to the stimulus)"
good_tp = tp[indices[0:n], :]
print 'gid\tx_pos\t\ty_pos\tvx\t\tvy'
for i in xrange(n):
    print indices[i], tp[indices[i], :]
# sort them according to their x-pos
x_sorted_indices = np.argsort(good_tp[:, 0])
gids = indices[x_sorted_indices]

#indices_list = list(indices)

#print good_tp[x_sorted_indices, :]

print "\nGID\tnspikes\tdistance_to_stim\tp and weight_to_next_cell that will see the stimulus (and weight back)\tx_pos"

for i in xrange(n-1):

    src = gids[i]
    tgt = gids[i+1]
    p, latency = CC.get_p_conn(tp, src, tgt, sigma_x, sigma_v)
    print gids[i], '\t', nspikes[gids[i]], '\t', distances[x_sorted_indices[i]], '\t\t', p, '\t', conn_mat[gids[i], gids[i+1]], '\t\t', \
            conn_mat[gids[i+1], gids[i]], '\t', good_tp[x_sorted_indices[i], 0]


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


