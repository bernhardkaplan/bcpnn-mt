
import os
import sys
import numpy as np
import utils
import json
import pylab

if __name__ == '__main__':

    folder_name = sys.argv[1]
    params = utils.load_params(folder_name)
    taui_ampa = 5
    taui_nmda = 150
    conn_fn_ampa = 'TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui%d_nHC20_nMC4_vtrain1.00-1.0/Connections/conn_matrix_mc.dat' % (taui_ampa)
    conn_fn_nmda = 'TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui%d_nHC20_nMC4_vtrain1.00-1.0/Connections/conn_matrix_mc.dat' % (taui_nmda)
    W_ampa = np.loadtxt(conn_fn_ampa)
    W_nmda = np.loadtxt(conn_fn_nmda)

    g_in_ampa_pos_target = 3.
    g_in_ampa_neg_target = -1.
    g_in_nmda_pos_target = 1.
    g_in_nmda_neg_target = -1.

    # for normalizing the relative componentes of AMPA / NMDA
    W_ampa_pos = np.zeros(W_ampa.shape)
    W_ampa_neg = np.zeros(W_ampa.shape)
    W_nmda_pos = np.zeros(W_nmda.shape)
    W_nmda_neg = np.zeros(W_nmda.shape)
    w_in_ampa_pos_sum = np.zeros(params['n_mc'])
    w_in_ampa_neg_sum = np.zeros(params['n_mc'])
    w_in_nmda_pos_sum = np.zeros(params['n_mc'])
    w_in_nmda_neg_sum = np.zeros(params['n_mc'])


    gain_ampa_pos = np.zeros(params['n_mc'])
    gain_ampa_neg = np.zeros(params['n_mc'])
    gain_nmda_pos = np.zeros(params['n_mc'])
    gain_nmda_neg = np.zeros(params['n_mc'])

    # making sure the incoming pos / neg weights are normalized as well
    g_exc_total_target = .8
    g_inh_total_target = -2.
    gamma_pos = np.zeros(params['n_mc'])
    gamma_neg = np.zeros(params['n_mc'])

    print '\t\t\tg_in_ampa_pos_target g_in_ampa_neg_target g_in_nmda_pos_target g_in_nmda_neg_target'

    for tgt_hc in xrange(params['n_hc']):
        for tgt_mc in xrange(params['n_mc_per_hc']):
            tgt_pop_idx = tgt_hc * params['n_mc_per_hc'] + tgt_mc
            pos_idx = np.where(W_ampa[:, tgt_pop_idx] > 0)[0]
            neg_idx = np.where(W_ampa[:, tgt_pop_idx] < 0)[0]
            print 'debug w pos in ampa', W_ampa[pos_idx, tgt_pop_idx].sum() 
            print 'debug w neg in ampa', W_ampa[neg_idx, tgt_pop_idx].sum()
            print 'debug w pos in nmda', W_nmda[pos_idx, tgt_pop_idx].sum()
            print 'debug w neg in nmda', W_nmda[neg_idx, tgt_pop_idx].sum()

            g_in_ampa_pos = W_ampa[pos_idx, tgt_pop_idx].sum()
            g_in_ampa_neg = W_ampa[neg_idx, tgt_pop_idx].sum()
            pos_idx = np.where(W_nmda[:, tgt_pop_idx] > 0)[0]
            neg_idx = np.where(W_nmda[:, tgt_pop_idx] < 0)[0]
            g_in_nmda_pos = W_nmda[pos_idx, tgt_pop_idx].sum()
            g_in_nmda_neg = W_nmda[neg_idx, tgt_pop_idx].sum()

            # normalize the relative contributions 
            gain_ampa_pos[tgt_pop_idx] = g_in_ampa_pos_target / g_in_ampa_pos 
            gain_ampa_neg[tgt_pop_idx] = g_in_ampa_neg_target / g_in_ampa_neg 
            gain_nmda_pos[tgt_pop_idx] = g_in_nmda_pos_target / g_in_nmda_pos 
            gain_nmda_neg[tgt_pop_idx] = g_in_nmda_neg_target / g_in_nmda_neg 

            # normalize the total positive incoming components 
            gamma_pos[tgt_pop_idx] = g_exc_total_target / (g_in_nmda_pos * gain_nmda_pos[tgt_pop_idx] + g_in_ampa_pos * gain_ampa_pos[tgt_pop_idx])
            gamma_neg[tgt_pop_idx] = g_inh_total_target / (g_in_nmda_neg * gain_nmda_neg[tgt_pop_idx] + g_in_ampa_neg * gain_ampa_neg[tgt_pop_idx])

#            gamma_pos[tgt_pop_idx] = g_exc_total_target / (gain_nmda_pos[tgt_pop_idx] + gain_ampa_pos[tgt_pop_idx])
#            gamma_neg[tgt_pop_idx] = g_inh_total_target / (gain_nmda_neg[tgt_pop_idx] + gain_ampa_neg[tgt_pop_idx])

    for src_hc in xrange(params['n_hc']):
        for src_mc in xrange(params['n_mc_per_hc']):
#            src_pop = list_of_exc_pop[src_hc][src_mc]
            src_pop_idx = src_hc * params['n_mc_per_hc'] + src_mc
            for tgt_hc in xrange(params['n_hc']):
                for tgt_mc in xrange(params['n_mc_per_hc']):
                    tgt_pop_idx = tgt_hc * params['n_mc_per_hc'] + tgt_mc

                    w_ampa = W_ampa[src_pop_idx, tgt_pop_idx]
                    w_nmda = W_nmda[src_pop_idx, tgt_pop_idx]

                    if w_ampa > 0:
                        w_ampa_pos = w_ampa * gain_ampa_pos[tgt_pop_idx] * gamma_pos[tgt_pop_idx]
                        W_ampa_pos[src_pop_idx, tgt_pop_idx] = w_ampa_pos
                    else:
                        w_ampa_neg = w_ampa * gain_ampa_neg[tgt_pop_idx] * gamma_neg[tgt_pop_idx]
                        W_ampa_neg[src_pop_idx, tgt_pop_idx] = w_ampa_neg
                    if w_nmda > 0:
                        w_nmda_pos = w_nmda * gain_nmda_pos[tgt_pop_idx] * gamma_pos[tgt_pop_idx]
                        W_nmda_pos[src_pop_idx, tgt_pop_idx] = w_nmda_pos
                    else:
                        w_nmda_neg = w_nmda * gain_nmda_neg[tgt_pop_idx] * gamma_neg[tgt_pop_idx]
                        W_nmda_neg[src_pop_idx, tgt_pop_idx] = w_nmda_neg


    # check after setting the scaled weights what the normalized values are
    for tgt_hc in xrange(params['n_hc']):
        for tgt_mc in xrange(params['n_mc_per_hc']):
            tgt_pop_idx = tgt_hc * params['n_mc_per_hc'] + tgt_mc
            w_in_ampa_pos_sum[tgt_pop_idx] = W_ampa_pos[:, tgt_pop_idx].sum()
            w_in_ampa_neg_sum[tgt_pop_idx] = W_ampa_neg[:, tgt_pop_idx].sum()
            w_in_nmda_pos_sum[tgt_pop_idx] = W_nmda_pos[:, tgt_pop_idx].sum()
            w_in_nmda_neg_sum[tgt_pop_idx] = W_nmda_neg[:, tgt_pop_idx].sum()

#                    print 'src hc mc %d %d\ttgt hc mc %d %d\t' % (src_hc, src_mc, tgt_hc, tgt_mc), ampa_gain_pos, ampa_gain_neg, nmda_gain_pos, nmda_gain_neg, w_ampa_, w_ampa_neg, w_nmda_, w_nmda_neg

    fn_out_ampa_pos = 'w_ampa_pos_normalized.dat'
    print fn_out_ampa_pos
    np.savetxt(fn_out_ampa_pos, W_ampa_pos)
    fn_out_ampa_neg = 'w_ampa_neg_normalized.dat'
    print fn_out_ampa_neg
    np.savetxt(fn_out_ampa_neg, W_ampa_neg)

    fn_out_nmda_pos = 'w_nmda_pos_normalized.dat'
    print fn_out_nmda_pos
    np.savetxt(fn_out_nmda_pos, W_nmda_pos)
    fn_out_nmda_neg = 'w_nmda_neg_normalized.dat'
    print fn_out_nmda_neg
    np.savetxt(fn_out_nmda_neg, W_nmda_neg)

    fig = pylab.figure()
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
#    ax5 = fig.add_subplot(325)
#    ax6 = fig.add_subplot(326)
    ax1.bar(range(params['n_mc']), w_in_ampa_pos_sum, width=1)
    ax2.bar(range(params['n_mc']), w_in_ampa_neg_sum, width=1)
    ax3.bar(range(params['n_mc']), w_in_nmda_pos_sum, width=1)
    ax4.bar(range(params['n_mc']), w_in_nmda_neg_sum, width=1)
    ax1.set_ylabel('w_in_ampa_pos_sum')
    ax2.set_ylabel('w_in_ampa_neg_sum')
    ax3.set_ylabel('w_in_nmda_pos_sum')
    ax4.set_ylabel('w_in_nmda_neg_sum')
    

    pylab.show()

