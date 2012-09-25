import pylab
import numpy as np
#import sys
#import os
#import re
import simulation_parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary

# TODO: make a class out of this messy script ... 

        # ---> time dependent values
        # speed arrays, containing confidence of certain speed over time
        vx_prediction_binwise = np.zeros((n_cells, n_bins))     # _binwise: momentary prediction based on global activity within one single time bin
        vx_prediction_trace = np.zeros((n_cells, n_bins, 2))    # _trace: prediction based on the momentary and past activity (moving average, and std) --> trace_length
        vy_prediction_binwise = np.zeros((n_cells, n_bins))     # speed in y-direction
        vy_prediction_trace = np.zeros((n_cells, n_bins, 2))    #

        # angle of motion based the different 'voting schemes'
        theta_prediction_binwise = np.zeros(n_bins)        # angle of speed theta = arctan (v_y / v_x) over time
        theta_prediction_trace = np.zeros((n_bins, 2))

        # ---> time INdependent values: 
        vx_prediction_fullrun= np.zeros(n_cells)                # time independent prediction based on the whole run --> voting histogram
        vx_prediction_nonlinear = np.zeros(n_cells)             # prediction based on non-linear transformation of output rates
        vy_prediction_fullrun= np.zeros(n_cells)                # same for v_y
        vy_prediction_nonlinear = np.zeros(n_cells)             # 
        theta_prediction_fullrun = np.zeros(n_cells)            # using normalized rates over the whole run
        theta_prediction_nonlinear = np.zeros(n_cells)          # nonlinear transformation of normalized rates

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Get data, compute activity over time, normalization, ...
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        """
        # # # # # # # # # # # # # # # # # # # # # # 
        # S P E E D    P R E D I C T I O N 
        # # # # # # # # # # # # # # # # # # # # # # 
         Compute speed prediction based on global information (from the whole population)

         On which time scale shall the prediction work?
         There are (at least) 3 different ways to do it:
           Very short time-scale:
           1) Compute the prediction for each time bin - based on the activitiy in the respective time bin 
           Short time-scale:
           2) Compute the prediction for each time bin based on all activity in the past
           Long time-scale:
           3) Compute the prediction based on the the activity of the whole run - not time dependent
           4) Non-linear 'voting' based on 3) 
        """

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Compute 'voting' results for time bins: single neuron level
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        for i in xrange(int(n_bins)): # i --> time
            # normalized output rate
            # 1) momentary vote
            vx_prediction_binwise[:, i] = normed_activity[sorted_indices_x, i]
            vy_prediction_binwise[:, i] = normed_activity[sorted_indices_y, i]
            past_bin = min(0, i - trace_length_in_bins)
            # 2) moving average
            vx_prediction_trace[:, i, 0] = normed_activity[sorted_indices_x, i-past_bin:i].mean()
            vx_prediction_trace[:, i, 1] = normed_activity[sorted_indices_x, i-past_bin:i].std()
            vy_prediction_trace[:, i, 0] = normed_activity[sorted_indices_x, i-past_bin:i].mean()
            vy_prediction_trace[:, i, 1] = normed_activity[sorted_indices_x, i-past_bin:i].std()

        # in the first step the trace can not have a standard deviation --> avoid NANs 
        vx_prediction_trace[:, 0, 1] = 0
        vy_prediction_trace[:, 0, 1] = 0

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Compute 'voting' results on the population neuron level --> compute theta
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # 3) integral over full run
        for cell in xrange(n_cells):
            vx_prediction_fullrun[cell] = spike_count[sorted_indices_x[cell]] / spike_count.sum()
            vy_prediction_fullrun[cell] = spike_count[sorted_indices_y[cell]] / spike_count.sum()

        # 4) non linear transformation
        exp_nspike_sum = np.exp(vx_prediction_fullrun - spike_count.max()).sum()
        nspikes_max = spike_count.max()
        for cell in xrange(n_cells):
            nspikes = vx_prediction_fullrun[cell]
            p_i_x = np.exp(nspikes - nspikes_max) / exp_nspike_sum 
            vx_prediction_nonlinear[cell] = p_i_x

            nspikes = vy_prediction_fullrun[cell]
            p_i_y = np.exp(nspikes - nspikes_max) / exp_nspike_sum 
            vy_prediction_nonlinear[cell] = p_i_y

        #speed_output = np.zeros(n_bins)
        avg_vx = np.zeros(n_bins) # momentary result, based on the activity in one time bin
        avg_vy = np.zeros(n_bins)
        # ---> gives theta_prediction_binwise

        moving_avg_vx = np.zeros((n_bins, 2))
        moving_avg_vy = np.zeros((n_bins, 2))
        # ---> gives theta_prediction_trace

        for i in xrange(int(n_bins)):
            # take the weighted average of the vx_prediction (weight = normalized activity)
            vx_pred = vx_prediction_binwise[:, i] * vx_tuning
            avg_vx[i] = np.sum(vx_pred)
            moving_avg_vx[i, 0] = np.sum(vx_prediction_trace[:, i, 0] * vx_tuning)
            moving_avg_vx[i, 1] = np.sum(vx_prediction_trace[:, i, 1] * vx_tuning)

            vy_pred = vy_prediction_binwise[:, i] * vy_tuning
            avg_vy[i] = np.sum(vy_pred)
            moving_avg_vy[i, 0] = np.sum(vy_prediction_trace[:, i, 0] * vy_tuning)
            moving_avg_vy[i, 1] = np.sum(vy_prediction_trace[:, i, 1] * vy_tuning)

        # ---> theta calculations
        theta_prediction_binwise = arctan2(avg_vy, avg_vx)

        # calculate theta based on the different 'voting' schemes
        theta_prediction_trace[:, 0] = np.arctan2(moving_avg_vy[:, 0], moving_avg_vx[:, 0])
        # error for theta is based on propagation of uncertainty of vx and vy
        theta_prediction_trace[:, 1] = theta_uncertainty(moving_avg_vx[:, 0], moving_avg_vx[:, 1], moving_avg_vy[:, 0], moving_avg_vy[:, 1])

        theta_prediction_binwise = np.arctan2(avg_vy, avg_vx)

        # compute the certainty for the theta votes as average from the vx and vy votes
        theta_prediction_fullrun = .5 * (vy_rate_vote_normalized[:, 0] + vx_rate_vote_normalized[:, 0])
        assert (theta_prediction_rate.sum() == 1), "Normalized theta vote based on output rates is not = 1, %f" % (theta_prediction_rate.sum())
        theta_prediction_nonlinear = .5(
        theta = np.arctan2(vy_tuning, vx_tuning)
            
        # # # # # # # # # # # # # # # # # # # # # # 
        # P L O T T I N G     F I G U R E  1
        # # # # # # # # # # # # # # # # # # # # # # 
        if False:
            ax2 = fig1.add_subplot(322)
            ax2.set_title('Normalized spiking activity over time')
            cax2 = ax2.pcolor(normed_activity)
            ax2.set_ylim((0, normed_activity[:, 0].size))
            ax2.set_xlim((0, normed_activity[0, :].size))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('GID')
            pylab.colorbar(cax2)

            ax3 = fig1.add_subplot(323)
            ax3.set_title('vx predictions:\nvx on y-axis, color=confidence')
            cax3 = ax3.pcolor(vx_prediction_binwise)#, edgecolor='k', linewidths='1')
            ax3.set_ylim((0, vx_prediction_binwise[:, 0].size))
            ax3.set_xlim((0, vx_prediction_binwise[0, :].size))
            ny = vx_tuning.size
            n_ticks = 5
            yticks = [vx_tuning[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
            ylabels = ['%.1e' % i for i in yticks]
            ax3.set_yticks([int(i * ny/n_ticks) for i in xrange(n_ticks)])
            ax3.set_yticklabels(ylabels)
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Velocity')
            pylab.colorbar(cax3)

            ax4 = fig1.add_subplot(324)
            ax4.set_title('vy predictions:\nvy on y-axis, color=confidence')
            cax4 = ax4.pcolor(vy_prediction_binwise)#, edgecolor='k', linewidths='1')
            ax4.set_ylim((0, vy_prediction_binwise[:, 0].size))
            ax4.set_xlim((0, vy_prediction_binwise[0, :].size))
            ny = vy_tuning.size
            n_ticks = 5
            yticks = [vy_tuning[int(i * ny/n_ticks)] for i in xrange(n_ticks)]
            ylabels = ['%.1e' % i for i in yticks]
            ax4.set_yticks([int(i * ny/n_ticks) for i in xrange(n_ticks)])
            ax4.set_yticklabels(ylabels)
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Velocity')
            pylab.colorbar(cax4)


            t_axis = np.arange(0, n_bins * time_binsize, time_binsize)
            ax5 = fig1.add_subplot(325)
            ax5.set_title('Predicted resulting vx')
            ax5.plot(t_axis, avg_vx)
            ax5.errorbar(t_axis, moving_avg_vx[:, 0], yerr=moving_avg_vx[:, 1])
            ax5.set_xlabel('Time (ms)')
            ax5.set_ylabel('velocity')
            pylab.show()

            ax5 = fig1.add_subplot(326)
            ax5.set_title('Predicted resulting vy')
            ax5.plot(t_axis, avg_vy)
            ax5.errorbar(t_axis, moving_avg_vy[:, 0], yerr=moving_avg_vy[:, 1])
            ax5.set_xlabel('Time (ms)')
            ax5.set_ylabel('velocity')
            pylab.show()

        # # # # # # # # # # # # # # # # # # # # # # 
        # P L O T T I N G     F I G U R E  2
        # # # # # # # # # # # # # # # # # # # # # # 
        fig2 = pylab.figure()

        ax1 = fig2.add_subplot(231)
        ax1.set_title('Vx rate vote over whole run')
        ax1.bar(vx_tuning, vx_rate_vote[:, 0], yerr= vx_rate_vote[:, 1])

        ax2 = fig2.add_subplot(231)
        ax2.set_title('vy rate vote over whole run')
        ax2.bar(vy_tuning, vy_rate_vote[:, 0], yerr= vy_rate_vote[:, 1])

        ax3 = fig2.add_subplot(231)
        #ax3.set_title('theta rate vs time')
        ax3.plot(t_axis, theta_prediction_binwise)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('$\theta$')


        #bbax=ax2.get_position()
        #posax = bbax.get_points()
        #print "ax pos:", posax
        #x0 = posax[0][0] + 0.1
        #x1 = posax[1][0] + 0.1
        #y0 = posax[0][1] - 0.1
        #y1 = posax[1][1] - 0.1
