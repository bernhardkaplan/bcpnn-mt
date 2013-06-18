# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:58:11 2013

@author: brain
"""

# to study anticipatory response of network
# we need to filter neurons and select the ones that are in the trajectory (choosing their y, if that is not enough to filter by their u and v)
# then to record those target cells and also illustrate the location of stimulus

#first in simulation_parameters: t_blank and t_start = 0,0 as we want to study motion anticipation in afull trajectory
#problem: if t_blank means blank duration? why I can not change it in simulation parameters? 

import utils as u
import numpy as np
import pylab
import matplotlib.pyplot as plt

import simulation_parameters as sp
#import plot_rasterplot as raster_plot
import plot_voltage_noCompatibleOutput as voltage


network_params = sp.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()
ex_cells = params['n_exc']
#idx, dist = u.sort_cells_by_distance_to_stimulus(ex_cells)
#
##to illustrate gids and cell distance to the stimulus
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(range(ex_cells),dist,lw=3,linestyle = '', marker = 'o')
#xticks = []
#for gid in idx:   
#    if (np.mod(idx.tolist().index(gid),100 ) == 0):
#        xticks.append(gid)
#print xticks
#yticks = np.arange(0,np.ceil(np.max(dist)))
#ax1.set_xticklabels(['%d' % i for i in xticks], fontsize =14)
#ax1.set_yticks(yticks)
#ax1.set_yticklabels(['%d' % i for i in yticks], fontsize =14)
#ax1.set_xlabel('Cell gid', fontsize = 14)
#ax1.set_ylabel('Distance from stimulus', fontsize = 14)
#
#output_fn = params['figures_folder'] + 'Cells sorted by distance'
#print "Saving to", output_fn
#plt.savefig(output_fn)
#
#
##to choose closer cells to the stimulus  (distance <2.0)
#fig = plt.figure()
#ax2 = fig.add_subplot(111)
#treshold = dist[np.where(np.floor(dist)== 2)]
#last_index = dist.tolist().index(treshold[0])
#shorter_dist = dist.tolist()[0:last_index]
#new_idx = idx.tolist()[0:last_index]
#ax2.plot(range(len(new_idx)),shorter_dist,lw=3,linestyle = '', marker = 'o')
#xticks = []
#for gid in new_idx:   
#    if (np.mod(new_idx.index(gid),100 ) == 0):
#        xticks.append(gid)
#yticks = np.arange(0,np.ceil(np.max(shorter_dist)))
#ax2.set_xticklabels(['%d' % i for i in xticks], fontsize =14)
#ax2.set_yticks(yticks)
#ax2.set_yticklabels(['%d' % i for i in yticks], fontsize =14)
#ax2.set_xlabel('Cell gid', fontsize = 14)
#ax2.set_ylabel('Distance from stimulus', fontsize = 14)
#
#output_fn = params['figures_folder'] + 'Cells sorted by short distance'
#print "Saving to", output_fn
#plt.savefig(output_fn)
#
## now the prefered y of selected_gids should be checked
## to load tuning properties of cells
tp = np.loadtxt(params['tuning_prop_means_fn'])
selected_gids = []
cells = np.arange(ex_cells)
for cell in cells:
    i = cells.tolist().index(cell)
    if (tp[i,1]>0.3 and tp[i,1]<0.7):
        selected_gids.append(i)
#selected_gids contains gids of cells which their prefered y is adjacent to y range of stimulus.        
   



selected_y = np.zeros(len(selected_gids))
selected_x = np.zeros(len(selected_gids))

# to get tuning properties of selected_cells
for gid in selected_gids:
    i = selected_gids.index(gid)
#    print gid, i

    current_tp = tp[gid]
    selected_y[i] = current_tp[1]
    selected_x[i] = current_tp[0]
    

fig = plt.figure()
ax3 = fig.add_subplot(111)
ax3.plot(range(len(selected_gids)),selected_y,lw=3,linestyle = '', marker = 'o')
xticks = []
for gid in selected_gids:   
    if (np.mod(selected_gids.index(gid),100 ) == 0):
        xticks.append(gid)
yticks = np.arange(np.floor(np.min(selected_y)),np.ceil(np.max(selected_y)))
ax3.set_xticklabels(['%d' % i for i in xticks], fontsize =14)
#ax3.set_yticks(yticks)
#ax3.set_yticklabels(['%d' % i for i in yticks], fontsize =14)
ax3.set_xlabel('Cell gid', fontsize = 14)
ax3.set_ylabel('Prefered y', fontsize = 14)

output_fn = params['figures_folder'] + 'Cells sorted by y'
print "Saving to", output_fn
plt.savefig(output_fn)



fig = plt.figure()
ax4 = fig.add_subplot(111)
ax4.plot(range(len(selected_gids)),selected_x,lw=3,linestyle = '', marker = 'o')
xticks = []
for gid in selected_gids:   
    if (np.mod(selected_gids.index(gid),100 ) == 0):
        xticks.append(gid)
yticks = np.arange(np.floor(np.min(selected_x)),np.ceil(np.max(selected_x)))
ax4.set_xticklabels(['%d' % i for i in xticks], fontsize =14)
ax4.set_yticks(yticks)
ax4.set_yticklabels(['%d' % i for i in yticks], fontsize =14)
ax4.set_xlabel('Cell gid', fontsize = 14)
ax4.set_ylabel('Prefered x', fontsize = 14)
output_fn = params['figures_folder'] + 'Cells sorted by x'
print "Saving to", output_fn
plt.savefig(output_fn)
#print len(selected_gids)


n_to_plot = len(selected_gids)
fn = params['exc_volt_fn_base'] + '.v'
#gids = np.loadtxt(params['gids_to_record_fn'])
selected_gids = u.all_anticipatory_gids(params)
pylab.figure()
voltage.plot_volt(fn, gid = selected_gids, n = n_to_plot)

pops = u.pop_anticipatory_gids(params)
#pylab.show()

for pop in pops: 
#    print pop
#    print pops.index(pop)
    volt = 0
    d = np.loadtxt(params['exc_volt_anticipation'])
    for gid in gids:
        time_axis, volt = extract_trace(d,gid)
        volt += volt
    avearged_voltage = volt.mean()
    pylab.figure()
    pylab.plot(time_axis, avergaed_voltage)

    
#pylab.xlabel('Time [ms]')
#pylab.ylabel('Voltage [mV]')

    pylab.show()

#    plot_volt(fn, n=5)
#    plot_volt(fn, 'all')




