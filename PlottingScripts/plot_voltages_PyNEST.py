import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import random
import numpy as np
import pylab
import utils
import re


def plot_volt(fn, gid=None, n=1):
    print 'Loading fn:', fn
    d = np.loadtxt(fn)
    if not d.size > 0:
        print 'No data found in:', fn
        return
    else:
        print 'Plotting data from:', fn

    if gid == None:
        recorded_gids = np.unique(d[:, 0])
        gids = random.sample(recorded_gids, n)
        print 'plotting random gids:', gids
    elif gid == 'all':
        gids = np.unique(d[:, 0])
    elif type(gid) == type([]):
        gids = gid
    else:
        gids = [gid]
    
    for gid in gids:
        time_axis, volt = utils.extract_trace(d, gid)
#        print 'debug gid volt size', gid, volt.size
        if volt.size > 0:
            pylab.plot(time_axis, volt, label='%d' % gid, lw=2)

    parts = fn.rsplit('.')
    output_fn = "" 
    for i in xrange(len(parts)-1):
        output_fn += "%s" % parts[i] + '.'
    output_fn += 'png'
    pylab.legend()
    pylab.title(fn)
#    print 'Saving to', output_fn
#    pylab.savefig(output_fn)


def plot_average_volt(fn, gid=None, n=1):
    print 'Plotting average voltage; loading', fn
    d = np.loadtxt(fn)
    if gid == None:
        gid_range = np.unique(d[:, 0])
        gids = np.random.randint(np.min(gid_range), np.max(gid_range) + 1, n)
        print 'plotting random gids:', gids
    elif gid == 'all':
        gids = np.unique(d[:, 0])
    elif type(gid) == type([]):
        gids = gid
    else:
        gids = [gid]
    
    time_axis, volt = utils.extract_trace(d, gids[0])
    all_volt = np.zeros((time_axis.size, len(gids)))

    for i_, gid in enumerate(gids):
        time_axis, volt = utils.extract_trace(d, gid)
        print 'gid %d v_mean, std = %.2f +- %.2f; min %.2f max %.2f, diff %.2f ' % (gid, volt.mean(), volt.std(), volt.min(), volt.max(), volt.max() - volt.min())
        all_volt[:, i_] = volt

    avg_volt = np.zeros((time_axis.size, 2))
    for t in xrange(time_axis.size):
        avg_volt[t, 0] = all_volt[t, :].mean()
        avg_volt[t, 1] = all_volt[t, :].std()

    print 'Average voltage and std: %.2e +- %.2e (%.2e)' % (avg_volt[:, 0].mean(), avg_volt[:, 0].std(), avg_volt[:, 1].mean())
    pylab.errorbar(time_axis, avg_volt[:, 0], yerr=avg_volt[:, 1], lw=3, c='k') 


if __name__ == '__main__':

    try: 
        params_fn = sys.argv[1]
        if os.path.isdir(params_fn):
            params_fn += '/Parameters/simulation_parameters.json'
        f = file(params_fn, 'r')
        params = json.load(f)
    except:
        import simulation_parameters
        ps = simulation_parameters.parameter_storage()
        params = ps.params
        params_fn = params['params_fn_json']


#    print sys.argv
#    try: 
#        cell_type = sys.argv[1]
#        assert cell_type in params['cell_types'], 'Wrong cell type given %s should be in: %s' % (cell_type, str(params['cell_types']))
#    except:
#        cell_type = 'exc'

#    print 'Getting parameters from:', params_fn
#    data_folder = params['spiketimes_folder'] # should be 'volt_folder', but pynest has only one data_path

#    list_of_files = []
#    to_match = '%s_volt' % cell_type
#    for fn in os.listdir(data_folder):
#        m = re.match(to_match, fn)
#        if m:
#            path = data_folder + fn
#            print 'Found file:', path
#            list_of_files.append(path)
    
    list_of_files = sys.argv[1:]
    pylab.figure()
    for i_, fn in enumerate(list_of_files):
        plot_volt(fn, gid='all')
#        plot_volt(fn, gid=[49, 91])
        

    pylab.show() 

