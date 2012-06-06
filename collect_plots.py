import os
import sys
from NeuroTools import parameters as ntp

param_fn = 'simulation_parameters.info'
source_filename = 'prediction_fig_fn_base'
tmp_folder = 'temp_figures/'
if not os.path.exists(tmp_folder):
    os.system('mkdir %s' % tmp_folder)
tgt_tar = 'prediction_figures.tar ' # don't forget the whitespace at the end

def check(arg, dirname, fnames):
    if dirname.find('Figures') != -1:
        print "Check, dirnames:", dirname, "\n fnames:", fnames


new_fns = []

def get_parameter_file(arg, dirname, fnames):
    if fnames.count(param_fn):
        param_path = dirname + '/' + param_fn
        print 'Loading parameters from', param_path
#        try:
        P = ntp.ParameterSet(param_path)
        src_fn = P[source_filename] + '0.png'
        tgt_fn = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_0.png' % (P[source_filename], P['initial_connectivity'], P['w_sigma_x'], P['w_sigma_v'])
        print src_fn, tgt_fn
        os.system('cp %s %s' % (src_fn, tgt_fn))
        os.system('cp %s %s' % (tgt_fn, tmp_folder))

        src_fn = P[source_filename] + '1.png'
        tgt_fn = '%s%s_wsigmaX_%.2f_wsigmaV%.2f_1.png' % (P[source_filename], P['initial_connectivity'], P['w_sigma_x'], P['w_sigma_v'])
        print src_fn, tgt_fn
        os.system('cp %s %s' % (src_fn, tgt_fn))
        os.system('cp %s %s' % (tgt_fn, tmp_folder))

        new_fns.append(tgt_fn)

#        except:
#            pass




if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = '.'

# rename
os.path.walk(folder, get_parameter_file, None)

# build a tar
#tar_command = 'tar -cvf %s ' % tgt_tar
#for fn in new_fns:
#    tar_command += fn 
#    tar_command += ' '

tar_command = 'tar -cvf %s %s' % (tgt_tar, tmp_folder)
print tar_command
os.system(tar_command)

