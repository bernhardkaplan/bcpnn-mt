import sys
import os
import utils
import numpy as np
import json

def plot_dirnames(dir_names):

    script_name = 'PlottingScripts/PlotInformationDiffusion.py'

    for i_, directory in enumerate(dir_names):
        print '\n\n=========================\nAnalysis %d / %d begins\n==================================\n\n' % (i_ + 1, len(dir_names))
        
        print 'dir:', directory
        dirn = add_dir + directory
        os.system('python %s %s' % (script_name, dirn))


def plot_single_dirname(folder):

    script_name = 'PlottingScripts/PlotInformationDiffusion.py'
    print 'folder:', folder
    os.system('python %s %s' % (script_name, folder))


if __name__ == '__main__':


    if len(sys.argv) == 1:
        #dir_names_fn = 'dir_names.json' # was created via ipython using recursive os.listdir
        dir_names_fn = 'dirnames_new.json' # was created via ipython using recursive os.listdir
        f = file(dir_names_fn, 'r')
        dir_names = json.load(f)
        plot_dirnames(dir_names)
    else:
        folders = sys.argv[1:]
        for folder in folders:
            plot_single_dirname(folder)



