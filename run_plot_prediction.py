import os
import re
import json
import sys


"""
This script has two different modes:

    1) It gets a list of folder names stored as a json file as command argument

    2) Run through all the folders that match as certain structure
"""

script_name = 'create_something_new.py'

#script_name = 'run_large_scale_analysis.py'
#script_name = 'plot_prediction.py'


def run_by_match(folder_to_match):

    for thing in os.listdir('.'):
        m = re.search('%s' % folder_to_match, thing)
        if m:
            cmd = 'python %s %s' % (script_name, thing)
            print cmd
            os.system(cmd)


def get_filenames(fn_with_missing_data):
    f = file(fn_with_missing_data, 'r')
    list_of_dirs = json.load(f)
    print '\nlist_of_dirs', list_of_dirs
    for dir_name in list_of_dirs:
        cmd = 'python %s %s' % (script_name, dir_name)
        os.system(cmd)


if len(sys.argv) > 1:
    fn_in = sys.argv[1]
    # 'missing_data_dirs.json'
    get_filenames(fn_in)


conn_code = 'AIII'
#to_match = '^LargeScaleModel_(.*)'
#to_match = '^LargeScaleModel_AIII_bx1.00e-01_bv1.00e-01_wsigmax2.50e-01_wsigmav2.50e-01_wee(.*)'
to_match = '^LargeScaleModel_AIII_(.*)'
run_by_match(to_match)
