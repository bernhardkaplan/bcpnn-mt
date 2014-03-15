"""
If a data folder has been moved after simulation, the path names need to be updated accordingly.
"""
import os
import sys
import utils
import simulation_parameters



if __name__ == '__main__':

    if sys.argv[1].find('.json') != -1:
        print 'Do not give the .json file name, but the folder to be updated!'
        exit(1)

    ps = simulation_parameters.parameter_storage(sys.argv[1])
    new_path = os.path.abspath(sys.argv[1]) + '/'
    print 'new path :', os.path.abspath(ps.params['folder_name'])
    ps.set_filenames(new_path)
    print 'path after write:', os.path.abspath(ps.params['folder_name'])
    ps.write_parameters_to_file()
