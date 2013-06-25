"""
    Run several times the same script with differently oriented stimulus.

    + = = = = = +
    | IMPORTANT |
    + = = = = = +

    Before running this script:

    1) Make sure that these lines appear in NetworkSimModule.py

        if __name__ == '__main__':
            input_created = False
            orientation = float(sys.argv[1])
            ps.params['motion_params'][4] = orientation
            ps.set_filenames()

    2) In simulation_parameters.py:
    Make sure that the 'folder_name' variable contains the params['motion_params'][4] value:
    e.g.
        folder_name = 'OrientationTuning_%.2e/' % self.params['motion_params'][4]

    you can add more information to the folder name:
        folder_name = 'OrientationTuning_%.2e' % self.params['motion_params'][4]
        folder_name += connectivity_code
        folder_name += '-'+ self.params['motion_type']
        folder_name += '-'+ self.params['motion_protocol']
        folder_name += '/'

"""
import os
import numpy as np

orientations = np.linspace(0, np.pi, 5, endpoint=False)

for sweep_parameter in orientations:
    os.system('mpirun -np 8 python NetworkSimModule.py %f'  % (sweep_parameter))

    # on 1 core
#    os.system('python NetworkSimModule.py %f'  % (sweep_parameter))

