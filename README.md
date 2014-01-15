# # # # # # # # # # # # #
#     bcppn-mt          #
# # # # # # # # # # # # #

Requirements:
    pyNN
    Brian, NEST or another pyNN supported simulator

Run:
    python main_noColumns.py

    or 

    mpirun -np [number_of_processes] python main_noColumns.py


installation on (neuro)debian(virtual machine on MacOsX)
--------------------------------------------------------

   sudo aptitude install ffmpeg 

#PyNN
   sudo aptitude install python-pynn
   sudo aptitude install python-pip
   sudo aptitude install python-imaging
#NEST dependencies
   sudo aptitude install gsl-bin libgsl0-dev
   sudo aptitude install python-dev
   sudo aptitude install libncurses-dev
   sudo aptitude install libreadline-dev
#NEST
   wget http://www.nest-initiative.org/download/yHTCUpeiTsCr/nest-2.0.0.tar.gz
   tar zxvf nest-2.0.0.tar.gz
   cd nest-2.0.0
   ./configure
   make
   sudo make install

