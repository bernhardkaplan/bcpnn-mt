


#sbatch jobfile_testing_milner_with_params.sh  5 200 1. 6
#sleep 0.2
#sbatch jobfile_testing_milner_with_params.sh  5 5 1. 6
#sleep 0.2

#taui_ampa = float(sys.argv[1])
#taui_nmda = float(sys.argv[2])
#bcpnn_gain = float(sys.argv[3])
#w_input_exc = float(sys.argv[4])
#ampa_nmda_ratio = float(sys.argv[5])

w_input_exc=5.
n=1
for taui_ampa in 5 10
do
    for taui_nmda in 100 150 200
    do
        for gain in 0.5 1.0 1.5 2.0 2.5 3.0
        do
            for ratio in 0.25 0.5 1.0 2.0 3.0 4.0 5.0
            do
                #echo "submitting job $((n++)) with parameter x =" $gain
                #python main_test.py  TrainingSim__1x70x1_0-70_taui5_nHC20_nMC16_blurXV_0.00_0.05_pi1.0e-04/Connections/conn_matrix_mc.dat  TrainingSim__1x70x1_0-70_taui5_nHC20_nMC16_blurXV_0.00_0.05_pi1.0e-04/Connections/conn_matrix_mc.dat $w_ee
                sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $w_input_exc $ratio 
                echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $w_input_exc $ratio 
                sleep 0.1
                n=$(expr $n + 1)
                echo $n
            done
        done
    done
done

