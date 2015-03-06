sbatch jobfile_testing_milner_with_params.sh 0.5 -5.0 1.0 0.10 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.5 -10.0 2.0 0.10 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.5 -25.0 5.0 0.10 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.5 -5.0 1.0 1.00 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.5 -10.0 2.0 1.00 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.5 -25.0 5.0 1.00 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.8 -5.0 1.0 0.10 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.8 -10.0 2.0 0.10 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.8 -25.0 5.0 0.10 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.8 -5.0 1.0 1.00 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.8 -10.0 2.0 1.00 -1.00
sleep 0.2
sbatch jobfile_testing_milner_with_params.sh 0.8 -25.0 5.0 1.00 -1.00
sleep 0.2

#n=1
#for w_ie in -0.1 -0.2 -0.5 -1.0 
#do
    #for w_ei in 0.1 0.2 0.5 1.0 
    #do
        #for gain in 0.1 1.0 5.0 10.
        #do
            #echo "submitting job $((n++)) with parameter x =" $gain
            ##python main_test.py  TrainingSim__1x70x1_0-70_taui5_nHC20_nMC16_blurXV_0.00_0.05_pi1.0e-04/Connections/conn_matrix_mc.dat  TrainingSim__1x70x1_0-70_taui5_nHC20_nMC16_blurXV_0.00_0.05_pi1.0e-04/Connections/conn_matrix_mc.dat $w_ee
            #sbatch jobfile_testing_milner_with_params.sh $gain $w_ie $w_ei
            #sleep 1.0
            ##n++
        #done
    #done
#done


#for gain in 
#do
    #echo "submitting job $((n++)) with parameter x =" $gain
    ##python main_test.py  TrainingSim__1x70x1_0-70_taui5_nHC20_nMC16_blurXV_0.00_0.05_pi1.0e-04/Connections/conn_matrix_mc.dat  TrainingSim__1x70x1_0-70_taui5_nHC20_nMC16_blurXV_0.00_0.05_pi1.0e-04/Connections/conn_matrix_mc.dat $w_ee
    #sbatch jobfile_testing_milner_with_params.sh $gain $w_ie $w_ei
    #sleep 1.0
#done

#wa_d1_d1_pos=0.0
#wa_d1_d1_neg=1.0

#for (($(wa_d1=0.2);  $wa_d1 < $(0.7); $wa_d1 = $($wa_d1 + 0.1)))

#w_to_action_inh=-8
#w_to_action_exc=8.

#for w_to_action_inh in -6. -8. # weight from strD2 to action layer
#do
#    for wa_d1_d1_pos in 0. 0.5 1.0
#    do
#        for w_to_action_exc in 6. 8. # weight from striatum D1 to action
#        do
#            for wa_d1_d1_neg in 1. 0.5 2. 0. # weight amplification d1 -> d1 negative weights
#            do
#                for wa_d1 in 0.5 1.0 1.5 2.0 4.0 
#                do
#                    for wa_d2 in 0.5 1.0 1.5 2.0 4.0 
#                    do
#                        wa_d2=$wa_d1
#                        echo "submitting job" $n " with parameter x =" $wa_d1 $wa_d2 $wa_d1_d1_pos $wa_d1_d1_neg $w_to_action_exc $w_to_action_inh
#                        sbatch jobfile_testing_milner_with_params.sh $wa_d1 $wa_d2 $wa_d1_d1_pos $wa_d1_d1_neg $w_to_action_exc $w_to_action_inh
#                        sleep 0.1
#                    done
#                done
#            done
#        done
#    done
#done


#echo 'Submitting jobfile_remove_empty_files.sh ...'
#sleep 1
#sbatch jobfile_remove_empty_files.sh
