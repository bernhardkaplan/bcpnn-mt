
#echo jobfile_testing_milner_with_params.sh 5 5 0 5 -5 
#sbatch jobfile_testing_milner_with_params.sh 5 5 0 5 -5 

#w_noise_exc=1.25
n=1
for w_noise_exc in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
do
    for ratio in 1 
    do
        taui_ampa=5
        taui_nmda=5
        for w_tgt_exc in 0.6
        do
            w_tgt_inh=$w_tgt_exc
            sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio $w_noise_exc
            echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio  $w_noise_exc
            echo $n
            sleep 0.2
            n=$(expr $n + 1)
        done

        taui_ampa=5
        taui_nmda=150
        for w_tgt_exc in 0.6
        do
            w_tgt_inh=$w_tgt_exc
            sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio $w_noise_exc
            echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio  $w_noise_exc
            echo $n
            sleep 0.2
            n=$(expr $n + 1)
        done

        taui_ampa=150
        taui_nmda=150
        for w_tgt_exc in 0.6
        do
            w_tgt_inh=$w_tgt_exc
            sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio $w_noise_exc
            echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio  $w_noise_exc
            echo $n
            sleep 0.2
            n=$(expr $n + 1)
        done


        taui_ampa=150
        taui_nmda=5
        for w_tgt_exc in 0.6
        do
            w_tgt_inh=$w_tgt_exc
            sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio $w_noise_exc
            echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $w_tgt_exc $w_tgt_inh $ratio  $w_noise_exc
            echo $n
            sleep 0.2
            n=$(expr $n + 1)
        done
    done
done
