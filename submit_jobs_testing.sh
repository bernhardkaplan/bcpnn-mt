
#echo jobfile_testing_milner_with_params.sh 5 5 0 5 -5 
#sbatch jobfile_testing_milner_with_params.sh 5 5 0 5 -5 

#w_noise_exc=1.25
n=1
for w_noise_exc in 1.
do
    for ratio in 0.5 1 3 5
    do
        taui_ampa=5
        taui_nmda=5
        for w_tgt_exc in 0.25 0.5 0.75 1. 1.25 1.5 1.75
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
        for w_tgt_exc in 0.25 0.5 0.75 1. 1.25 1.5 1.75
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
        for w_tgt_exc in 0.25 0.5 0.75 1. 1.25 1.5 1.75
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
