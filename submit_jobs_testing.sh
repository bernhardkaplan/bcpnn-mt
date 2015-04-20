

ratio=5.
n=1
for w_ie in -5 
do
    taui_ampa=5
    taui_nmda=5
    for gain in 0.05 0.1 
    do
        sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio $w_ie
        echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio $w_ie
        echo $n
        sleep 0.2
        n=$(expr $n + 1)
    done

    taui_ampa=5
    taui_nmda=150
    for gain in 0.05 0.1 
    do
        sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio $w_ie
        echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio $w_ie
        echo $n
        sleep 0.2
        n=$(expr $n + 1)
    done

    taui_ampa=150
    taui_nmda=150
    for gain in 0.03 0.04 0.06
    do
        sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio $w_ie
        echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio $w_ie
        echo $n
        sleep 0.2
        n=$(expr $n + 1)
    done
done

#for taui_ampa in 5 150 200
#do
    #taui_nmda=$taui_ampa
    #for gain in 0.1 0.3 0.5 0.7 1.0 1.5 2.0
    #do
        #sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio 
        #echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $ratio 
        #echo $n
        #sleep 0.2
        #n=$(expr $n + 1)
    #done
#done

#for taui_ampa in 
#do
    #for taui_nmda in 100 150 200
    #do
        #for gain in 0.5 1.0 1.5 2.0 2.5 3.0
        #do
            #for ratio in 0.25 0.5 1.0 2.0 3.0 4.0 5.0
            #do
                #echo "submitting job $((n++)) with parameter x =" $gain
                #sbatch jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $w_input_exc $ratio 
                #echo jobfile_testing_milner_with_params.sh $taui_ampa $taui_nmda $gain $w_input_exc $ratio 
                #echo $n
                #sleep 0.2
                #n=$(expr $n + 1)
            #done
        #done
    #done
#done

