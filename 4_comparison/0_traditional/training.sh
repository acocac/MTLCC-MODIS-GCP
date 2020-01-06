
#!/bin/bash

project=(AMZ)
year=(15)
experiment=(4_comparison)
MODELS=(RF)
REFERENCES=(MCD12Q1v6raw_LCType1)
trainsize=1000
testsize=1000
channels=(8)

for reference in ${REFERENCES[@]}; do
    for model in ${MODELS[@]}; do

        if [ "$reference" = "MCD12Q1v6raw_LCType1" ]; then
             num_classes=(17)
        elif [ "$reference" = "MCD12Q1v6raw_LCProp1" ]; then
             num_classes=(16)
        elif [ "$reference" = "MCD12Q1v6raw_LCProp2" ]; then
             num_classes=(11)
        elif [ "$reference" = "ESAraw" ]; then
             num_classes=(37)
        elif [ "$reference" = "Copernicusraw" ]; then
             num_classes=(22)
        elif [ "$reference" = "Copernicusnew_all" ]; then
             num_classes=(22)
        elif [ "$reference" = "Copernicusnew_cebf" ]; then
             num_classes=(22)
        fi

        mkdir -p "E:/acocac/research/${project}/models/$experiment/_logs"
        echo "Processing $model"
        logfname="E:/acocac/research/${project}/models/$experiment/_logs/${model}.log"
        python train_classifier.py "E:/acocac/research/${project}/models/$experiment/${model}_train${trainsize}test${testsize}_${year}_${reference}_8d" \
            --classifier=$model \
            --train="F:/acoca/research/gee/dataset/${project}/RF/train_dataset.csv" \
            --test="F:/acoca/research/gee/dataset/${project}/RF/test_dataset.csv" \
            --channels=$channels > $logfname 2>&1
    done
done
