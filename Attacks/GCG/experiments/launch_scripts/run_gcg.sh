#!/bin/bash

#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # the official name of target model
export time_str=$2 # behaviors or strings
export model_name=$3 # the name of target model in GCG
export dataset=$4

# Create results folder if it doesn't exist
if [ ! -d "./result/GCG/${time_str}/" ]; then
    mkdir -p "./result/GCG/${time_str}"
    echo "Folder './result/GCG/${time_str}/' created."
else
    echo "Folder './result/GCG/${time_str}/' already exists."
fi

for data_offset in 0
do

    python3 -u ./Attacks/GCG/experiments/main.py \
        --config="./Attacks/GCG/experiments/configs/individual_${model_name}.py" \
        --config.attack=gcg \
        --config.result_prefix="./result/GCG/${time_str}/${model}" \
        --config.data_offset=$data_offset \
        --config.train_data=$dataset \

done