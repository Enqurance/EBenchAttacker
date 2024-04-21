#!/bin/bash
#SBATCH --job-name=python
#SBATCH --gres=gpu:V100:1
#SBATCH --output=output.txt
#SBATCH --error=error.txt

bash run_gcg.sh llama2