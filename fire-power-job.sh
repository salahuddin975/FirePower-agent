#!/bin/bash 
#SBATCH -J fire-power 
#SBATCH -o fire-power.log 
#SBATCH -t 24:10:00 
#SBATCH -N 1 -n 2
#SBATCH -p gpu  #needed only on Sabine or Opuntia
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
 
module load python/3.7
python agent.py -g "configurations/configuration.json" -p "assets/case24_ieee_rts.py"
