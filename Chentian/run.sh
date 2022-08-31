#!/bin/bash

#SBATCH --job-name=bertss512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=336:0:0
#SBATCH --mem=100G

python /user/home/ud21703/2022summer/code/codebert.py\

hostname