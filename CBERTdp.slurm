#!/bin/bash
#SBATCH --job-name=CBERTdp   # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=875532@stud.unive.it     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=app_log.log   # Standard output and error log
#SBATCH --gres=gpu:1

date

conda activate nlp_env

python CBERTdp.py

date