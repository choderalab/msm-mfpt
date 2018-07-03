#!/bin/bash
#BSUB -J chignolin_score
#BSUB -n 32
#BSUB -R span[ptile=32]
#BSUB -R rusage[mem=3]
#BSUB -W 24:00
#BSUB -o /home/rafal.wiewiora/job_outputs/%J.stdout
#BSUB -eo /home/rafal.wiewiora/job_outputs/%J.stderr
 
source /home/rafal.wiewiora/.bashrc
export PYEMMA_NJOBS=1 OMP_NUM_THREADS=1
cd $LS_SUBCWD
python chignolin_score_msmlag50ns_splittica_alleigen.py
