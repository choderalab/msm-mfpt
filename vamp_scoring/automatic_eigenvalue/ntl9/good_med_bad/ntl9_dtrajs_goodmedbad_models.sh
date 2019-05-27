#!/bin/bash
#BSUB -J ntl9_models
#BSUB -n 10
#BSUB -R span[ptile=10]
#BSUB -R rusage[mem=5]
#BSUB -W 24:00
#BSUB -o /home/rafal.wiewiora/job_outputs/%J.stdout
#BSUB -eo /home/rafal.wiewiora/job_outputs/%J.stderr

export PYEMMA_NJOBS=1 OMP_NUM_THREADS=1
cd $LS_SUBCWD
python ntl9_dtrajs_goodmedbad_models.py
