#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J reg_main
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o hpc_files/Output_%J.out 
#BSUB -e hpc_files/Error_%J.err 

# here follow the commands you want to execute 
#/zhome/17/6/118138/master-thesis/env/bin/python /zhome/17/6/118138/master-thesis/regression_experiments/regression_validation_coco.py
#/zhome/17/6/118138/master-thesis/env/bin/python /zhome/17/6/118138/master-thesis/regression_experiments/regression_coco_2D_illustration.py
/zhome/17/6/118138/master-thesis/env/bin/python /zhome/17/6/118138/master-thesis/regression_experiments/regression_validation_sklearn.py
#/zhome/17/6/118138/master-thesis/env/bin/python /zhome/17/6/118138/master-thesis/regression_experiments/1D_reg_main.py