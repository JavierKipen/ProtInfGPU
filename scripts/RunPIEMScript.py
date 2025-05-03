import numpy as np
#import pandas as pd
import os




results_path="/home/jkipen/ProtInfGPU/results/"
opt_prot_inf="../code/cuda/opt.out" 
in_path= "/home/jkipen/raid_storage/ProtInfGPU/data/20642_Prot/binary/ProbeamBetterConfig"
out_path= results_path+"20642_Prot/SparsityProbeamBetterConfigW2"

#Parameters we will tune
n_epochs=30;
cv_runs=2;
device=0;
n_threads=1024;
#ns_sparsity=[1,2,10,50,500];
ns_sparsity=[5,20,100,200];

for n_sparsity in ns_sparsity:
    base_command= opt_prot_inf + " " + in_path + " " +out_path 
    command= base_command + " -d "+ str(device) + " -e "+str(n_epochs) +" -c "+str(cv_runs)+" -m 80 -M 30 -t "+str(n_threads)+" -n "+str(n_sparsity);
    print(command)
    os.system(command)