import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

whatprot_path= "/../../ext/whatprot/cc_code/bin/release/whatprot"
common_path_datasets= "/raid/jkipen/ProtInfGPU/data/WhatprotGen/Review/"
dye_seqs_path=common_path_datasets + "dye-seqs.tsv"
Div2_path=common_path_datasets+"Div2/";Div5_path=common_path_datasets+"Div5/";Div10_path=common_path_datasets+"Div10/";
Divs_paths=[Div2_path,Div5_path,Div10_path];
ErrParamsJsonFile="ErrParams.json"
n_samples_per_dataset=10000000; 
n_multidatasets=10;
n_dye_seqs = 152291;

div_path=Divs_paths[2];
dfs = [] 

for i in range(n_multidatasets):  #Groups multidatasets in one big dataset
    print("Analyzing multidataset " + str(i))
    dfr = pd.read_csv(div_path+ "MultiDatasets/radiometries_"+str(i)+".tsv", sep="\t", header=None, skiprows=3); #Skipping metadata rows
    dft = pd.read_csv(div_path+ "MultiDatasets/true-ids_"+str(i)+".tsv", sep="\t", header=None, skiprows=1); #Skipping metadata rows
    dfr['true_ids'] = dft[0]
    dfs.append(dfr)
    result_df = pd.concat(dfs, axis=0, ignore_index=True)
    result_df = result_df.groupby('true_ids', group_keys=False).apply(lambda group: group.head(100))
    dfs=[result_df];
    print("Number of samples last dataset: " + str(len(result_df))+ ". Samples per 150k: " + str(len(result_df)/150000))
    if (len(result_df)==n_dye_seqs*100):
        print("We already have 100 of each!")
        break
    

result_df.to_csv(div_path+"aux.csv", index=False)

radiometries_df = result_df.drop(columns=['true_ids'])
true_ids_series = result_df['true_ids']

# Get the number of reads (rows) from the filtered data
num_reads = len(result_df)
# Save the final radiometries file with custom header
num_timesteps = 40   # Given constant: number of timesteps
num_channels = 3     # Given constant: number of channels
# Save the final true IDs file with custom header (only number of reads)
true_ids_output_file = div_path+"true-ids.tsv"
with open(true_ids_output_file, "w") as f:
    f.write(f"{num_reads}\n")
    true_ids_series.to_csv(f, sep="\t", index=False, header=False)
    
radiometries_output_file = div_path+"radiometries.tsv"
with open(radiometries_output_file, "w") as f:
    f.write(f"{num_timesteps}\n")
    f.write(f"{num_channels}\n")
    f.write(f"{num_reads}\n")
    radiometries_df.to_csv(f, sep="\t", index=False, header=False)
##Now lets do the reduced dataset
reduced_path=div_path+"Reduced/"; #We will saved the reduced in other folder
df_small = result_df.groupby('true_ids', group_keys=False).apply(lambda group: group.head(2))
num_reads = len(df_small) #Update number of points

radiometries_df_s = df_small.drop(columns=['true_ids'])
true_ids_series_s = df_small['true_ids']

true_ids_output_file = reduced_path+"true-ids.tsv"
with open(true_ids_output_file, "w") as f:
    f.write(f"{num_reads}\n")
    true_ids_series_s.to_csv(f, sep="\t", index=False, header=False)
    
radiometries_output_file = reduced_path+"radiometries.tsv"
with open(radiometries_output_file, "w") as f:
    f.write(f"{num_timesteps}\n")
    f.write(f"{num_channels}\n")
    f.write(f"{num_reads}\n")
    radiometries_df_s.to_csv(f, sep="\t", index=False, header=False)