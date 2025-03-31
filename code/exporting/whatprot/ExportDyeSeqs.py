# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:30:36 2025

@author: JK-WORK
"""

import pandas as pd
import os
import numpy as np


def flu_df_to_dye_seqs_file(flu_df, path_out):
    seqs=flu_df["Flustring"].values
    seqs_reversed=[seq[::-1] for seq in seqs]
    f = open(path_out, 'w')
    f.write("3\n")  # num channels, we are working with 3 by default.
    f.write(str(len(flu_df)) + "\n")
    for i in range(len(flu_df)):
        f.write("".join(seqs_reversed[i]) + "\t")
        f.write("1" + "\t") # We will generate a dataset with all flustrings equally likely. 
        f.write(str(i) + "\n")
    f.close()

def gen_dye_seq_file(path_exp_table,path_out):
    df=pd.read_csv(path_exp_table);
    flu_ids=df["Flustring Id"].to_numpy();
    df_filtered = df.drop_duplicates(subset="Flustring Id", keep="first")[["Flustring Id", "Flustring"]]; #Keeps first row where each flustring appears, and takes only the columns that matter.
    flu_df_to_dye_seqs_file(df_filtered,path_out)
    
    
    
path_exp_table="/raid/jkipen/ProtInfGPU/data/WhatprotGen/WholeProteomeTests/ProbeamPaperParams/ExpTable.csv"
path_out="/raid/jkipen/ProtInfGPU/data/WhatprotGen/WholeProteomeTests/ProbeamPaperParams/DyeSeqs.tsv"
gen_dye_seq_file(path_exp_table,path_out)