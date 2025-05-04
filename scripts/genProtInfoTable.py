# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:13:55 2025

@author: JK-WORK
"""
import pandas as pd

def gen_exp_prot_data(fasta, not_obs_prots=None, csv_out="protein_descriptions.csv"):
    """
    Parses a FASTA-like file, extracts description lines, removes specified indices,
    reindexes, writes to CSV, and returns a DataFrame.
    
    Parameters:
      fasta            : Path to the FASTA file.
      not_obs_prots    : List of string or int indices to remove (before reindexing).
                         Defaults to the set provided earlier.
      csv_out          : Output CSV filename.
    
    Returns:
      df : pandas.DataFrame with columns ['Index', 'Description'].
    """
    if not_obs_prots is None:
        not_obs_prots = ['3782','5619','10909','10948','11149','13130',
                         '13516','13539','13771','13840','14508',
                         '15193','15737','15930','16125','17518',
                         '19035','19227']
    # Convert to integer set
    remove_set = set(map(int, not_obs_prots))
    
    # Step 1: Read only description lines
    descriptions = []
    with open(fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                descriptions.append(line.strip())
    
    # Step 2: Filter out unwanted indices
    filtered = [(idx, desc) 
                for idx, desc in enumerate(descriptions) 
                if idx not in remove_set]
    
    # Step 3: Reindex and build DataFrame
    df = pd.DataFrame({
        'Index': range(len(filtered)),
        'Description': [desc for (_, desc) in filtered]
    })
    
    # Save to CSV
    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} descriptions to {csv_out}")
    
    return df

gen_exp_prot_data("../examples/UP000005640_9606.fasta")