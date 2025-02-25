# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:42:14 2025

@author: JK-WORK
"""


from random import sample,seed
from common.peptide import Peptide
from simulate.label_peptides import label_peptides
from collections import defaultdict
from common.dye_seq import DyeSeq
import pandas as pd
import os
import numpy as np


#cleave_proteins("../examples/UP000005640_9606.fasta","../temp/peptides.tsv","trypsin",n=100)

def pick_prots(fasta,n = -1):
    fpro = open(fasta, "r")
    fpro.readline()  # skip first '>' line for convenience.
    proteins = []
    pid = 0
    while True:
        protein = ""
        line = ""
        while True:
            line = fpro.readline()[0 : -1]
            if (not line):
                break
            if (line[0] == '>'):
                break
            protein += line
        proteins += [(pid, protein)]
        if (not line):
            break
        pid += 1
    fpro.close()
    npros = 0
    if (n == -1):
        npros = proteins
    else:
        npros = sample(proteins, n)
    #print(len(npros))
    return npros


def gen_peps(npros,protease):
    peptides=[];
    pep_count=0;
    for (pid,uniProtId, protein) in npros:
        lastcut = 0
        for i in range(len(protein) - 1):
            cut = False
            if (protease == "trypsin"):
                # Here we trypsinize, cutting the C-terminal side of lysine (K)
                # and arginine (R) residues, but only if they are not followed
                # by Proline (P).
                if ((protein[i] == 'K' or protein[i] == 'R') and (protein[i + 1] != 'P')):
                    cut = True
            elif (protease == "cyanogen bromide"):
                # Now we use cyanogen bromide, cutting the C-terminal side of
                # methionine (M).
                if ((protein[i] == 'M')):
                    cut = True
            elif (protease == "EndoPRO"):
                # Here we use EndoPRO, cutting the C-terminal side of proline
                # (P) and alanine (A).
                if ((protein[i] == 'P' or protein[i] == 'A')):
                    cut = True
            else:
                print('error, invalid protease: ' + protease)
            if (cut == True):
                peptide = protein[lastcut : i + 1]
                lastcut = i + 1
                peptides += [Peptide(peptide,pep_id=pep_count,src_proteins=[pid])]
                pep_count=pep_count+1;
        peptide = protein[lastcut:]
        peptides += [Peptide(peptide,pep_id=pep_count,src_proteins=[pid])]
        pep_count=pep_count+1;
    return peptides;



def gen_flustrings(peptides, labels_by_channel):
    str_dye_seqs = defaultdict(lambda: [])
    for peptide in peptides:
        list_dye_seq = ['.'] * len(peptide.seq)
        for i in range(len(peptide.seq)):
            for ch in range(len(labels_by_channel)):
                for aa in labels_by_channel[ch]:
                    if aa == peptide.seq[i]:
                        list_dye_seq[i] = str(ch)
        while len(list_dye_seq) > 0 and list_dye_seq[-1] == '.':
            list_dye_seq.pop()
        str_dye_seq = ''.join(list_dye_seq)
        str_dye_seqs[str_dye_seq] += [peptide]
    dye_seqs = [0] * len(str_dye_seqs)
    index = 0
    for str_dye_seq in str_dye_seqs:
        dye_seqs[index] = DyeSeq(len(labels_by_channel), str_dye_seq, index,
                str_dye_seqs[str_dye_seq])
        index += 1
    dye_seqs.sort(key=lambda obj: not not obj.dye_seq ) #Locates the null dye first (empty list of dye seq)!
    dye_seqs.pop(0); #We actually pop the null dye seq to simplify the analysis.
    return dye_seqs

def gen_df_dataset(peptides,flustrs):
     
    idxs=(-1)*np.ones(shape=(len(peptides),3),dtype=np.int32) #Save space for pep id,prot id, flustr id,
    flustrings=["" for x in range(len(peptides))];
    pepstrings=["" for x in range(len(peptides))];
    for flustr_id,flustr in enumerate(flustrs):
        for pep in flustr.src_peptides:
            idxs[pep.pep_id,:]=[pep.pep_id,pep.src_proteins[0],flustr_id];
            flustrings[pep.pep_id]="".join(flustr.dye_seq);
            pepstrings[pep.pep_id]=pep.seq;
    cols_to_remove=np.argwhere(idxs[:,0]==-1)[:,0];
    idxs=np.delete(idxs,cols_to_remove,axis=0)
    #Cutting the arrays and lists
    pepstrings = [item for item in pepstrings if item.strip()]
    flustrings = [item for item in flustrings if item.strip()]
    
    #Solving missing proteins in table (due to that they only produce null flustring! (not visible))
    d_idxs=np.diff(idxs[:,1])
    jumps=np.argwhere(d_idxs>1)[:,0]
    for j in jumps: #
        print("Protein N " + str(idxs[(j+1),1]-1) + " does not produce exp flus")
        idxs[(j+1):,1]=idxs[(j+1):,1]-d_idxs[j]+1                     
    
    default_row = {"Peptide Id": 0, "Peptide String": "Def", "Original Protein Id":0,
                                     "Flustring Id":0,"Flustring":"Def"};
    df = pd.DataFrame([default_row] * len(idxs)); #Table has to have the len of peptides!  
    df.iloc[:,[0,2,3]]=idxs;
    
    
    df["Peptide String"]=pepstrings;
    df["Flustring"]=flustrings;
    return df;
    

seed(0)
#20660 is the max!
n_proteins=20660;
path_datasets="../../DatasetsProtInf/";
prot_folder=path_datasets+str(n_proteins)+"_Prot/"
output_whatprot_path=prot_folder+"whatprot/"
if not os.path.exists(output_whatprot_path):  ##Creating Classifier subfolders!
    os.makedirs(output_whatprot_path)

npros=pick_prots("../examples/UP000005640_9606.fasta",n=n_proteins)
npros_idxd = [(i, uniProtIdx, seq) for i, (uniProtIdx, seq) in enumerate(npros)] ##Adds the new indexing.
peptides=gen_peps(npros_idxd,"trypsin");
print("Generated peptides")
label_set = ['DE','C','Y']
flustr = gen_flustrings(peptides, label_set)
print("Generated flustr")
df=gen_df_dataset(peptides,flustr)
print("Generated df")

df.to_csv(output_whatprot_path+"ExpTable.csv", index=False)