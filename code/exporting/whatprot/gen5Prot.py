# -*- coding: utf-8 -*-
"""
Created on Sun May  4 13:22:36 2025

@author: JK-WORK
"""

import pandas as pd
import numpy as np

from common.peptide import Peptide
from simulate.label_peptides import label_peptides
from collections import defaultdict
from common.dye_seq import DyeSeq

def pick_specific_prots(fasta, target_genes=None):
    """
    Parses a FASTA‐style file and returns sequences for only the target genes,
    enumerated as pid 0..N-1 in the order found, plus the list of gene names.
    
    Parameters:
      fasta (str): path to your FASTA file
      target_genes (set of str): gene names to pick (e.g. {'PSMB10','PSME3','PSME4','HBB','HBA2'})
    
    Returns:
      picked: List of (pid, sequence) for each target found.
      gene_names: List of matching gene names in the same order.
    """
    if target_genes is None:
        target_genes = {'PSMB10','PSME3','PSME4','HBB','HBA2'}
    
    sequences = []
    gene_names = []
    
    with open(fasta, 'r') as f:
        current_gene = None
        current_seq = []
        
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('>'):
                # finalize previous if collecting
                if current_gene is not None:
                    sequences.append(''.join(current_seq))
                    gene_names.append(current_gene)
                    current_seq = []
                    current_gene = None
                    if len(sequences) == len(target_genes):
                        break
                
                # check this header for GN=
                parts = line.split('GN=')
                if len(parts) > 1:
                    gene = parts[1].split()[0]
                    if gene in target_genes:
                        current_gene = gene
            
            else:
                if current_gene is not None:
                    current_seq.append(line)
        
        # finalize last one at EOF
        if current_gene is not None and len(sequences) < len(target_genes):
            sequences.append(''.join(current_seq))
            gene_names.append(current_gene)
    
    # enumerate pids 0..N-1
    picked = list(enumerate(sequences))
    return picked, gene_names
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

npros,gene_names=pick_specific_prots("../examples/UP000005640_9606.fasta")
df = pd.DataFrame({
    'Index': [pid for pid, _ in npros],
    'GeneName': gene_names
})
df.to_csv("protein_descriptions_5prot.csv", index=False)
npros_idxd = [(i, uniProtIdx, seq) for i, (uniProtIdx, seq) in enumerate(npros)] ##Adds the new indexing.
peptides=gen_peps(npros_idxd,"trypsin");
print("Generated peptides")
label_set = ['DE','C','Y']
flustr = gen_flustrings(peptides, label_set)
print("Generated flustr")
df=gen_df_dataset(peptides,flustr)
print("Generated df")

df.to_csv("ExpTable5Prot.csv", index=False)