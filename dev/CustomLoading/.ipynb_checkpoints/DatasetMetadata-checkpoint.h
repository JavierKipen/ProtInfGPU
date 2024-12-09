#ifndef DATASETMETADATA_H
#define DATASETMETADATA_H

#include <iostream>
#include <vector>

using namespace std;
//Metadata from datasets that is used for calculations and others
typedef struct {
    vector<unsigned int> fluExpIdForI; //Vectore that has concatenated all the flu exps Indexes that a protein I can generate
    vector<unsigned int> nFluExpForI; //Number of different flu exps each protein can generate (size=n_prot)
    vector<float> probFluExpForI;     //for every element in fluExpIdForId it states the probability.
    vector<float> expNFluExpGenByI; //Expected number of flu exps by each protein.
    unsigned int nSparsity,nReadsTotal,nFluExp,nProt; //Other important variables of the dataset
} DatasetMetadata; 


#endif