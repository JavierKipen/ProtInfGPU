#ifndef CROSSVALWRAPPER_H
#define CROSSVALWRAPPER_H

#include<iostream>
#include<string>
#include<vector>
#include "FileSystemInterface.h"

using namespace std;

#define GB_GPU_USAGE_DEFAULT 4 //Limiting memory usage on GPU
#define N_EPOCHS_DEFAULT 60 

class CrossValWrapper {  //Class to wrap both the filesystem dataset pulling, cross validation score selection and EM calculation.
    
    public:
        CrossValWrapper();
        ~CrossValWrapper();
        void init(string outFolder,FileSystemInterface * pFSI); //Initializes the class
        void setGPUMemLimit(float nGb); //Number of gygabites as maximum to use in the GPU calculations
        void setNSparsity(unsigned int nSparsityRed);
    
        void computeEMCrossVal();
        void computeEMCrossValEpoch();
        

    
        FileSystemInterface * pFSI; //Interface for loading the reads
        unsigned int nEpochs;
        unsigned int nSparsityRed; //We can reduce the sparsity compared to the native dataset to compare results!
        vector<vector<float>> pIEsts; //Estimations of P_I for all crossValidation
        vector<vector<float>> updates; //updates to then change P_I_Ests.
    private:
        void loadScoresInBuffer(vector<unsigned int> &IdxsCv); //Given the Indexes of the reads picked for the crossval, copies from the ram to the vector to send to GPU
        void updatePIs(); //Uses the update weights to update the PIs estimations!
        void partitionDataset(unsigned int cvIndex);
        void computeUpdateSubSet(unsigned int cvIndex,unsigned int subSetIdx);
        void getValidCVScoresIds();
    
        string outFolderPath;
    
        unsigned long nBytesToUseGPU; //Limit of memory to use of the GPU in Bytes.
        unsigned int maxReadsToProcessInGpu; //Given the restriction in memory in GPU, we have a limit on the amount of reads we can use in the moment
        unsigned int nReadsOffset;
        vector<float> topNFluExpScores; //Sparse vectors representation of the scores to compute
        vector<float> topNFluExpScoresIds; 
    
        vector<unsigned int> idToCvScoreIdsStart,idToCvScoreIdsEnd; //For the loaded batch of the scores dataset on RAM, these vectores point to the beg and end of the scores that are valid
        vector<vector<unsigned int>> cvScoreIdsVecOfVec; //Used to partition the scoreIds of the crossval into vectors of size maxReadsToProcessInGpu, to send to gpu

};

#endif