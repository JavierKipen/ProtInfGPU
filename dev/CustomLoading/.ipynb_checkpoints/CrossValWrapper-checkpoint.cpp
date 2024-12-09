#include "CrossValWrapper.h"
#include <math.h>

CrossValWrapper::CrossValWrapper()
{
    
}
CrossValWrapper::~CrossValWrapper()
{
    
}
void CrossValWrapper::init(string outFolder,FileSystemInterface * pFSI)
{
    outFolderPath=outFolder;
    this->pFSI=pFSI;
    idToCvScoreIdsEnd.resize(pFSI->nCrossVal,0);idToCvScoreIdsStart.resize(pFSI->nCrossVal,0);
    //CM.setMetadata(); //Data To configure the processing
    nSparsityRed=pFSI->datasetMetadata.nSparsity;
    setGPUMemLimit(GB_GPU_USAGE_DEFAULT);
     //CM.memAlloc(maxReadsToProcessInGpu); //Allocates the memory to process this amount of reads.
    topNFluExpScores.resize(maxReadsToProcessInGpu*nSparsityRed,0);topNFluExpScoresIds.resize(maxReadsToProcessInGpu*nSparsityRed,0);
    
    float norm=0;
    for(unsigned int j=0;j<pFSI->datasetMetadata.nProt;j++)
        norm+=pFSI->datasetMetadata.expNFluExpGenByI[j];
    vector<float> auxPI(pFSI->datasetMetadata.nProt,0);
    for(unsigned int j=0;j<pFSI->datasetMetadata.nProt;j++)
        auxPI[j]=pFSI->datasetMetadata.expNFluExpGenByI[j]/norm;
    for(unsigned int i=0;i<pFSI->nCrossVal;i++)
    {   
        vector<float> aux(pFSI->datasetMetadata.nProt,0);
        updates.push_back(aux);
        pIEsts.push_back(auxPI); //equally likely proteins assumption!
    }
}
void CrossValWrapper::setGPUMemLimit(float nGb)
{
    nBytesToUseGPU=nGb*pow(2,30);
    maxReadsToProcessInGpu = 100;
    //maxReadsToProcessInGpu = CM.maxReadsToCompute(nBytesToUseGPU); //Gets how many reads can be computed with the restriction. Assumes CM was initialized with metdata
}

void CrossValWrapper::computeEMCrossVal()
{
    for(unsigned int i=0;i<nEpochs;i++)
    {
        computeEMCrossValEpoch(); //Gets the updates values looping through one read of the whole dataset with all crossval picks
        updatePIs(); //Uses the update weights to obtain new P(I) estimates
    }
}
void CrossValWrapper::computeEMCrossValEpoch()
{
    vector<vector<unsigned int>> emptyVecOfVec; //To reset our variable
    nReadsOffset=0; //Keeps track of the number of reads got before
    while(!pFSI->finishedReading) //The dataset may not fit in RAM, so we load batches!
    {
        
        pFSI->readPartialScores(); //Loads batch of scores from disk
        getValidCVScoresIds(); //Selects the end IDs of the scores so we use scores that are in RAM
        for(unsigned int i=0;i<pFSI->nCrossVal;i++) //For every cross validation dataset
        {
            for(auto& vec : cvScoreIdsVecOfVec)
                vec.clear();
            cvScoreIdsVecOfVec.clear(); //We clean the vect of vectors before using it!
            partitionDataset(i);//The cvScoreIds are partitioned in cvScoreIdsVecOfVec so each compute can be done easily
            for(unsigned int j=0;j<cvScoreIdsVecOfVec.size();j++)
                computeUpdateSubSet(i,j);
        }
        idToCvScoreIdsStart=idToCvScoreIdsEnd; // for the next data, we know that will start after what we had
        for(auto& itr:idToCvScoreIdsStart)
            itr=itr+1; //But we add one so we dont take the same sample twice! Not sure of this.
        nReadsOffset += pFSI->nReadsInMemory; //Saves the amount of reads that have already been visited.
    }
    pFSI->restartReading();
    for(auto& itr:idToCvScoreIdsStart)
        itr=0; //Restart idxs of start
}

void CrossValWrapper::getValidCVScoresIds()
{
    for(unsigned int i=0;i<pFSI->nCrossVal;i++) //For every cross validation dataset
    {
        unsigned int j=0;
        for(j=idToCvScoreIdsStart[i];j<pFSI->cvScoreIds[i].size();j++)
            if((pFSI->cvScoreIds[i][j])>=nReadsOffset+pFSI->nReadsInMemory) //The reads in ram are [nReadsOffset,nReadsOffset+FSI->nReadsInMemory)
                break;
        idToCvScoreIdsEnd[i]=j-1; //when using break it still advances j.
    }
}

void CrossValWrapper::partitionDataset(unsigned int cvIndex)
{
    unsigned int nScores=idToCvScoreIdsEnd[cvIndex]-idToCvScoreIdsStart[cvIndex];//This is the total amount of scores we are going to process of this cross validation run
    auto itr=pFSI->cvScoreIds[cvIndex].begin()+idToCvScoreIdsStart[cvIndex];
    auto itr_end=pFSI->cvScoreIds[cvIndex].begin()+idToCvScoreIdsEnd[cvIndex];
    while(itr != itr_end)
    {
        auto beg=itr;
        auto end=itr_end; //by default, it would be the end of the scores we are considering
        if(distance(itr, itr_end)>maxReadsToProcessInGpu) //When we have more than this, we can fill a buffer completely
            end=itr+maxReadsToProcessInGpu;
        
        vector<unsigned int> auxVec(beg, end);
        for (auto& it : auxVec) 
            it = it - nReadsOffset; //Conversion from pointer to whole dataset TO pointer in the RAM memory.
        cvScoreIdsVecOfVec.push_back(auxVec);
        itr=end; //the next batch starts where the previous started.
    }
}
    
void CrossValWrapper::computeUpdateSubSet(unsigned int cvIndex,unsigned int subSetIdx)
{
    loadScoresInBuffer(cvScoreIdsVecOfVec[subSetIdx]); //Puts certain scores on the buffer of this class
    //calculateUpdates(updates[cvIndex].data()); //Calls the update calculation on the GPU, and sums the updates obtained.
}

void CrossValWrapper::loadScoresInBuffer(vector<unsigned int> &IdxsCv)
{//The idxs refer to which point in the buffer from pFSI will be selected: NOTE that they need to be normalized if the dataset enters in RAM memory
    unsigned int len=IdxsCv.size(); //Has to be lower or eq than the buffer length!
    for(unsigned int i=0;i<len;i++)
    {
        unsigned int currId=IdxsCv[i]; //Score Id that will be copied
        for(unsigned int j=0;j<nSparsityRed;j++) //We copy nSparsityRed from the 
        {
            unsigned long idxInPFSI=currId*pFSI->datasetMetadata.nSparsity+j; //In pFSI the array has metadata.nSparsity columns, while the array to send to gpu has this->nSparsityRed columns
            unsigned long idxInBuffer=i*nSparsityRed+j;
            if (idxInPFSI>pFSI->TopNScoresPartialFlattened.size())
                cout << "ERR1" << endl;
            if (idxInBuffer>topNFluExpScores.size())
                cout << "ERR2" << endl;
            topNFluExpScores[idxInBuffer] = pFSI->TopNScoresPartialFlattened[idxInPFSI];
            topNFluExpScoresIds[idxInBuffer] = pFSI->TopNScoresIdsPartialFlattened[idxInPFSI];
        }
    }
}
void CrossValWrapper::updatePIs()
{
    unsigned int nProt=pFSI->datasetMetadata.nProt;
    unsigned int nDatasets=pFSI->nCrossVal;
    for(unsigned int i=0;i<nDatasets;i++)
    {
        float normVal=0;
        for(unsigned int j=0;j<nProt;j++)
            normVal+=updates[i][j];
        for(unsigned int j=0;j<nProt;j++)
            pIEsts[i][j]=updates[i][j]/normVal;
        for(unsigned int j=0;j<nProt;j++)
            updates[i][j]=0;
    }
}


void CrossValWrapper::setNSparsity(unsigned int nSparsityRed)
{
    if(nSparsityRed<pFSI->datasetMetadata.nSparsity)
        this->nSparsityRed=nSparsityRed;
    else
        cout << "Sparsity specified cannot be lower than the datasets one!" << endl;
}
