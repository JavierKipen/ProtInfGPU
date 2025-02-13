#include "GPUDataManager.h"
#include <iostream>
#include <vector>
#include <algorithm>    // std::max


using namespace std;

GPUDataManager::GPUDataManager()
{
}
  
bool GPUDataManager::allocateForNumberOfReads(DatasetMetadata *pDM,DeviceData *pdevData, DeviceData **d_ppdevData)
{
    bool retVal=false;
    unsigned long nReadsMax=pDM->nReadsTotal;//In NReads Total is actually stored the number of reads we are using for computing!
    onesVecLen=std::max({nReadsMax,(unsigned long)pDM->nSparsity,(unsigned long)pDM->nProt}); 
    if(!cudaMalloc(&(pdevData->d_NFexpForI), sizeof(unsigned int)*pDM->nProt))
        if(!cudaMalloc(&(pdevData->d_TopNFluExpId), (unsigned long)sizeof(unsigned int)*nReadsMax*(unsigned long)pDM->nSparsity))
            if(!cudaMalloc(&(pdevData->d_FexpIdForI), sizeof(unsigned int)*pDM->fluExpIdForI.size()))
                if(!cudaMalloc(&(pdevData->d_PFexpForI), sizeof(float)*pDM->fluExpIdForI.size()))
                    if(!cudaMalloc(&(pdevData->d_TopNFluExpScores), (unsigned long)sizeof(float)*nReadsMax*(unsigned long)pDM->nSparsity))
                        if(!cudaMalloc(&(pdevData->d_pRem), sizeof(float)*nReadsMax))
                            if(!cudaMalloc(&(pdevData->d_MatAux), (unsigned long)sizeof(float)*nReadsMax*(unsigned long)pDM->nProt))
                                if(!cudaMalloc(&(pdevData->d_ones), sizeof(float)*onesVecLen))
                                    if(!cudaMalloc(d_ppdevData, sizeof(*pdevData)))
                                        if(!cudaMalloc(&(pdevData->d_PIEst), sizeof(float)*pDM->nProt))
                                            if(!cudaMalloc(&(pdevData->d_VecAux), sizeof(float)*onesVecLen))
                                                retVal=true;
    return retVal;
}

unsigned long GPUDataManager::maxReadsToCompute(DatasetMetadata *pDM,unsigned long nBytesLim)
{ //Given a byte limit, we can calculate how many reads could be computed at the same update.
    unsigned long metadataBytes = sizeof(unsigned int)* (pDM->fluExpIdForI.size(),pDM->nFluExpForI.size()+5)
                                    + sizeof(float)* pDM->probFluExpForI.size();//+4 unsigned int for the nsparsity,nreads... . 
    unsigned long calcVariables = sizeof(float)* (2*pDM->nProt); //VecAux and PIEst has length nProt. Problem with d_ones: Can depend or not of length of reads (we dont consider it to be simpler, but its ok because its much less size than other variables.
    unsigned long bytesToAllocPerRead= sizeof(float) * (1+(pDM->nProt)+1+(pDM->nSparsity)) + sizeof(unsigned int) *pDM->nSparsity; //Floats corresponding to  d_pRem, *d_MatAux, *d_ones and d_TopNFluExpScores, and uint d_TopNFluExpScores.
    
    return (nBytesLim-calcVariables-metadataBytes)/bytesToAllocPerRead;
}


void GPUDataManager::metadataToGPU(DatasetMetadata *pDM,DeviceData *pdevData, DeviceData *d_pdevData)
{
    cudaMemcpy(pdevData->d_NFexpForI, pDM->nFluExpForI.data(), sizeof(unsigned int)*pDM->nProt, cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_FexpIdForI, pDM->fluExpIdForI.data(), sizeof(unsigned int)*pDM->fluExpIdForI.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_PFexpForI, pDM->probFluExpForI.data(), sizeof(float)*pDM->fluExpIdForI.size(), cudaMemcpyHostToDevice);
    pdevData->nSparsity=pDM->nSparsity;
    pdevData->nReadsMax=pDM->nReadsTotal;//In NReads Total is actually stored the number of reads we are using for computing!
    pdevData->nReadsProcess=pDM->nReadsTotal; //Assumes that nReadProcess will be max (can change in the processing calls.
    pdevData->nFluExp=pDM->nFluExp;
    pdevData->nProt=pDM->nProt;
    createOnesVec(pdevData);
    unsigned int size_data=sizeof(*pdevData);
    cudaMemcpy(d_pdevData, pdevData, size_data, cudaMemcpyHostToDevice);
}

void GPUDataManager::loadNewDataToGPU(PNewData pNewData,DeviceData *pdevData,DeviceData *d_pdevData)
{
    pdevData->nReadsProcess=pNewData.nReads;//Amount of reads to process
    cudaMemcpy(&(d_pdevData->nReadsProcess), &(pNewData.nReads), sizeof(unsigned int), cudaMemcpyHostToDevice); //Copies reads to process to device
    cudaMemcpy(pdevData->d_TopNFluExpId, pNewData.pTopNFluExpIds, sizeof(unsigned int)*pdevData->nReadsProcess*pdevData->nSparsity, cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_TopNFluExpScores, pNewData.pTopNFluExpScores, sizeof(float)*pdevData->nReadsProcess*pdevData->nSparsity, cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_PIEst, pNewData.pPIEst, sizeof(float)*pdevData->nProt, cudaMemcpyHostToDevice); //Could be passed less times!
}


void GPUDataManager::retrieveOutput(float * updateVectorOut,DeviceData *devData)
{
    cudaMemcpy(updateVectorOut, devData->d_VecAux, sizeof(float)*devData->nProt, cudaMemcpyDeviceToHost); //The update is contained in the auxiliar vector.
}

void GPUDataManager::createOnesVec(DeviceData *pdevData)
{
    vector<float> ones(onesVecLen, 1);
    cudaMemcpy(pdevData->d_ones, ones.data(), sizeof(float)*onesVecLen, cudaMemcpyHostToDevice);
    cudaMemcpy(pdevData->d_VecAux, ones.data(), sizeof(float)*onesVecLen, cudaMemcpyHostToDevice); //We also initialize dvecAux
}
void GPUDataManager::freeData(DeviceData *pdevData, DeviceData *d_pdevData)
{
    cudaFree( pdevData->d_NFexpForI );
    cudaFree( pdevData->d_TopNFluExpId );
    cudaFree( pdevData->d_FexpIdForI );
    cudaFree( pdevData->d_PFexpForI );
    cudaFree( pdevData->d_TopNFluExpScores );
    cudaFree( pdevData->d_pRem );
    cudaFree( pdevData->d_MatAux );
    cudaFree( pdevData->d_ones );
    cudaFree( pdevData->d_PIEst );
    cudaFree( pdevData->d_VecAux );
    cudaFree( d_pdevData );
}

GPUDataManager::~GPUDataManager()
{
}
