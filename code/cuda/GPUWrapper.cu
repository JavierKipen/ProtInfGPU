#include "GPUWrapper.h"



GPUWrapper::GPUWrapper()
{
    allocatedData=false;
}
GPUWrapper::~GPUWrapper()
{
    if(allocatedData)
        gDM.freeData(&devData, d_pdevData);
}


void GPUWrapper::init(DatasetMetadata *pDatasetMetadata,unsigned int deviceN)
{
    cudaSetDevice(deviceN); //Sets the device to do the calculations
    gCM.init();
    this->pDatasetMetadata=pDatasetMetadata;
    auxUpdates.resize(pDatasetMetadata->nProt,0); //aux vector to sum up the updates
}
unsigned long GPUWrapper::maxReadsToCompute(unsigned long nBytesToUseGPU)
{
    return gDM.maxReadsToCompute(pDatasetMetadata,nBytesToUseGPU);
}
void GPUWrapper::allocateWorkingMemory(unsigned long nReadsPerUpdate)
{
    DatasetMetadata aux = *pDatasetMetadata;
    aux.nReadsTotal=nReadsPerUpdate; //We set this amount of reads for the calculations!
    bool allocateRes=gDM.allocateForNumberOfReads(&aux,&devData, &d_pdevData); //Allocates the memory to work with it
    if(!allocateRes)
        cout << "Could not allocate in GPU!" <<endl;
    gDM.metadataToGPU(&aux,&devData, d_pdevData); //Sends the data that wont change during the run
    allocatedData=true; //Data has been allocated, then has to be freed in destruction.
}


void GPUWrapper::accumulateUpdates(float *updateVectorOut, PNewData pNewData)
{
    gDM.loadNewDataToGPU(pNewData ,&devData, d_pdevData); //Puts the new data on device
    gCM.calculateUpdate(&devData, d_pdevData); //Calculates the new update
    gDM.retrieveOutput(auxUpdates.data(),&devData); //Retrieves the data in an auxiliary variable
    for(unsigned int i=0;i<pDatasetMetadata->nProt;i++)
        updateVectorOut[i]+=auxUpdates[i]; //Sums the updates obtained by the variable in the out vector
}