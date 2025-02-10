#ifndef GPUWRAPPER_H
#define GPUWRAPPER_H

#include "DatasetMetadata.h"
#include "DeviceData.h"
#include "GPUCalcManager.h"
#include "GPUDataManager.h"

//Definition of the device variables for calculation
class GPUWrapper{  //Class to handle the data movement (later would be also binding and data transfer to python).
    
    public:
        GPUWrapper();
        ~GPUWrapper();
        
        void init(DatasetMetadata *pDatasetMetadata,unsigned int deviceN);
        unsigned long maxReadsToCompute(unsigned long nBytesToUseGPU); //Given the bytes to use in the GPU, returns how many reads could be computed at the time
        void allocateWorkingMemory(unsigned long nReadsPerUpdate); //Given the amount of reads to compute at each update, allocates the working memory on device.
        void accumulateUpdates(float *updateVectorOut, PNewData pNewData);
        GPUDataManager gDM;
        GPUCalcManager gCM;
    private:
        bool allocatedData;
        vector<float> auxUpdates;
        DatasetMetadata *pDatasetMetadata; //Points to datasetMetadata.
        DeviceData devData; //devData ON THE HOST.
        DeviceData *d_pdevData; //Pointer to devData ON THE DEVICE.
        
        
    
};



#endif