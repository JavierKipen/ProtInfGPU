#ifndef GPUDATAMANAGER_H
#define GPUDATAMANAGER_H

#include "DeviceData.h"
#include "DatasetMetadata.h"

using namespace std;




class GPUDataManager {  //Class to handle the data allocation and movement for the calculations.
    
    public:
        GPUDataManager();
        ~GPUDataManager();
        unsigned long maxReadsToCompute(DatasetMetadata *pDM,unsigned long nBytesLim); //Does not look at nReads in datasetMetadata. 
        bool allocateForNumberOfReads(DatasetMetadata *pDM,DeviceData *pdevData, DeviceData **d_ppdevData); //uses nReads in metadata as the numbers of reads to use in each calculation (max).
        void metadataToGPU(DatasetMetadata *pDM,DeviceData *pdevData, DeviceData *d_pdevData); //Loads in GPU the vectores that will remain unchanged with different data.
        void loadNewDataToGPU(PNewData pNewData,DeviceData *pdevData,DeviceData *d_pdevData); //Loads the data dependant variables to GPU.
        void retrieveOutput(float * updateVectorOut,DeviceData *devData); //Sends the data that wont change during the run
        void retrieveOutput(float * updateVectorOut,DeviceData *devData,unsigned long nRows);
        void freeData(DeviceData *pdevData, DeviceData *d_pdevData);
    private:
        void createOnesVec(DeviceData *pdevData);
        unsigned long onesVecLen;
    
};


#endif