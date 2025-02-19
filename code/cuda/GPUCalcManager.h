#ifndef GPUCALCMANAGER_H
#define GPUCALCMANAGER_H

#include "DeviceData.h"

#include <cublas_v2.h>
#include <map> //To map batching lengths for different proteins

#define DEFAULT_N_THREADS_PER_BLOCK 16

class GPUCalcManager {  //Class to handle the data movement (later would be also binding and data transfer to python).
    
    public:
        GPUCalcManager(); //
        ~GPUCalcManager();
    
        void init();
    
        void calculateUpdate(DeviceData *pdevData, DeviceData *d_pdevData); //Assumes data is loaded on device, then calculates the update
        unsigned int NThreadsPerBlock;
        std::map<unsigned int, unsigned long> batchingLenForNProt; //Keeps the batching lengths for the lengths of proteins
        unsigned long minBatchSize;
    private:
        DeviceData *pdevData, *d_pdevData; //p to dev data both on device and host, save on the call to pass within the functions easier.
        void sumAlphas();
        void calcAlphas();
        void PXIRelSumRows();
        void calcPXIRel();
        void calcPXgIRel();
        void calcPRem();
        unsigned long retrieveBatch(unsigned int nProt);
        cublasHandle_t cuBlasHandle;
        cublasStatus_t cuBlasStatus; //Cublas status for debugging/error printing
        
        
    
};



#endif